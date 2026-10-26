"""Compose operator stages and collect their execution metrics."""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from itertools import groupby
from typing import Any, ClassVar

import numpy as np

from quail.backends.quail.executor.chunk import InputStaging
from quail.backends.quail.executor.stages import Stage, frame_writes, run_stages
from quail.backends.quail.executor.state import QueryExecutionState
from quail.catalog.tokens import prefix_tree
from quail.execution.rows import Rows
from quail.logical import effective_selectivity
from quail.physical import AiFilter, AiJoin, Filter, ValueType
from quail.progress import logger


def shared_prefix_tree(name, prefixes, arena):
    """Build the prefix tree of an operator's document prefixes.

    Args:
        name: The operator's output name, for the log line.
        prefixes: Token sequences, one per document.
        arena: KVArena whose page size the shares round down to.

    Returns:
        The PrefixTree over prefixes.
    """
    started = time.perf_counter()
    tree = prefix_tree(prefixes, arena.page_tokens)
    logger.info(
        "prefix sharing on %s: %s documents borrow %s tokens "
        "(tree built in %.2f s)", name, len(prefixes),
        tree.shared_tokens, time.perf_counter() - started)
    return tree


def input_staging(state: QueryExecutionState):
    """Return reusable transfer buffers with an empty fixed-token cache."""
    if state.loaded_model.input_staging is None:
        state.loaded_model.input_staging = InputStaging(state.torch)
    state.loaded_model.input_staging.fixed_tokens.clear()
    return state.loaded_model.input_staging


def prior_inputs(node, parts, inputs):
    """Return preceding parts and external row tables read by a node."""
    by_node = {member.node_id: part for part in parts for member in part.nodes}
    priors, prior_tables = [], []
    for port in node.inputs:
        if port.value_type is ValueType.ROWS:
            if port.source.node_id in by_node:
                priors.append((by_node[port.source.node_id], port.source))
            else:
                prior_tables.append(inputs[port.name])
    return priors, prior_tables


def join_prior_rows(rows: Rows, alias, priors, prior_tables) -> Rows:
    """Join columns from preceding operators onto the surviving documents."""
    prior_results = (part.results()[port.node_id].outputs[port.port]
                     for part, port in priors)
    for prior in (*prior_tables, *prior_results):
        rows = rows.joined(prior, alias)
    return rows


class ExecutionPart(ABC):
    """Operator work prepared for the shared document scheduler.

    Implementations prepare stages, finish their answers, and return node
    results. combine groups consecutive nodes into one part. prepare_group
    prepares stages across consecutive parts of the same type. execute_group
    optionally handles a run containing only that operator type.
    """

    stages: Sequence[Stage]
    # the physical nodes the part executes, every member of a group
    nodes: Sequence = ()
    # the computed columns a later condition may test
    tested_columns: tuple = ()
    combine: ClassVar[bool] = False
    execute_group: ClassVar[Callable | None] = None
    gate = None
    document_done = None
    result_value = None
    appended_tokens = 0
    # the frame tokens the part's first stage writes; 0 when the stage
    # before it left the same frame
    frame_tokens = 0
    on_settled = None
    attention = None

    @classmethod
    def prepare(cls, state, nodes, ids, parts, inputs, context):
        """Prepare physical nodes for execution through an operator adapter."""
        raise NotImplementedError(f"{cls.__name__} needs prepared requests")

    @staticmethod
    def model_inputs(node, inputs, context, upstream=None, nodes=None):
        """Turn the node's graph input ports into scheduler inputs.

        Returns None for an operator that reads its port values as
        they are.

        Args:
            node: The physical node.
            inputs: Port values from the preceding operators.
            context: The execution context with the documents.
            upstream: The preceding pipeline state of a join, or None.
            nodes: The nodes sharing one scheduler run, or None.
        """
        return None

    @staticmethod
    def prepare_group(parts):
        """Prepare stages shared by consecutive parts of the same type."""

    def execution_context(self):
        """Hold resources needed while the scheduler runs this part."""
        return nullcontext()

    @abstractmethod
    def finish(self, every) -> int | None:
        """Finalize answers and return this part's suffix token count."""

    @abstractmethod
    def complete(self, tokens, spans, stats, offset) -> None:
        """Build results from the scheduler's token counts and timing spans."""

    @abstractmethod
    def results(self) -> dict:
        """Return completed results keyed by node or output name."""


class DocumentAnswerPart(ExecutionPart):
    """A part over one table's documents, fed their ids or an operator's rows."""

    @staticmethod
    def model_inputs(node, inputs, context, upstream=None, nodes=None):
        if nodes is not None:
            inputs = inputs[node.node_id]
        # the one input carries the documents' ids, or the rows of an
        # earlier operator over them
        (value,) = inputs.values()
        ids = (value.ids[node.spec.alias].to_numpy() if isinstance(value, Rows)
               else np.asarray(value, dtype=np.int64))
        return {"document_ids": ids,
                "documents": context.state["documents"],
                "texts": context.state.get("texts", {}),
                "limit": context.state.get("filter_limit")}


class SingleResultPart(ExecutionPart):
    """Prepared work whose execution produces one result."""

    # the physical node the part executes; None for a standalone batch
    node = None

    @property
    def nodes(self) -> tuple:
        """Return the part's physical node, or nothing for a standalone batch."""
        return () if self.node is None else (self.node,)

    @abstractmethod
    def result(self, tokens, gpu_s, chunks, stats) -> Any:
        """Build the result with its measured execution work."""

    def complete(self, tokens, spans, stats, offset):
        """Build the operator's result from its measured work."""
        self.result_value = self.result(
            tokens, gpu_seconds(spans), len(spans), stats)

    def results(self) -> dict:
        """Return results keyed by physical node ID."""
        return {self.node.node_id: self.result_value}


def execute_parts(state, parts, prefixes, anchor_keys, *, keep_kv, prefix_tree=None,
                  attention=None, label="", limit=None, stop_groups=None,
                  paged=True, on_answers=None) -> dict:
    """Run prepared operator work together and collect per-node results.

    A limit ends the run once that many documents passed every part;
    documents are admitted only while the ones in flight are expected
    to fall short of it, by the parts' selectivities. When one part's
    KV grows as it writes, its StageGrowth sizes each document's pages
    as it goes.
    """
    stages = gated_stages(parts)
    growth = growing_stages(parts)
    paged = paged or state.loaded_model.pipeline.needs_pages
    if keep_kv and not paged:
        raise ValueError("retaining document KV requires arena writes")
    if prefix_tree is not None and not prefix_tree.shared_tokens:
        prefix_tree = None
    settled = next((part.on_settled for part in parts
                    if part.on_settled is not None), None)
    if settled is None:
        settled = settle_documents(state.loaded_model.arena, anchor_keys, prefixes,
                                   keep_kv, state.chunk_tokens)
    stats = {}
    with ExitStack() as resources:
        for part in parts:
            resources.enter_context(part.execution_context())
        every, spans, tokens = run_stages(
            state.torch, state.loaded_model.arena, state.loaded_model.pipeline,
            stages, prefixes, state.chunk_tokens, anchor_keys=anchor_keys,
            on_settled=settled, prefix_tree=prefix_tree, stats=stats,
            gpu_timing=state.gpu_timing,
            attention_mode=next((part.attention for part in parts if part.attention),
                                attention),
            staging=input_staging(state), label=label, limit=limit,
            survival=survival(parts),
            stop_groups=stop_groups, paged=paged,
            on_chunk=transition_reporter(parts), on_answers=on_answers,
            appended_tokens=max(part.appended_tokens for part in parts) or None,
            growth=growth)
    complete_parts(parts, every, spans, tokens, stats)
    return {node_id: result for part in parts
            for node_id, result in part.results().items()}


def growing_stages(parts):
    """Return the StageGrowth of the one part whose KV grows as it writes.

    Returns None unless exactly one part grows and no other part's
    stage repeats: the other parts' requests are then sized when a
    document is admitted.
    """
    growing = [part for part in parts
               if getattr(getattr(part, "answer_stages", None), "grows", False)]
    if len(growing) != 1:
        return None
    if any(stage.repeats for part in parts if part is not growing[0]
           for stage in part.stages):
        return None
    return growing[0].answer_stages


def gated_stages(parts) -> list:
    """Combine operator stages and attach conditions to the stages after them.

    Args:
        parts: Pipeline parts in execution order. Parts without stages
            contribute a condition checked before the next stage, or
            before the document completes when no stage follows.

    Returns:
        The combined Stage list, with conditions attached to stage requests.
    """
    for implementation, group in groupby(parts, type):
        implementation.prepare_group(list(group))
    stages = []
    pending = []
    for part in parts[1:]:
        if part.stages and not part.stages[0].rewind:
            # an operator's first request follows the document, in
            # place of whatever the operator before it appended, as
            # the next question of an extraction does
            part.stages = [replace(part.stages[0], rewind=True), *part.stages[1:]]
    for part in parts:
        if not part.stages:
            pending.append(part.gate)
            continue
        first = part.stages[0]
        if pending:
            gates = list(pending)
            asked = first.requests

            def guarded(key, gates=gates, asked=asked):
                for gate in gates:
                    if gate(key) is Stage.DROP:
                        return Stage.DROP
                return None if asked is None else asked(key)

            first.requests = guarded
            pending = []
        stages.extend(part.stages)
    if pending and stages:
        def finish(key, gates=tuple(pending)):
            for gate in gates:
                if gate(key) is Stage.DROP:
                    return Stage.DROP
            return Stage.SKIP

        # a document that passes the trailing conditions skips this
        # stage and counts as a survivor, with nothing asked
        terminal = Stage(suffixes=[], readout=stages[-1].readout, requests=finish)
        parts[-1].stages = [terminal]
        stages.append(terminal)
    writes = frame_writes(stages)
    for part, (low, high) in zip(parts, _part_slices(parts)):
        part.frame_tokens = len(stages[low].frame) if low < high and writes[low] else 0
    return stages


def _part_slices(parts) -> list:
    offset = 0
    slices = []
    for part in parts:
        slices.append((offset, offset + len(part.stages)))
        offset += len(part.stages)
    return slices


def transition_reporter(parts):
    """Build a stage-indexed callback for operators that report completions."""
    listeners = []
    owners = []
    for part, (low, high) in zip(parts, _part_slices(parts)):
        if part.document_done is None:
            owners.extend([None] * (high - low))
            continue
        index = len(listeners)
        listeners.append(part.document_done)
        owners.extend([(index, low)] * (high - low))
    if not listeners:
        return None

    def report(transitions):
        finished = [[] for _ in listeners]
        for anchor, stage, passed in transitions:
            owner = owners[stage]
            if owner is not None:
                index, low = owner
                finished[index].append((anchor, stage - low, passed))
        for listener, records in zip(listeners, finished):
            if records:
                listener(records)

    return report


RUN_STATS = ("host_s", "run_s", "graph_replays", "readout_gpu_s", "idle_gpu_s",
             "kv_growth")


def complete_parts(parts, every, spans, tokens, stats):
    """Finalize pipeline parts and store each operator's result.

    The first part with stages receives document token costs, prefix-sharing
    counts, and packing time. Later parts receive their own frame and suffix
    token costs, and the run's RUN_STATS. GPU time is apportioned by the
    rows each part processed.

    Args:
        parts: Pipeline parts in execution order.
        every: Answer mappings for all stages.
        spans: Per-chunk row counts and timing events.
        tokens: Fresh token count excluding the final consumer's own work.
        stats: Packing time and prefix-sharing counts.
    """
    slices = _part_slices(parts)
    # what the run measured as a whole, reported by every part
    shared = {key: stats[key] for key in RUN_STATS if key in stats}
    own = [part.finish(every[low:high])
           for part, (low, high) in zip(parts, slices)]
    staged = [index for index, part in enumerate(parts) if part.stages]
    remainder = tokens - sum(own[index] or 0 for index in staged[1:])
    for index, (part, (low, high)) in enumerate(zip(parts, slices)):
        part_spans = stage_spans(spans, low, high)
        part.complete(
            remainder if staged and index == staged[0] else (own[index] or 0),
            part_spans, stats if staged and index == staged[0] else shared,
            low)


def survival(parts) -> float:
    """The fraction of documents expected to pass every part, by selectivity."""
    fraction = 1.0
    for part in parts:
        for node in part.nodes:
            if isinstance(node, (AiFilter, AiJoin, Filter)):
                fraction *= effective_selectivity(node.selectivity)
    return fraction


def execution_prefix_tree(node, documents, arena):
    """Build the requested prefix tree, or return None if sharing is disabled."""
    if not node.share_prefixes:
        return None
    return shared_prefix_tree(node.document_aliases[0], documents, arena)


def stage_spans(spans, low, high) -> list:
    """Select chunk timing spans covering a range of stages.

    Args:
        spans: Tuples of stage row counts, chunk tokens, and CUDA events.
        low: First included stage index.
        high: First excluded stage index.

    Returns:
        Spans with only the selected stage row counts. Total chunk token
        counts remain unchanged for proportional GPU time accounting.
    """
    narrowed = []
    for by_stage, tokens, start, end in spans:
        rows = {j: n for j, n in by_stage.items() if low <= j < high}
        if rows:
            narrowed.append((rows, tokens, start, end))
    return narrowed


def gpu_seconds(spans) -> float:
    """Calculate GPU seconds attributable to the selected stage rows.

    Args:
        spans: Stage row counts and completed CUDA timing events per chunk.
            Empty when GPU timing is disabled.
    """
    return sum(
        start.elapsed_time(end) * min(1.0, sum(rows.values()) / tokens)
        for rows, tokens, start, end in spans if tokens) / 1000.0


def settle_documents(arena, keys, prefixes, keep_kv: bool, budget: int):
    """Return a callback that applies the final operator's KV retention setting."""
    if keep_kv and arena.retention_cap_pages is None:
        arena.retention_cap_pages = max(
            0, arena.num_pages - arena.pages_needed(2 * budget))

    def settled(anchor, survived, row):
        key = keys[anchor]
        if not arena.is_resident(key):
            return
        # only an (alias, document) key names KV a later operator
        # reads; a branch's key holds one answer and is freed
        if survived and keep_kv and isinstance(key, tuple) and len(key) == 2:
            arena.retain(key, len(prefixes[anchor]))
        else:
            arena.free_key(key)

    return settled


class AnswerStream:
    """Send documents' finished answers to a callback, each document once.

    Args:
        on_answers: Callable(document positions, values), or None to send
            nothing.
    """

    def __init__(self, on_answers):
        self.on_answers = on_answers
        self.sent = set()

    def chunk(self, value_of):
        """Return an on_chunk callback for run_stages, or None.

        Args:
            value_of: Callable(document position) -> its answer, or None
                while it has none.
        """
        if self.on_answers is None:
            return None
        return lambda transitions: self.send(
            sorted({anchor for anchor, _, _ in transitions}), value_of)

    def send(self, anchors, value_of):
        """Send the answers of the anchors that have one and were not sent."""
        ready = []
        for anchor in anchors:
            value = None if anchor in self.sent else value_of(anchor)
            if value is not None:
                ready.append((anchor, value))
        if ready:
            self.sent.update(anchor for anchor, _ in ready)
            self.on_answers(np.asarray([anchor for anchor, _ in ready]),
                            [value for _, value in ready])

    def finish(self, anchors, value_of):
        """Send every remaining answer after the run."""
        if self.on_answers is not None:
            self.send(anchors, value_of)
