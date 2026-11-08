"""Model operators: the nodes of a region that read documents through the model.

A ModelOperator runs one physical node, or several consecutive nodes of
one kind, over the documents a RegionExecution pushes into it. An
operator with stages describes its model work as Stage objects; the
region runs every operator's stages in one GPU loop, a document moving
to the next operator's stages when it settles in this one. An operator
without stages, such as a condition, decides at the next operator's
first stage.
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import replace
from itertools import groupby
from typing import Any, ClassVar

from quail.backends.quail.executor.chunk import InputStaging
from quail.backends.quail.executor.stages import Stage, frame_writes
from quail.backends.quail.executor.state import QueryExecutionState
from quail.common.expressions import effective_selectivity
from quail.common.tokens import prefix_tree
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
        "prefix sharing for %s: %s documents reuse %s tokens of earlier "
        "documents (found in %.2f s)", name, f"{len(prefixes):,}",
        f"{tree.shared_tokens:,}", time.perf_counter() - started)
    return tree


def input_staging(state: QueryExecutionState):
    """Return reusable transfer buffers with an empty fixed-token cache."""
    if state.loaded_model.input_staging is None:
        state.loaded_model.input_staging = InputStaging(state.torch)
    state.loaded_model.input_staging.fixed_tokens.clear()
    return state.loaded_model.input_staging


def prior_inputs(node, operators, inputs):
    """Return the preceding operators and external row tables a node reads."""
    by_node = {member.node_id: operator for operator in operators
               for member in operator.nodes}
    priors, prior_tables = [], []
    for port in node.inputs:
        if port.value_type is ValueType.ROWS:
            if port.source.node_id in by_node:
                priors.append((by_node[port.source.node_id], port.source))
            else:
                prior_tables.append(inputs[port.name])
    return priors, prior_tables


def join_prior_rows(rows, alias, priors, prior_tables):
    """Join columns from preceding operators onto the surviving documents."""
    prior_results = (operator.results()[port.node_id].outputs[port.port]
                     for operator, port in priors)
    for prior in (*prior_tables, *prior_results):
        rows = rows.joined(prior, alias)
    return rows


class ModelOperator(ABC):
    """Physical nodes run over the documents a region pushes into them.

    The region pushes documents into the first operator of a chain and
    pushes each document an operator keeps into the next. It settles a
    document in an operator once the document leaves the operator's
    stages, passed or dropped, and takes the rows each operator finished
    after every forward pass, so a region can send them on. Once no
    document can arrive any more, it ends the operator's input; once
    the GPU loop is done, it reads the answers and closes the operator,
    which returns each node's result.

    combine groups consecutive nodes of one type into one operator;
    prepare_group prepares stages across consecutive operators of one
    type.
    """

    stages: Sequence[Stage] = ()
    # the physical nodes the operator executes, every member of a group
    nodes: Sequence = ()
    # the computed columns a later condition may test
    tested_columns: tuple = ()
    combine: ClassVar[bool] = False
    appended_tokens = 0
    # the frame tokens the operator's first stage writes; 0 when the
    # stage before it left the same frame
    frame_tokens = 0
    attention = None
    # the answer limits of the maps the operator runs, in order; the
    # region's MapDecoder writes their answers into operator.decoder
    answer_limits = ()
    answer_dependencies = ()
    # the documents pushed into the operator so far
    pushed = 0

    @classmethod
    def build(cls, region, nodes, chain):
        """Build the operator for nodes after the operators of chain.

        Args:
            region: The RegionExecution running the chain.
            nodes: The physical nodes the operator runs.
            chain: The operators before it, in order.
        """
        raise NotImplementedError(f"{cls.__name__} is not built by a region")

    @classmethod
    def alone(cls, region, operators, table, settings):
        """Return the GPU loop's settings for a chain of this type alone.

        Args:
            region: The RegionExecution running the chain.
            operators: The chain's operators, every one of this type.
            table: The chain's RegionDocuments.
            settings: The ChainRun a mixed chain runs with.
        """
        return settings

    @staticmethod
    def prepare_group(operators):
        """Prepare stages shared by consecutive operators of the same type."""

    def layout(self, region):
        """Return the operator's own (prefixes, keys, prefix tree), or None.

        None, the default, runs the region's document table. An
        operator that shows its documents in its own prompt layout,
        such as a score of pairs, returns that layout when it is first.
        """
        return None

    def execution_context(self):
        """Hold resources needed while the GPU loop runs this operator."""
        return nullcontext()

    def push(self, documents) -> None:
        """Take documents, by their positions in the region's table."""
        self.pushed += len(documents)

    def settle(self, document, survived: bool, row) -> None:
        """Record that a document left this operator, kept or dropped.

        Args:
            document: The document's position in the region's table.
            survived: Whether the operator kept it.
            row: Its answer row at the operator's last stage it reached.
        """

    def answered(self, anchor, stage, start, end, answers) -> bool:
        """Take the answers one group of an anchor's requests got.

        Args:
            anchor: The anchor's position in the region's table.
            stage: The index of the stage among the operator's stages.
            start: The first request of the group.
            end: The request after its last.
            answers: The group's answers, one per request.

        Returns:
            Whether the answers finished rows that take_finished returns.
        """
        return False

    def take_finished(self) -> list:
        """Return the rows the operator finished since the last call.

        Each is a table finished_table tagged with the node and port
        that wrote it: a node's ids, one column per alias, or a join's
        anchor and partner ids, and the answer columns of those rows.
        Empty when the operator finished none, or reports its rows only
        with its result.
        """
        return []

    def finish_input(self) -> None:
        """Record that no more documents will be pushed."""

    @abstractmethod
    def read_answers(self, every) -> int | None:
        """Read the GPU loop's answers; return the operator's own fresh tokens.

        Args:
            every: Per stage of the operator, document -> its answers.
        """

    @abstractmethod
    def close(self, tokens, spans, stats, offset) -> dict:
        """Build each node's result from the work the region measured.

        Args:
            tokens: The fresh tokens charged to the operator.
            spans: The timing spans of the chunks that packed its stages.
            stats: The run's packing time and borrowed tokens, for the
                first operator with stages; empty for the others.
            offset: The index of the operator's first stage in the run.

        Returns:
            Node id -> NodeResult.
        """

    @abstractmethod
    def results(self) -> dict:
        """Return the results close built, by node id."""


class ColumnOperator:
    """Settled output columns available to conditions in a chain."""

    @property
    def column_sources(self) -> dict:
        """Map each computed column to its node and value indices."""
        return {column: (index, field)
                for index, node in enumerate(self.nodes)
                for field, column in enumerate(
                    node.spec.output_columns[len(node.spec.aliases):])}

    @property
    def tested_columns(self) -> tuple[str, ...]:
        return tuple(self.column_sources)

    def tested_values(self, column, positions):
        """Read one settled column at positions among the operator's documents."""
        index, field = self.column_sources[column]
        return self.column_values(index, positions)[field]

    @abstractmethod
    def column_values(self, index, positions) -> tuple:
        """Read a node's settled arrays in its output-column order."""


class SingleNodeOperator(ModelOperator):
    """An operator that runs one physical node and produces one result."""

    # the physical node the operator executes
    node = None
    result_value = None

    @property
    def nodes(self) -> tuple:
        """Return the operator's physical node, or nothing without one."""
        return () if self.node is None else (self.node,)

    @abstractmethod
    def result(self, tokens, gpu_s, chunks, stats) -> Any:
        """Build the result with its measured execution work."""

    def close(self, tokens, spans, stats, offset) -> dict:
        """Build the node's result from its measured work."""
        self.result_value = self.result(
            tokens, gpu_seconds(spans), len(spans), stats)
        return self.results()

    def results(self) -> dict:
        """Return the result keyed by physical node ID."""
        return {self.node.node_id: self.result_value}


def operator_label(word: str, nodes) -> str:
    """Return a progress label naming a run's operators, three at most."""
    names = [node.spec.name for node in nodes]
    head, dot, _ = names[0].partition(".")
    if dot and all(name.startswith(head + ".") for name in names):
        # fields of one struct, such as contract.parties
        word, names = f"{word} {head}:", [name[len(head) + 1:] for name in names]
    shown = ", ".join(names[:3])
    if len(names) > 3:
        shown += f" and {len(names) - 3} more"
    return f"{word} {shown}"


def _prepare(gates, prepare):
    """Return a Stage.prepare that decides keys at each gate in turn.

    A gate decides only the keys every gate before it kept, as the
    stage's requests ask them one at a time.
    """
    def prepared(keys):
        for gate in gates:
            keys = gate.decide(keys)
        if prepare is not None:
            prepare(keys)

    return prepared


def gated_stages(operators, gate=None) -> list:
    """Combine operator stages and attach conditions to the stages after them.

    Args:
        operators: The chain's operators in execution order. Operators
            without stages contribute a condition checked before the
            next stage, or before the document completes when no stage
            follows.
        gate: A condition checked before each operator's first stage
            after the chain's first stage, such as the region's hint,
            or None.

    Returns:
        The combined Stage list, with conditions attached to stage requests.
    """
    for implementation, group in groupby(operators, type):
        implementation.prepare_group(list(group))
    stages = []
    pending = []
    dependencies = []
    for operator in operators[1:]:
        if operator.stages and not operator.stages[0].rewind:
            # an operator's first request follows the document, in
            # place of whatever the operator before it appended, as
            # the next question of an extraction does
            operator.stages = [replace(operator.stages[0], rewind=True),
                               *operator.stages[1:]]
    for operator in operators:
        if not operator.stages:
            pending.append(operator)
            dependencies.extend(operator.answer_dependencies)
            continue
        first = operator.stages[0]
        gates = tuple(pending) + ((gate,) if gate is not None and stages else ())
        if pending:
            first.answer_dependencies = tuple(dict.fromkeys(dependencies))
            dependencies = []
        if gates:
            asked = first.suffixes if first.own_suffixes else first.requests

            def guarded(key, gates=gates, asked=asked):
                for condition in gates:
                    if condition.gate(key) is Stage.DROP:
                        return Stage.DROP
                return None if asked is None else asked(key)

            if first.own_suffixes:
                first.suffixes = guarded
            else:
                first.requests = guarded
            first.prepare = _prepare(gates, first.prepare)
            pending = []
        stages.extend(operator.stages)
    if pending and stages:
        def finish(key, gates=tuple(pending)):
            for gate in gates:
                if gate.gate(key) is Stage.DROP:
                    return Stage.DROP
            return Stage.SKIP

        # a document that passes the trailing conditions skips this
        # stage and counts as a survivor, with nothing asked
        terminal = Stage(suffixes=[], readout=stages[-1].readout, requests=finish,
                         answer_dependencies=tuple(dict.fromkeys(dependencies)),
                         prepare=_prepare(tuple(pending), None))
        operators[-1].stages = [terminal]
        stages.append(terminal)
    writes = frame_writes(stages)
    for operator, (low, high) in zip(operators, stage_slices(operators)):
        operator.frame_tokens = (len(stages[low].frame)
                                 if low < high and writes[low] else 0)
    return stages


def stage_slices(operators) -> list:
    """Each operator's (first, end) stage indices in the combined stage list."""
    offset = 0
    slices = []
    for operator in operators:
        slices.append((offset, offset + len(operator.stages)))
        offset += len(operator.stages)
    return slices


def close_operators(operators, every, spans, tokens, stats) -> dict:
    """Read each operator's answers and build its nodes' results.

    The first operator with stages receives the document token costs,
    the prefix-sharing counts, and the packing time. Later operators
    receive their own frame and suffix token costs. GPU time is
    apportioned by the rows each operator processed.

    Args:
        operators: The chain's operators in execution order.
        every: Answer mappings for all stages.
        spans: Per-chunk row counts and timing events.
        tokens: Fresh token count of the whole run.
        stats: Packing time and prefix-sharing counts.

    Returns:
        Node id -> NodeResult, for every operator's nodes.
    """
    slices = stage_slices(operators)
    own = [operator.read_answers(every[low:high])
           for operator, (low, high) in zip(operators, slices)]
    staged = [index for index, operator in enumerate(operators) if operator.stages]
    remainder = tokens - sum(own[index] or 0 for index in staged[1:])
    results = {}
    for index, (operator, (low, high)) in enumerate(zip(operators, slices)):
        results.update(operator.close(
            remainder if staged and index == staged[0] else (own[index] or 0),
            stage_spans(spans, low, high),
            stats if staged and index == staged[0] else {},
            low))
    return results


def survival(operators) -> float:
    """The fraction of documents expected to pass every operator."""
    fraction = 1.0
    for operator in operators:
        for node in operator.nodes:
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


def settle_documents(arena, keys, prefixes, keep_kv: bool, budget: int,
                     dependencies=()):
    """Return a callback that applies the final operator's KV retention setting."""
    if keep_kv and arena.retention_cap_pages is None:
        arena.retention_cap_pages = max(
            0, arena.num_pages - arena.pages_needed(2 * budget))

    def settled(anchor, survived, row):
        key = keys[anchor]
        if not arena.is_resident(key):
            return
        if survived and keep_kv and anchor not in dependencies:
            arena.retain(key, len(prefixes[anchor]))
        else:
            arena.free_key(key)

    return settled
