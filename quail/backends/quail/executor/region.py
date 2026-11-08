"""Run a region's model operators on one GPU.

A RegionExecution holds what a region's operators share on one GPU:
the arena, the documents' tokens, and the model execution that builds
the operators. The runner hands it each chain of operators whose
documents flow from one operator straight into the next, a pipeline,
and the chains run one after another: a chain starts once every input
it reads is complete, so the edge into it holds each document until
the operators before it finish. Within a chain, a document an operator
keeps goes on to the next operator's stages in the same forward passes.

Each chain runs as one StageExecutor over one document table: the
chain's first operator's documents, under the prompt head the first
node shows them with, keyed by (alias, document id), with one prefix
tree over them.

A chain whose first node reads an input as it arrives (see
quail.execution.arrivals) starts with the documents that have arrived;
the GPU loop takes the others between forward passes, and the chain
ends once the input has ended and every document left its stages.
"""

from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from itertools import groupby
from typing import Any

import numpy as np

from quail.backends.quail.executor.map_decoder import map_decoder
from quail.backends.quail.executor.model_operator import (
    close_operators,
    execution_prefix_tree,
    gated_stages,
    input_staging,
    settle_documents,
    stage_slices,
    survival,
)
from quail.backends.quail.executor.stages import Stage
from quail.common.tokens import DocumentKeys, DocumentPrefixes
from quail.execution.answer_stops import answer_stop
from quail.execution.arrivals import Arrivals
from quail.execution.finished import (
    finished_rows,
    hinted,
    hinted_out,
    hinting,
    report,
)
from quail.execution.ids import document_ids as arrow_ids
from quail.execution.ids import id_positions
from quail.execution.rows import Rows
from quail.physical import AiFilter, AiJoin


@dataclass
class RegionDocuments:
    """The documents a chain reads, as its GPU loop's anchors.

    Attributes:
        prefixes: Per document, its prompt head and tokens.
        keys: Per document, its arena key.
        tree: A PrefixTree over prefixes, or None without sharing.
    """

    prefixes: Any
    keys: Any
    tree: Any = None


@dataclass
class ChainRun:
    """How a chain's GPU loop runs, by what its operators are.

    Attributes:
        label: The progress label.
        limit: Stop once this many documents passed every operator, or None.
        attention: The attention path the plan chose, or None.
        paged: Whether documents' KV is written to arena pages.
        stop_keys: The StopKeyGroups numbering each document's stop key
            group, or None.
        stop_groups: Per document of the table, its stop key group, or
            None.
        num_key_survivors: How many documents of a group must pass.
        skip: Callable(document) -> bool, asked before a document is
            admitted, or None to skip what the region's hint drops.
    """

    label: str = ""
    limit: int | None = None
    attention: str | None = None
    paged: bool = True
    stop_keys: Any = None
    stop_groups: Any = None
    num_key_survivors: int = 1
    skip: Any = None


def anchor_port(first, inputs) -> str:
    """The name of the input port whose rows a chain's first operator reads."""
    alias = first.document_aliases[0]
    return next((port.name for port in first.inputs
                 if port.source.port == f"rows:{alias}"),
                next(iter(inputs[first.node_id])))


def pipeline_rows(first, inputs):
    """The rows, or ids, a chain's first operator reads."""
    return inputs[first.node_id][anchor_port(first, inputs)]


def pipeline_ids(first, inputs) -> np.ndarray:
    """The document ids a chain's first operator reads."""
    alias = first.document_aliases[0]
    source = pipeline_rows(first, inputs)
    return id_positions(source.ids[alias] if isinstance(source, Rows) else source)


class RegionExecution:
    """A region's model operators on one GPU, run one chain at a time.

    Args:
        model: The model execution: its ``operator_types`` map each
            node type to its ModelOperator class, and its ``query`` is
            the QueryExecutionState, or None for a model without a GPU
            loop.
        context: The ExecutionContext of the region's graph.
    """

    def __init__(self, model, context):
        self.model = model
        self.context = context
        self.nodes = ()
        self.inputs = {}
        self.operators = []
        self.executor = None
        self._ids = None
        # the input the chain reads as it arrives, or None
        self.arrivals = None
        # operators whose rows an answer finished since the last chunk
        self._answered = []

    @property
    def query(self):
        """The query's GPU state, or None for a model without a GPU loop."""
        return self.model.query

    @property
    def documents(self) -> dict:
        """The tokenized documents by alias."""
        return self.context.state["documents"]

    @property
    def one_family(self) -> bool:
        """Whether every node of the chain is of one type."""
        return all(node.type_name == self.nodes[0].type_name for node in self.nodes)

    def node_inputs(self, node) -> dict:
        """The node's input values from outside the chain, by port name."""
        return self.inputs[node.node_id]

    def document_ids(self) -> np.ndarray:
        """The document ids the chain's first operator reads."""
        if self._ids is None:
            self._ids = pipeline_ids(self.nodes[0], self.inputs)
        return self._ids

    def run(self, nodes, inputs) -> dict:
        """Run a chain of nodes over its documents.

        Args:
            nodes: The chain's nodes, in order.
            inputs: Per node id, its input values from outside the chain.

        Returns:
            Node id -> NodeResult.
        """
        from quail.backends.quail.executor.pipeline import execute_set_pipeline

        query = self.query
        self.nodes, self.inputs, self._ids = tuple(nodes), inputs, None
        self.arrivals = self._arriving()
        if query is not None:
            query.gpu_timing = bool(self.context.state.get("gpu_timing", False))
            # a set-attention model runs every pipeline as passes over
            # whole sequences; this is the one place that choice is made
            if (query.loaded_model.pipeline.whole_sequences
                    and nodes[0].document_aliases):
                return execute_set_pipeline(query, nodes, inputs, self.context)
        self.operators = self._chain(self.nodes)
        if not any(operator.stages for operator in self.operators):
            return self._run_answered()
        return self._run_stages()

    def _arriving(self) -> Arrivals | None:
        """The input the chain reads while its rows arrive, or None.

        A chain whose first node reads an input as it arrives streams
        when it runs a GPU loop and each of its nodes' operators takes
        documents while the loop runs. Otherwise the input is read to its
        end first, and the chain reads every row.
        """
        first = self.nodes[0]
        if not first.document_aliases:
            return None
        port = anchor_port(first, self.inputs)
        source = self.inputs[first.node_id][port]
        if not isinstance(source, Arrivals):
            return None
        types, query = self.model.operator_types, self.query
        if (query is not None and not query.loaded_model.pipeline.whole_sequences
                and all(types[node.type_name].streams(node) for node in self.nodes)):
            self._ids = list(range(len(source)))
            return source
        source.wait()
        self.inputs[first.node_id][port] = Rows(
            {first.document_aliases[0]: arrow_ids(range(len(source)))})
        return None

    def _chain(self, nodes) -> list:
        """Build the chain's operators, grouping nodes an operator combines."""
        types = self.model.operator_types

        def group_key(node):
            return (node.type_name,
                    None if types[node.type_name].combine else node.node_id)

        operators = []
        for _, group in groupby(nodes, group_key):
            members = list(group)
            operators.append(types[members[0].type_name].build(
                self, members, tuple(operators)))
        return operators

    def _table(self) -> RegionDocuments:
        """The chain's documents: the first operator's own, or the table's."""
        own = self.operators[0].layout(self)
        if own is not None:
            return own
        first = self.nodes[0]
        # the documents arrived so far, apart from the ids still growing
        alias, ids = first.document_aliases[0], np.array(self.document_ids())
        prefixes = DocumentPrefixes(first.document_head, self.documents[alias], ids)
        return RegionDocuments(prefixes, DocumentKeys(alias, ids),
                               execution_prefix_tree(first, prefixes,
                                                     self.query.loaded_model.arena))

    def _run_answered(self) -> dict:
        """Run a chain whose operators answer as documents are pushed.

        Each operator takes every document the one before it kept once
        that one has finished, and closes before the next starts.
        """
        operators = self.operators
        own = operators[0].layout(self)
        operators[0].push(np.arange(len(
            own.prefixes if own is not None else self.document_ids())))
        results = {}
        for operator in operators:
            operator.finish_input()
            results.update(operator.close(0, [], {}, 0))
        return results

    def send_finished(self, operator) -> None:
        """Report the operator's finished rows to the region's FinishedRows."""
        report(operator.take_finished())

    def _chain_run(self, table) -> ChainRun:
        """The GPU loop's settings for this chain."""
        first, operators = self.nodes[0], self.operators
        settings = ChainRun(
            label=f"pipeline {(first.document_aliases or ('',))[0]}",
            limit=self.context.state.get("filter_limit"),
            attention=(first.attention or None
                       if isinstance(first, (AiFilter, AiJoin)) else None))
        if self.one_family:
            settings = type(operators[0]).alone(self, operators, table, settings)
        return settings

    def _run_stages(self) -> dict:
        """Run the chain's stages in one GPU loop and close its operators."""
        table = self._table()
        return self.run_operators(self.operators, table, self._chain_run(table),
                                  keep_kv=self._keep_kv())

    def run_operators(self, operators, table, settings=None, *,
                      keep_kv=False) -> dict:
        """Run operators' stages over a document table in one GPU loop.

        Args:
            operators: The chain's operators, in order.
            table: The RegionDocuments the first operator reads.
            settings: The ChainRun, or None for a mixed chain's defaults
                with no limit.
            keep_kv: Whether documents that pass every operator keep
                their KV.

        Returns:
            Node id -> NodeResult.
        """
        from quail.backends.quail.executor.operators.map import branch_operators
        from quail.backends.quail.executor.stage_executor import StageExecutor

        state = self.query
        settings = settings or ChainRun()
        prefixes, keys, tree = table.prefixes, table.keys, table.tree
        operators = branch_operators(operators)
        stages = gated_stages(operators, HintGate() if hinting() else None)
        dependencies = {}
        for operator, (low, _) in zip(operators, stage_slices(operators)):
            if operator.answer_limits:
                prefixes, keys, tree, added = operator.branch_inputs(
                    prefixes, keys, tree, state.loaded_model.arena.page_tokens, low)
                dependencies.update(added)
        pipeline = state.loaded_model.pipeline
        # a document that arrives later may be too long for one pass,
        # which needs its KV in pages
        paged = settings.paged or self.arrivals is not None
        if pipeline.whole_sequences:
            if keep_kv:
                raise ValueError("a set-attention model keeps no document KV")
            # each request is one unpaged sequence: no pages, no borrowing
            paged, tree = False, None
        paged = paged or pipeline.needs_pages
        if keep_kv and not paged:
            raise ValueError("retaining document KV requires arena writes")
        if not paged and tree is not None and not tree.shared_tokens:
            # the unpaged path borrows nothing; a paged run keeps the tree
            # for documents that may arrive and share with these
            tree = None
        owner = next((operator for operator in operators
                      if getattr(operator, "settles_kv", False)), None)
        anchor_keys, anchor_prefixes = keys, prefixes
        if self.arrivals is not None:
            # documents that arrive join the GPU loop's anchors, so the
            # callbacks read them from the loop
            anchor_keys, anchor_prefixes = (_Anchors(self, "keys"),
                                            _Anchors(self, "prefixes"))
        settled = (owner.settle_kv if owner is not None else settle_documents(
            state.loaded_model.arena, anchor_keys, anchor_prefixes, keep_kv,
            state.chunk_tokens, dependencies))
        decoder = self._decoder(operators, len(prefixes))
        stats = {}
        skip = settings.skip or hinted_out(anchor_keys)
        self._routes = _Routes(operators)
        self._answered = []
        with ExitStack() as resources:
            for operator in operators:
                resources.enter_context(operator.execution_context())
            self.executor = StageExecutor(
                state.torch, state.loaded_model.arena, pipeline, stages, prefixes,
                state.chunk_tokens, anchor_keys=keys, on_settled=settled,
                staging=input_staging(state),
                attention_mode=next((operator.attention for operator in operators
                                     if operator.attention), settings.attention),
                prefix_tree=tree, stats=stats, limit=settings.limit,
                survival=survival(operators),
                paged=(paged or pipeline.needs_pages) and not pipeline.whole_sequences,
                unit="documents", count_answers=False, label=settings.label,
                default_attention="tree", on_chunk=self._on_chunk,
                on_answers=(None if finished_rows() is None else self._on_answers),
                stop_groups=settings.stop_groups,
                appended_tokens=max(operator.appended_tokens
                                    for operator in operators) or None,
                gpu_timing=state.gpu_timing, decoder=decoder,
                dependencies=dependencies,
                knee_rows=getattr(state, "knee_rows", 0),
                decode_cost=getattr(state, "decode_cost", None),
                num_key_survivors=settings.num_key_survivors, skip=skip,
                arrivals=(None if self.arrivals is None
                          else partial(self._arrive, operators, settings)))
            operators[0].push(np.arange(len(table.prefixes)))
            if self.arrivals is None:
                operators[0].finish_input()
            every, spans, tokens = self.executor.run()
        for operator in operators[1:]:
            operator.finish_input()
        return close_operators(operators, every, spans, tokens, stats)

    def _arrive(self, operators, settings, executor, block) -> None:
        """Hand the GPU loop the documents that have arrived.

        Once the input has ended, the loop's input and the first
        operator's end too.

        Args:
            operators: The chain's operators, in order.
            settings: The chain's ChainRun.
            executor: The StageExecutor.
            block: Wait until documents arrive or the input ends.
        """
        arrived = self.arrivals.arrive(block)
        if arrived is None:
            if self.arrivals.ended:
                executor.input_ended()
                operators[0].finish_input()
            return
        start = len(self._ids)
        self._ids.extend(arrived.tolist())
        for operator in operators:
            operator.add_documents(len(arrived))
        first = self.nodes[0]
        alias = first.document_aliases[0]
        executor.add_documents(
            DocumentPrefixes(first.document_head, self.documents[alias], arrived),
            DocumentKeys(alias, arrived),
            stop_groups=(None if settings.stop_keys is None
                         else settings.stop_keys.of(arrived)))
        operators[0].push(np.arange(start, start + len(arrived)))

    def _keep_kv(self) -> bool:
        """Whether the chain's last model node retains its documents' KV."""
        last = max(index for index, node in enumerate(self.nodes)
                   if node.backend is not None)
        return self.nodes[last].keep_kv

    def _decoder(self, operators, num_anchors):
        """The MapDecoder writing the answers of the chain's maps, or None."""
        limits = [limit for operator in operators for limit in operator.answer_limits]
        if not limits:
            return None
        expected_tokens = [expected for operator in operators
                           if operator.answer_limits
                           for expected in operator.expected_answer_tokens]
        decoder = map_decoder(self.query, limits, num_anchors, expected_tokens)
        for operator in operators:
            if operator.answer_limits:
                operator.decoder = decoder
                for q, node in enumerate(operator.nodes):
                    stop = answer_stop(node)
                    if stop is not None:
                        decoder.answer_stops[operator.first + q] = stop
        return decoder

    def _on_answers(self, anchor, stage, start, end, answers) -> None:
        """Hand an operator the answers one group of an anchor's requests got.

        An operator whose rows are an anchor's requests, such as a score
        of pairs, finishes each row with its answer, before the anchor
        settles; the region reports those rows after the chunk.
        """
        routes = self._routes
        index = routes.owner[stage]
        operator = routes.operators[index]
        if operator.answered(anchor, stage - routes.slices[index][0], start, end,
                             answers) and operator not in self._answered:
            self._answered.append(operator)

    def _on_chunk(self, transitions) -> None:
        """Settle documents in the operators they left; send finished rows.

        A document settles in an operator when it fails one of the
        operator's stages, or passes and is no longer at any of them; a
        kept document then enters the next operator.
        """
        routes, answers = self._routes, self.executor.sched.answers
        touched, self._answered = self._answered, []
        for anchor, stage, passed in transitions:
            index = routes.owner[stage]
            low, high = routes.slices[index]
            if passed and low <= self.executor.sched.stage_of(anchor) < high:
                continue
            operator = routes.operators[index]
            operator.settle(anchor, passed, answers[stage].get(anchor, []))
            if operator not in touched:
                touched.append(operator)
            if passed and index + 1 < len(routes.operators):
                routes.operators[index + 1].push([anchor])
        for operator in touched:
            self.send_finished(operator)


class HintGate:
    """Drop a document the region's hint rejects as it enters an operator's stages.

    It gates each operator's first stage after the chain's first, as a
    condition does.
    """

    answer_dependencies = ()

    def gate(self, key):
        """Return Stage.DROP when the hint rejects the key's document."""
        return Stage.DROP if hinted(key) else None

    def decide(self, keys) -> list:
        """Return the keys whose documents the hint keeps."""
        return [key for key in keys if not hinted(key)]


class _Anchors(Sequence):
    """One of the running StageExecutor's per-anchor lists, read on each access.

    Anchors that arrive extend the executor's own lists.

    Args:
        region: The RegionExecution whose executor holds the list.
        name: The executor's attribute: "keys" or "prefixes".
    """

    def __init__(self, region, name: str):
        self.region = region
        self.name = name

    def __len__(self):
        return len(getattr(self.region.executor, self.name))

    def __getitem__(self, index):
        return getattr(self.region.executor, self.name)[index]


class _Routes:
    """Which operator owns each stage of a chain's combined stage list."""

    def __init__(self, operators):
        self.operators = operators
        self.slices = stage_slices(operators)
        self.owner = [index for index, (low, high) in enumerate(self.slices)
                      for _ in range(low, high)]



def region_of(model, context) -> RegionExecution:
    """Return the context's RegionExecution for the model, created on first use."""
    region = context.state.get("region")
    if region is None or region.model is not model:
        region = context.state["region"] = RegionExecution(model, context)
    return region
