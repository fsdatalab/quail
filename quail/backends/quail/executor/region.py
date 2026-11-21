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
"""

from contextlib import ExitStack
from dataclasses import dataclass
from itertools import groupby
from typing import Any

import numpy as np

from quail.backends.quail.executor.map_decoder import MapDecoder
from quail.backends.quail.executor.model_operator import (
    execution_prefix_tree,
    gated_stages,
    input_staging,
    settle_documents,
    stage_slices,
    stage_spans,
    survival,
)
from quail.backends.quail.executor.stages import Stage
from quail.common.tokens import DocumentKeys, DocumentPrefixes
from quail.execution.answer_stops import answer_stop
from quail.execution.finished import (
    finished_rows,
    hinted,
    hinting,
    region_hint,
    report,
)
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
        stop_groups: Per document, its stop key group, or None.
        num_key_survivors: How many documents of a group must pass.
        skip: Callable(document) -> bool, asked before a document is
            admitted, or None to skip what the region's hint drops.
    """

    label: str = ""
    limit: int | None = None
    attention: str | None = None
    paged: bool = True
    stop_groups: Any = None
    num_key_survivors: int = 1
    skip: Any = None


def pipeline_rows(first, inputs):
    """The rows, or ids, a chain's first operator reads."""
    alias = first.document_aliases[0]
    anchor_port = next((port.name for port in first.inputs
        if port.source.port == f"rows:{alias}"), None)
    return (inputs[first.node_id][anchor_port] if anchor_port is not None
        else next(iter(inputs[first.node_id].values())))


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
        from quail.backends.quail.executor.pipeline import SetChain

        query = self.query
        if query is not None:
            query.gpu_timing = bool(self.context.state.get("gpu_timing", False))
            # a set-attention model runs every pipeline as passes over
            # whole sequences; this is the one place that choice is made
            if (query.loaded_model.pipeline.whole_sequences
                    and nodes[0].document_aliases):
                return SetChain(query, nodes, inputs, self.context).execute()
        self.nodes, self.inputs, self._ids = tuple(nodes), inputs, None
        types = self.model.operator_types

        def group_key(node):
            return (node.type_name,
                None if types[node.type_name].combine else node.node_id)

        operators = []
        for _, group in groupby(self.nodes, group_key):
            members = list(group)
            operators.append(types[members[0].type_name].build(
                self, members, tuple(operators)))
        self.operators = operators
        if not any(operator.stages for operator in self.operators):
            own = operators[0].layout(self)
            operators[0].push(np.arange(len(
                own.prefixes if own is not None else self.document_ids())))
            results = {}
            for operator in operators:
                operator.finish_input()
                results.update(operator.close(0, [], {}, 0))
            return results
        table = self._table()
        first, operators = self.nodes[0], self.operators
        kinds = [node.type_name.rsplit("ai_", 1)[-1] for node in self.nodes]
        steps = [kind if (count := len(list(group))) == 1 else f"{kind} ×{count}"
            for kind, group in groupby(kinds)]
        alias = (self.nodes[0].document_aliases or ("",))[0]
        settings = ChainRun(
            label=f"{' → '.join(steps)} over {alias}",
            limit=self.context.state.get("filter_limit"),
            attention=(first.attention or None
                if isinstance(first, (AiFilter, AiJoin)) else None))
        if self.one_family:
            settings = type(operators[0]).alone(self, operators, table, settings)
        last = max(index for index, node in enumerate(self.nodes)
            if node.backend is not None)
        return self.run_operators(self.operators, table, settings,
            keep_kv=self.nodes[last].keep_kv)

    def _table(self) -> RegionDocuments:
        """The chain's documents: the first operator's own, or the table's."""
        own = self.operators[0].layout(self)
        if own is not None:
            return own
        first = self.nodes[0]
        alias, ids = first.document_aliases[0], self.document_ids()
        prefixes = DocumentPrefixes(first.document_head, self.documents[alias], ids)
        return RegionDocuments(prefixes, DocumentKeys(alias, ids),
            execution_prefix_tree(first, prefixes,
                self.query.loaded_model.arena))

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
        from quail.backends.quail.executor.operators import map
        from quail.backends.quail.executor.stage_executor import StageExecutor

        state = self.query
        settings = settings or ChainRun()
        prefixes, keys, tree = table.prefixes, table.keys, table.tree
        combined = []
        for answers, group in groupby(operators,
            lambda operator: isinstance(operator, map.MapOperator)):
            members = list(group)
            combined.extend(
                [map.AnswerBranches(members)] if answers and len(members) > 1
                else members)
        operators = combined
        stages = gated_stages(operators, HintGate() if hinting() else None)
        dependencies = {}
        for operator, (low, _) in zip(operators, stage_slices(operators)):
            if operator.answer_limits:
                prefixes, keys, tree, added = operator.branch_inputs(
                    prefixes, keys, tree, state.loaded_model.arena.page_tokens, low)
                dependencies.update(added)
        pipeline = state.loaded_model.pipeline
        paged = settings.paged
        if pipeline.whole_sequences:
            if keep_kv:
                raise ValueError("a set-attention model keeps no document KV")
            # each request is one unpaged sequence: no pages, no borrowing
            paged, tree = False, None
        paged = paged or pipeline.needs_pages
        if keep_kv and not paged:
            raise ValueError("retaining document KV requires arena writes")
        owner = next((operator for operator in operators
            if getattr(operator, "settles_kv", False)), None)
        settled = (owner.settle_kv if owner is not None else settle_documents(
            state.loaded_model.arena, keys, prefixes, keep_kv, state.chunk_tokens,
            dependencies))
        decoder = self._decoder(operators, len(prefixes))
        stats = {}
        skip = settings.skip or (
            (lambda anchor: hinted(keys[anchor])) if hinting() else None)
        self._routes = _Routes(operators)
        self._answered = []

        def on_answers(anchor, stage, start, end, answers) -> None:
            """Hand an operator one request group's answers."""
            routes = self._routes
            index = routes.owner[stage]
            operator = routes.operators[index]
            if operator.answered(anchor, stage - routes.slices[index][0], start, end,
                answers) and operator not in self._answered:
                self._answered.append(operator)

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
                paged=paged, label=settings.label, on_chunk=self._on_chunk,
                on_answers=(None if finished_rows() is None else on_answers),
                stop_groups=settings.stop_groups,
                appended_tokens=max(operator.appended_tokens
                    for operator in operators) or None,
                gpu_timing=state.gpu_timing, decoder=decoder,
                dependencies=dependencies,
                knee_rows=getattr(state, "knee_rows", 0),
                decode_cost=getattr(state, "decode_cost", None),
                num_key_survivors=settings.num_key_survivors, skip=skip,
                retains_kv=keep_kv or owner is not None)
            operators[0].push(np.arange(len(table.prefixes)))
            operators[0].finish_input()
            every, spans, tokens = self.executor.run()
        for operator in operators[1:]:
            operator.finish_input()
        slices = stage_slices(operators)
        own = [operator.read_answers(every[low:high])
            for operator, (low, high) in zip(operators, slices)]
        staged = [index for index, operator in enumerate(operators)
            if operator.stages]
        remainder = tokens - sum(own[index] or 0 for index in staged[1:])
        results = {}
        for index, (operator, (low, high)) in enumerate(zip(operators, slices)):
            results.update(operator.close(
                remainder if staged and index == staged[0] else (own[index] or 0),
                stage_spans(spans, low, high),
                stats if staged and index == staged[0] else {},
                low))
        return results

    def _decoder(self, operators, num_anchors):
        """The MapDecoder writing the answers of the chain's maps, or None."""
        limits = [limit for operator in operators for limit in operator.answer_limits]
        if not limits:
            return None
        expected_tokens = [expected for operator in operators
            if operator.answer_limits
            for expected in operator.expected_answer_tokens]
        state = self.query
        from quail.backends.quail.executor.model_loading import full_output_head
        from quail.common.budgets import copy_verification_workspace_bytes

        decoder = MapDecoder(
            state.torch, full_output_head(state.loaded_model.model),
            num_anchors, limits, state.loaded_model.tokenizer.stop_ids,
            state.loaded_model.arena.page_tokens,
            expected_tokens=expected_tokens,
            workspace_bytes=copy_verification_workspace_bytes(
                state.loaded_model.model_spec, state.chunk_tokens),
            row_budget=state.chunk_tokens)
        decoder.tokenizer = state.loaded_model.tokenizer
        for operator in operators:
            if operator.answer_limits:
                operator.decoder = decoder
                for q, node in enumerate(operator.nodes):
                    stop = answer_stop(node, region_hint())
                    if stop is not None:
                        decoder.answer_stops[operator.first + q] = stop
        return decoder

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
            report(operator.take_finished())


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


class _Routes:
    """Which operator owns each stage of a chain's combined stage list."""

    def __init__(self, operators):
        self.operators = operators
        self.slices = stage_slices(operators)
        self.owner = [index for index, (low, high) in enumerate(self.slices)
            for _ in range(low, high)]
