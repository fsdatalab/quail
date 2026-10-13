"""Execute LLM filters and conditions on classification results."""

from quail.backends.quail.executor import loop
from quail.backends.quail.executor.parts import (
    ExecutionPart,
    execution_prefix_tree,
    gpu_seconds,
    input_staging,
    stage_spans,
)
from quail.backends.quail.executor.stages import Stage, filter_stages
from quail.backends.quail.executor.state import QueryExecutionState
from quail.backends.quail.graph import filter_document_sink, filter_result
from quail.execution.reranker import filter_scores
from quail.execution.runner import NodeResult
from quail.execution.tokens import DocumentKeys


def execute_filter(state: QueryExecutionState, node, inputs) -> NodeResult:
    """Run one model filter and collect its results."""
    return execute_filters(state, (node,), inputs)[node.node_id]


def execute_filters(state: QueryExecutionState, nodes, inputs) -> dict:
    """Run filter predicates together while each document retains its KV."""
    node = nodes[0]
    torch = state.torch
    arena = state.loaded_model.arena
    pipeline = state.loaded_model.pipeline
    async_answers = state.async_answers
    chunk_tokens = state.chunk_tokens

    document_ids = inputs["document_ids"]
    retain_survivors = True if nodes[-1].keep_kv else ()
    # sorted admission would change which rows a limit keeps
    tree = (None if inputs.get("limit") is not None
            else execution_prefix_tree(node, inputs["documents"], arena))
    sinks = [filter_document_sink(member, document_ids) for member in nodes]

    def document_done(finished):
        for j, sink in enumerate(sinks):
            if sink is not None:
                rows = [(document, 0, passed if stage == j else True)
                        for document, stage, passed in finished if stage >= j]
                if rows:
                    sink(rows)

    stats = {}
    answers, spans, tokens = loop.run_filter(
        torch,
        arena,
        pipeline,
        async_answers,
        inputs["documents"],
        [list(member.question_token_ids) for member in nodes],
        chunk_tokens,
        limit=inputs.get("limit"),
        stop_groups=inputs.get("stop_groups"),
        arena_writes=node.arena_writes or len(nodes) > 1,
        arena_keys=DocumentKeys(node.alias, document_ids),
        retain_survivors=retain_survivors,
        document_done=(inputs.get("document_done") if len(nodes) == 1
                       else document_done),
        prefix_tree=tree,
        attention_mode=node.attention or None,
        stats=stats, staging=input_staging(state),
        gpu_timing=bool(inputs.get("gpu_timing")),
    )
    counts = [sum(len(row) > j for row in answers.values())
              for j in range(len(nodes))]
    own = [count * (len(member.question_token_ids) + len(pipeline.canvas_ids))
           for count, member in zip(counts, nodes)]
    own[0] = tokens - sum(own[1:])
    results = {}
    for j, member in enumerate(nodes):
        rows = {document: [row[j]] for document, row in answers.items()
                if len(row) > j}
        own_spans = stage_spans(spans, j, j + 1)
        result = filter_result(
            member, rows, own[j], document_ids,
            gpu_s=gpu_seconds(own_spans),
            chunks=len(own_spans),
            borrowed_tokens=stats.get("borrowed_tokens", 0) if j == 0 else 0,
            pack_s=stats.get("pack_s", 0.0) if j == 0 else 0.0)
        results[member.node_id] = result
    return results


class FilterPart(ExecutionPart):
    """Filter stages and result tables within a pipeline."""

    execute_group = staticmethod(execute_filters)

    @classmethod
    def prepare(cls, state, nodes, ids, parts, inputs, context):
        """Prepare one filter's requests and answer notifications."""
        return cls(nodes[0], ids, state.async_answers,
                   len(state.loaded_model.pipeline.canvas_ids))

    def __init__(self, node, ids, async_answers, canvas_tokens=0):
        self.node = node
        self.ids = ids
        self.canvas_tokens = canvas_tokens
        self.stages = filter_stages(
            [list(node.question_token_ids)],
            async_answers)
        self.document_done = filter_document_sink(node, ids)
        self.answers = {}

    def finish(self, every) -> int | None:
        """Record filter answers and leave token accounting to the pipeline."""
        for stage in every:
            for document, row in stage.items():
                self.answers.setdefault(document, []).append(int(row[0]))
        return len(self.answers) * (len(self.node.question_token_ids)
                                    + self.canvas_tokens)

    def result(self, tokens, gpu_s, chunks, stats) -> NodeResult:
        return filter_result(
            self.node, self.answers, tokens, self.ids, gpu_s=gpu_s,
            chunks=chunks, borrowed_tokens=stats.get("borrowed_tokens", 0),
            pack_s=stats.get("pack_s", 0.0))


class LabelGate(ExecutionPart):
    """Pipeline condition that keeps documents with selected categories."""

    stages = ()

    @classmethod
    def prepare(cls, state, nodes, ids, parts, inputs, context):
        """Bind a label condition to its classification and input rows."""
        from quail.backends.quail.executor.operators.classify import ClassifyPart

        node = nodes[0]
        labeled = next(part for part in reversed(parts)
                       if isinstance(part, ClassifyPart)
                       and part.node.spec.name == node.column)
        source = next(part for part in parts
                      if part.node.node_id == node.inputs[0].source.node_id)
        return cls(node, labeled, source)

    def __init__(self, node, labeled, source):
        self.node = node
        self.labeled = labeled
        self.source = source
        self.predicate = node.predicate
        self.position = {document: index
                         for index, document in enumerate(labeled.ids)}

    def gate(self, key):
        label = self.labeled.plan.labels[self.position[key[1]]]
        return None if self.predicate.accepts(label) else Stage.DROP

    def finish(self, every):
        return None

    def result(self, tokens, gpu_s, chunks, stats) -> NodeResult:
        return filter_scores(self.node,
                             self.source.result_value.outputs["rows"])
