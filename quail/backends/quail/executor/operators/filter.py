"""Execute LLM filters and conditions on classification results."""

from quail.backends.quail.executor.parts import (
    ExecutionPart,
    execute_parts,
    execution_prefix_tree,
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
    state.gpu_timing = bool(inputs.get("gpu_timing", False))
    node = nodes[0]
    documents, ids = inputs["documents"], inputs["document_ids"]
    parts = [FilterPart(member, ids, state.async_answers,
                        len(state.loaded_model.pipeline.canvas_ids))
             for member in nodes]
    if len(parts) == 1 and "document_done" in inputs:
        parts[0].document_done = inputs["document_done"]
    # Sorted admission would change which rows a limit keeps.
    tree = (None if inputs.get("limit") is not None else execution_prefix_tree(
        node, documents, state.loaded_model.arena))
    return execute_parts(
        state, parts, documents, DocumentKeys(node.alias, ids),
        keep_kv=nodes[-1].keep_kv, tree=tree, attention=node.attention or "unified",
        limit=inputs.get("limit"), stop_groups=inputs.get("stop_groups"),
        paged=node.arena_writes or len(nodes) > 1,
        label=f"filter ({len(nodes)} predicates)")


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
