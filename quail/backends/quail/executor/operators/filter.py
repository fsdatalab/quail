"""Execute LLM filters and conditions on classification results."""

from quail.backends.quail.executor.parts import (
    SingleResultPart,
    execute_parts,
    execution_prefix_tree,
)
from quail.backends.quail.executor.stages import Stage, filter_stages
from quail.backends.quail.executor.state import QueryExecutionState
from quail.backends.quail.reranker import filter_scores
from quail.catalog.tokens import (
    DocumentKeys,
    DocumentPrefixes,
)
from quail.execution.runner import (
    NodeMetrics,
    NodeResult,
)
from quail.progress import answer_sink


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


class FilterPart(SingleResultPart):
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
        self.frame_tokens = 0
        self.stages = filter_stages(
            [list(node.question_token_ids)],
            async_answers)
        self.document_done = filter_document_sink(node, ids)
        self.answers = {}

    @staticmethod
    def prepare_group(parts):
        """Retain the common question preamble across consecutive filters."""
        stages = filter_stages(
            [part.node.question_token_ids for part in parts],
            parts[0].stages[0].readout)
        for index, (part, stage) in enumerate(zip(parts, stages)):
            part.stages = [stage]
            part.frame_tokens = len(stage.frame) if index == 0 else 0

    def finish(self, every) -> int | None:
        """Record filter answers and leave token accounting to the pipeline."""
        for stage in every:
            for document, row in stage.items():
                self.answers.setdefault(document, []).append(int(row[0]))
        return len(self.answers) * (len(self.stages[0].suffixes[0])
                                    + self.frame_tokens
                                    + self.canvas_tokens)

    def result(self, tokens, gpu_s, chunks, stats) -> NodeResult:
        return filter_result(
            self.node, self.answers, tokens, self.ids, gpu_s=gpu_s,
            chunks=chunks, borrowed_tokens=stats.get("borrowed_tokens", 0),
            pack_s=stats.get("pack_s", 0.0))


class LabelGate(SingleResultPart):
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


def filter_result(node, answers, tokens, document_ids,
                  gpu_s: float = 0.0, chunks: int = 0,
                  borrowed_tokens: int = 0, pack_s: float = 0.0) -> NodeResult:
    """Build a filter result using global document IDs and execution metrics.

    Args:
        node: Physical filter node.
        answers: Mapping from local document index to Boolean stage answers.
        tokens: Fresh tokens processed by the model.
        document_ids: Global document IDs in local index order.
        gpu_s: Forward-pass GPU seconds.
        chunks: Number of timed forward passes.
        borrowed_tokens: Prefix tokens read from another document's KV.
        pack_s: Host seconds spent building forward chunks.

    Returns:
        A NodeResult containing surviving IDs, filter answers, and metrics.
    """
    global_answers = {
        document_ids[int(local)]: row
        for local, row in answers.items()
    }
    survivors = sorted(
        document
        for document, row in global_answers.items()
        if len(row) == 1 and all(row)
    )
    return NodeResult(
        outputs={
            f"ids:{node.alias}": survivors,
            f"filter_answers:{node.alias}": global_answers,
        },
        metrics=NodeMetrics(
            input_rows=len(document_ids),
            output_rows=len(survivors),
            evaluated_documents=len(global_answers),
            fresh_tokens=tokens,
            gpu_s=gpu_s,
            chunks=chunks,
            extension={
                **({"borrowed_prefix_tokens": borrowed_tokens}
                   if borrowed_tokens else {}),
                **({"pack_s": round(pack_s, 3)} if pack_s else {}),
            },
        ),
    )


def filter_document_sink(node, document_ids):
    """The chain's document_done callback, or None when nobody listens.

    Reports each chunk's finished documents to the answer sink as
    ``{"kind": "filter", "node", "alias", "stages", "documents"}`` where
    ``documents`` lists ``[row index, last stage asked, passed]`` per
    document. A document that passed its last stage survived the filter;
    every other listed document failed at the stage given.
    """
    sink = answer_sink()
    if sink is None:
        return None
    stages = 1

    def document_done(finished):
        sink({"kind": "filter", "node": node.node_id, "alias": node.alias,
              "stages": stages,
              "documents": [[int(document_ids[position]), stage, passed]
                            for position, stage, passed in finished]})

    return document_done


def stop_groups(node, columns, document_ids) -> list | None:
    """Return each document's stop key group, or None without a stop key.

    Documents with equal values in the node's ``stop_key`` columns
    share a group number; the scheduler reads one document of a group
    at a time and skips the rest once one survives.

    Args:
        node: The AiFilter.
        columns: The alias's value table, one row per document.
        document_ids: The documents' positions in that table.
    """
    if not node.stop_key:
        return None
    if columns is None:
        raise KeyError(
            f"the stop key of {node.alias!r} needs its value columns")
    import pyarrow as pa

    positions = pa.array([int(document) for document in document_ids],
                         pa.int64())
    values = zip(*(columns.column(name).take(positions).to_pylist()
                   for name in node.stop_key))
    groups = {}
    return [groups.setdefault(value, len(groups)) for value in values]


def filter_inputs(state, node, document_ids) -> dict:
    """Scheduler inputs for one filter chain over the given documents."""
    return {
        "documents": DocumentPrefixes(
            state["pre"], state["docs"][node.alias], document_ids
        ),
        "document_ids": document_ids,
        "limit": state["filter_limit"],
        "stop_groups": stop_groups(
            node, state.get("columns", {}).get(node.alias), document_ids),
        "document_done": filter_document_sink(node, document_ids),
    }
