"""Execute model filters and label gates."""

from quail.backends.quail.executor import loop
from quail.backends.quail.executor.parts import (
    chunks,
    execution_prefix_tree,
    gpu_seconds,
    input_staging,
    require_execution_state,
)
from quail.backends.quail.executor.stages import Stage, filter_stages
from quail.backends.quail.graph import filter_document_sink, filter_result
from quail.execution.reranker import filter_scores
from quail.execution.runner import NodeResult
from quail.execution.tokens import DocumentKeys


def execute_filter(state, node, inputs) -> NodeResult:
    """Run one model filter and collect its results."""
    require_execution_state(state)
    torch = state["torch"]
    arena = state["arena"]
    pipeline = state["pipeline"]
    async_answers = state["async_answers"]
    chunk_tokens = state["chunk_tokens"]

    document_ids = inputs["document_ids"]
    retain_survivors = inputs.get("retain_survivors", ())
    if retain_survivors is False:
        retain_survivors = ()
    # sorted admission would change which rows a limit keeps
    tree = (None if inputs.get("limit") is not None
            else execution_prefix_tree(node, inputs["documents"], arena))
    stats = {}
    answers, spans, tokens = loop.run_filter(
        torch,
        arena,
        pipeline,
        async_answers,
        inputs["documents"],
        [list(question) for question in node.question_token_ids],
        chunk_tokens,
        limit=inputs.get("limit"),
        arena_writes=node.arena_writes,
        arena_keys=DocumentKeys(node.alias, document_ids),
        retain_survivors=retain_survivors,
        document_done=inputs.get("document_done"),
        prefix_tree=tree,
        attention_mode=node.attention or None,
        stats=stats, staging=input_staging(state),
    )
    return filter_result(
        node, answers, tokens, document_ids,
        gpu_s=gpu_seconds(torch, spans, inputs),
        chunks=chunks(spans, inputs),
        borrowed_tokens=stats.get("borrowed_tokens", 0),
        pack_s=stats.get("pack_s", 0.0))


class FilterPart:
    """A filter chain's stages inside a pipeline."""

    gate = None
    result_value = None

    def __init__(self, node, ids, async_answers):
        self.node = node
        self.ids = ids
        self.stages = filter_stages(
            [list(question) for question in node.question_token_ids],
            async_answers)
        self.document_done = filter_document_sink(node, ids)
        self.answers = {}

    def finish(self, every) -> int | None:
        """Record the chain's answers; its tokens are the run's remainder."""
        for stage in every:
            for document, row in stage.items():
                self.answers.setdefault(document, []).append(int(row[0]))
        return None

    def result(self, tokens, gpu_s, chunks, stats) -> NodeResult:
        return filter_result(
            self.node, self.answers, tokens, self.ids, gpu_s=gpu_s,
            chunks=chunks, borrowed_tokens=stats.get("borrowed_tokens", 0),
            pack_s=stats.get("pack_s", 0.0))


class LabelGate:
    """A filter on a label between two stages: it gates on the label read."""

    stages = ()
    document_done = None
    result_value = None

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
                             self.source.result_value.outputs["scores"])
