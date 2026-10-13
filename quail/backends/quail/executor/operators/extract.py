"""Run a pipeline's extractions and build their answer tables."""

import time

import numpy as np
import pyarrow as pa

from quail.backends.quail.executor.extract import (
    ExtractionStages,
    load_vocabulary,
)
from quail.backends.quail.executor.parts import (
    ExecutionPart,
    chunks,
    gpu_seconds,
    input_staging,
    settle_documents,
    shared_prefix_tree,
    stage_spans,
)
from quail.backends.quail.executor.stages import run_stages
from quail.backends.quail.executor.state import QueryExecutionState
from quail.execution.runner import NodeMetrics, NodeResult
from quail.physical import AiExtract

SPAN_TYPE = pa.struct([("start", pa.int32()), ("end", pa.int32())])


def extract_answer_table(spec, document_ids, answers, spans) -> pa.Table:
    """Return one row per document: its id, its answer, and the answer's span."""
    valid = np.fromiter((span is not None for span in spans), dtype=bool,
                        count=len(spans))
    bounds = np.zeros((len(spans), 2), dtype=np.int32)
    bounds[valid] = np.asarray([span for span in spans if span is not None],
                               dtype=np.int32).reshape(-1, 2)
    span_array = pa.StructArray.from_arrays(
        [pa.array(bounds[:, 0]), pa.array(bounds[:, 1])],
        fields=list(SPAN_TYPE), mask=pa.array(~valid))
    return pa.Table.from_arrays(
        [pa.array(document_ids, pa.int32()), pa.array(answers, pa.string()),
         span_array], names=[spec.alias, spec.name, spec.span_name])


def execute_extract(state: QueryExecutionState, node, inputs) -> NodeResult:
    """Copy each document's answer to one extraction's question."""
    (result,) = execute_extractions(state, [node], inputs).values()
    return result


def execute_extractions(state: QueryExecutionState, nodes, inputs) -> dict:
    """Run a pipeline's extractions over its documents as one stage run.

    Each document is read once and asked every node's question while
    its KV is resident. The first node is charged the documents'
    tokens, the GPU time, and the wall time; each later node is charged
    the tokens fed after the documents for its question.

    Args:
        state: The loaded model and current query state.
        nodes: The AiExtract nodes, in pipeline order.
        inputs: The documents' ids, the tokenized documents and texts
            by alias, the query's shared preamble, and the gpu_timing
            setting.

    Returns:
        Node id -> NodeResult with the node's answer table.

    Raises:
        TypeError: A node is not an extraction.
        ValueError: The table's text was not loaded.
    """
    state.gpu_timing = bool(inputs.get("gpu_timing", False))
    started = time.perf_counter()
    document_ids, stages = extract_stages(state, nodes, inputs)
    stats = {}
    fresh = 0
    spans = []
    arena = state.loaded_model.arena
    on_settled = settle_documents(arena, stages.keys, stages.prefixes,
                                  nodes[-1].keep_kv, state.chunk_tokens)

    if document_ids:
        tree = (shared_prefix_tree(nodes[0].spec.name, stages.prefixes, arena)
                if nodes[0].spec.share_prefixes else None)
        _, spans, fresh = run_stages(
            state.torch, arena, state.loaded_model.pipeline,
            stages.stages, stages.prefixes, state.chunk_tokens,
            anchor_keys=stages.keys, on_settled=on_settled,
            staging=input_staging(state), prefix_tree=tree, stats=stats,
            label=f"extract {', '.join(node.spec.name for node in nodes)}",
            appended_tokens=stages.appended_tokens)
    stages.finish()
    wall = time.perf_counter() - started
    timing = {"gpu_timing": state.gpu_timing}
    per_question = len(stages.stages) // len(nodes)

    def own(q):
        # GPU time by the rows each question's stages read in a chunk
        spans_q = stage_spans(spans, q * per_question, (q + 1) * per_question)
        return gpu_seconds(state.torch, spans_q, timing), chunks(spans_q, timing)

    return extraction_results(nodes, stages, document_ids, wall=wall,
                              fresh=fresh, own=own, stats=stats)


def extract_stages(state: QueryExecutionState, nodes, inputs) -> tuple:
    """Build the stages that ask every node's question of the documents.

    Returns:
        The documents' ids and the ExtractionStages.

    Raises:
        TypeError: A node is not an extraction.
        ValueError: The table's text was not loaded.
    """
    if any(not isinstance(node, AiExtract) for node in nodes):
        raise TypeError("an extraction pipeline holds AiExtract nodes only")
    specs = [node.spec for node in nodes]
    alias = specs[0].alias
    document_ids = [int(document) for document in inputs["document_ids"]]
    texts = inputs.get("texts", {}).get(alias)
    if texts is None:
        raise ValueError(
            f"AI.EXTRACT needs the text of {alias!r}, which the scan did "
            f"not load")
    stages = ExtractionStages(state, specs, load_vocabulary(state), document_ids,
                           inputs["documents"][alias], texts,
                           shared_head=inputs.get("pre"),
                           document_positions=inputs.get("document_positions"))
    return document_ids, stages


def extraction_results(nodes, stages, document_ids, *, wall, fresh, own,
                       stats, kept=None) -> dict:
    """Build each extraction node's answer table and metrics.

    Args:
        nodes: The AiExtract nodes, in question order.
        stages: The finished ExtractionStages.
        document_ids: The documents' ids.
        wall: The run's wall seconds, charged to the first node.
        fresh: The run's fresh tokens; the first node is charged them
            less the later questions' suffix tokens.
        own: Question index -> (its GPU seconds, its timed chunks).
        stats: The run's packing time and borrowed tokens.
        kept: The indices of the documents the answer tables hold;
            None holds every document.
    """
    counts = [stages.counts(q) for q in range(len(nodes))]
    later = sum(count["suffix_tokens"] for count in counts[1:])
    if kept is not None:
        document_ids = [document_ids[index] for index in kept]
    results = {}
    for q, node in enumerate(nodes):
        spec = node.spec
        answers, answer_spans = stages.answers(q, kept)
        table = extract_answer_table(spec, document_ids, answers, answer_spans)
        first = q == 0
        gpu_s, timed = own(q)
        results[node.node_id] = NodeResult(
            {"rows": table, f"ids:{spec.alias}": document_ids},
            NodeMetrics(
                wall_s=wall if first else 0.0,
                input_rows=len(document_ids), output_rows=len(document_ids),
                evaluated_documents=len(document_ids),
                fresh_tokens=(fresh - later if first
                              else counts[q]["suffix_tokens"]),
                gpu_s=gpu_s, chunks=timed,
                extension={"output": spec.name, "aliases": list(spec.aliases),
                           "input_rows": len(document_ids), **counts[q],
                           "borrowed_prefix_tokens": (
                               stats.get("borrowed_tokens", 0) if first else 0),
                           "pack_s": (round(stats.get("pack_s", 0.0), 3)
                                      if first else 0.0)}))
    return results


class ExtractPart(ExecutionPart):
    """The extractions that continue a filter's pipeline.

    The documents are resident from the filter's stages, so the part
    is charged only the tokens its questions feed after them.

    Args:
        state: The loaded model and current query state.
        nodes: The consecutive AiExtract members, in pipeline order.
        ids: The pipeline's document ids.
        inputs: The first node's scheduler inputs.

    Raises:
        ValueError: A document's prompt differs from the pipeline's,
            in its head or its tokens.
    """

    combine = True
    execute_group = staticmethod(execute_extractions)

    @classmethod
    def prepare(cls, state, nodes, ids, parts, inputs, context):
        """Prepare consecutive questions over the preceding operator's documents."""
        first = parts[0].node
        head = first.document_head
        return cls(state, nodes, ids, {
            "documents": context.state["docs"],
            "texts": context.state.get("texts", {}),
            "pre": list(context.state["pre"] if head is None else head)})

    def __init__(self, state, nodes, ids, inputs):
        self.nodes = list(nodes)
        self.node = self.nodes[0]
        self.ids = [int(document) for document in ids]
        _, self.extraction = extract_stages(
            state, self.nodes, {**inputs, "document_ids": self.ids})
        alias = self.node.spec.alias
        for doc, document in zip(self.extraction.docs, self.ids):
            if doc.key != (alias, document):
                raise ValueError(
                    f"AI.EXTRACT {self.node.spec.name!r} shows document "
                    f"{document} of {alias!r} unlike the pipeline before it")
        self.stages = self.extraction.stages
        self._results = {}

    @property
    def appended_tokens(self) -> int:
        return self.extraction.appended_tokens

    def finish(self, every) -> int:
        """Settle every answer; return the tokens the questions fed."""
        self.extraction.finish()
        return sum(self.extraction.counts(q)["suffix_tokens"]
                   for q in range(len(self.nodes)))

    def result(self, tokens, gpu_s, chunks, stats) -> NodeResult:
        # the answer tables hold the documents the stages before let through
        self._results = extraction_results(
            self.nodes, self.extraction, self.ids, wall=0.0, fresh=tokens,
            own=lambda q: (gpu_s, chunks) if q == 0 else (0.0, 0), stats={},
            kept=self.extraction.asked())
        return self._results[self.node.node_id]

    def results(self) -> dict:
        """Return the answer table and metrics for each extraction."""
        return self._results
