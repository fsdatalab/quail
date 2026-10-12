"""Run a pipeline's extractions and build their answer tables."""

import time

import pyarrow as pa

from quail.backends.quail.executor.extract import ExtractStages, load_tables
from quail.backends.quail.executor.parts import (
    chunks,
    gpu_seconds,
    input_staging,
    stage_spans,
)
from quail.backends.quail.executor.stages import run_stages
from quail.backends.quail.executor.state import QueryExecutionState
from quail.execution.runner import NodeMetrics, NodeResult
from quail.physical import AiExtract

SPAN_TYPE = pa.struct([("start", pa.int32()), ("end", pa.int32())])


def extract_answer_table(spec, document_ids, answers, spans) -> pa.Table:
    """Return one row per document: its id, its answer, and the answer's span."""
    return pa.table({
        spec.alias: pa.array([int(d) for d in document_ids], pa.int32()),
        spec.name: pa.array(answers, pa.string()),
        spec.span_name: pa.array(
            [None if span is None else {"start": span[0], "end": span[1]}
             for span in spans], SPAN_TYPE),
    })


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
            by alias, and the gpu_timing setting.

    Returns:
        Node id -> NodeResult with the node's answer table.

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
    state.gpu_timing = bool(inputs.get("gpu_timing", False))
    started = time.perf_counter()
    stages = ExtractStages(state, specs, load_tables(state), document_ids,
                           inputs["documents"][alias], texts)
    stats = {}
    fresh = 0
    spans = []
    arena = state.loaded_model.arena
    if arena.retention_cap_pages is None:
        arena.retention_cap_pages = max(
            0, arena.n_pages - arena.pages_needed(2 * state.chunk_tokens))

    def on_settled(anchor, survived, row):
        # the document's KV stays for a later extraction of the table
        key = stages.keys[anchor]
        if arena.is_resident(key):
            arena.retain(key, len(stages.prefixes[anchor]))

    if document_ids:
        _, spans, fresh = run_stages(
            state.torch, arena, state.loaded_model.pipeline,
            stages.stages, stages.prefixes, state.chunk_tokens,
            anchor_keys=stages.keys, on_settled=on_settled,
            staging=input_staging(state), stats=stats,
            label=f"extract {', '.join(spec.name for spec in specs)}")
    stages.finish()
    wall = time.perf_counter() - started
    timing = {"gpu_timing": state.gpu_timing}
    counts = [stages.counts(q) for q in range(len(specs))]
    later = sum(count["suffix_tokens"] for count in counts[1:])
    per_question = len(stages.stages) // len(specs)
    results = {}
    for q, (node, spec) in enumerate(zip(nodes, specs)):
        answers, answer_spans = stages.answers(q)
        table = extract_answer_table(spec, document_ids, answers, answer_spans)
        first = q == 0
        # GPU time by the rows each question's stages read in a chunk
        own = stage_spans(spans, q * per_question, (q + 1) * per_question)
        results[node.node_id] = NodeResult(
            {"scores": table, f"ids:{alias}": document_ids},
            NodeMetrics(
                wall_s=wall if first else 0.0,
                input_rows=len(document_ids), output_rows=len(document_ids),
                evaluated_documents=len(document_ids),
                fresh_tokens=(fresh - later if first
                              else counts[q]["suffix_tokens"]),
                gpu_s=gpu_seconds(state.torch, own, timing),
                chunks=chunks(own, timing),
                extension={"output": spec.name, "aliases": list(spec.aliases),
                           "input_rows": len(document_ids), **counts[q],
                           "borrowed_prefix_tokens": (
                               stats.get("borrowed_tokens", 0) if first else 0),
                           "pack_s": (round(stats.get("pack_s", 0.0), 3)
                                      if first else 0.0)}))
    return results
