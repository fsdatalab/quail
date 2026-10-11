"""Run one extraction and build its answer table."""

import time

import pyarrow as pa

from quail.backends.quail.executor.extract import ExtractStages, load_tables
from quail.backends.quail.executor.parts import input_staging
from quail.backends.quail.executor.stages import run_stages
from quail.backends.quail.executor.state import QueryExecutionState
from quail.execution.runner import NodeMetrics, NodeResult

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
    """Copy each document's answer and return the answer table.

    Args:
        state: The loaded model and current query state.
        node: The AiExtract node.
        inputs: The documents' ids, the tokenized documents and texts
            by alias, and the gpu_timing setting.

    Raises:
        ValueError: The table's text was not loaded.
    """
    spec = node.spec
    alias = spec.alias
    document_ids = [int(document) for document in inputs["document_ids"]]
    texts = inputs.get("texts", {}).get(alias)
    if texts is None:
        raise ValueError(
            f"AI.EXTRACT needs the text of {alias!r}, which the scan did "
            f"not load")
    state.gpu_timing = bool(inputs.get("gpu_timing", False))
    started = time.perf_counter()
    stages = ExtractStages(state, spec, load_tables(state), document_ids,
                           inputs["documents"][alias], texts)
    stats = {}
    fresh = 0
    spans = []
    if document_ids:
        _, spans, fresh = run_stages(
            state.torch, state.loaded_model.arena, state.loaded_model.pipeline,
            stages.stages, stages.prefixes, state.chunk_tokens,
            anchor_keys=stages.keys, staging=input_staging(state), stats=stats,
            label=f"extract {spec.name}")
    stages.finish()
    counts = stages.counts()
    table = extract_answer_table(
        spec, document_ids, [doc.answer for doc in stages.docs],
        [doc.span for doc in stages.docs])
    gpu_s = 0.0
    if state.gpu_timing:
        state.torch.cuda.synchronize()
        gpu_s = sum(start.elapsed_time(end)
                    for _, _, start, end in spans) / 1000.0
    return NodeResult({"scores": table, f"ids:{alias}": document_ids}, NodeMetrics(
        wall_s=time.perf_counter() - started,
        input_rows=len(document_ids), output_rows=len(document_ids),
        evaluated_documents=len(document_ids), fresh_tokens=fresh,
        gpu_s=gpu_s, chunks=len(spans) if state.gpu_timing else 0,
        extension={"output": spec.name, "aliases": list(spec.aliases),
                   "input_rows": len(document_ids), **counts,
                   "borrowed_prefix_tokens": stats.get("borrowed_tokens", 0),
                   "pack_s": round(stats.get("pack_s", 0.0), 3)}))
