"""Quail runtimes for the typed model graph."""

from __future__ import annotations

import json

import _quail_native

from quail.backends.quail.executor.memory import memory_report
from quail.backends.quail.retention import apply_retention
from quail.common.pairs import columns_key
from quail.common.tokens import documents_from_table
from quail.execution.metrics import NodeResult
from quail.execution.runner import (
    ExecutionContext,
    compute_subgraph,
)
from quail.execution.stages import (
    answer_hint,
    payload_first_rows,
    ran_message,
    serve_stages,
)
from quail.physical import (
    AiClassify,
    AiExtract,
    AiFilter,
    AiJoin,
    AiMap,
    AiSaeFeatures,
    AiScore,
    PhysicalGraph,
    PortRef,
)


class RegionRuntime:
    """Run one model node as a chain of its own in the model execution."""

    def execute(self, node, inputs, context) -> NodeResult:
        return context.model_execution.execute_pipeline(
            (node,), {node.node_id: inputs}, context)[node.node_id]


def quail_runtimes() -> dict:
    """Return runtimes for the Quail backend's physical nodes."""
    model_runtime = RegionRuntime()
    return {
        AiFilter.runtime_key: model_runtime,
        AiJoin.runtime_key: model_runtime,
        AiScore.runtime_key: model_runtime,
        AiClassify.runtime_key: model_runtime,
        AiExtract.runtime_key: model_runtime,
        AiSaeFeatures.runtime_key: model_runtime,
        AiMap.runtime_key: model_runtime,
    }


def record_model_result(node, result: NodeResult,
                        context: ExecutionContext) -> None:
    """Record reuse of anchor KV supplied by preceding operators."""
    state = context.state
    if isinstance(node, AiJoin) and state["join_streamed"]:
        state["kv_stats"]["join_anchor_hits"] += result.metrics.kv_hits


def execution_context(state, payload, graph: PhysicalGraph) -> ExecutionContext:
    """Bind a query's inputs, retention, and runtimes to one GPU executor.

    The arena is emptied first: nothing survives from the last query, and
    trace files it left are written.
    """
    arena = state["gpu"].arena
    arena.write_trace_files()
    for key in arena.resident_keys():
        arena.free_key(key)
    arena.reset_stats()
    # the peak memory reported is this query's, not the process's
    state["gpu"].torch.cuda.reset_peak_memory_stats()
    retention = payload.get("retention", {})
    apply_retention(arena, retention, retention.get("initial", {}))
    documents = state["documents"]
    sources = {
        alias: range(len(tokens))
        for alias, tokens in documents.items()
    }
    for alias, table in state.get("columns", {}).items():
        sources[columns_key(alias)] = table
    runtime_state = {
        **state,
        "retention": retention,
        "filter_limit": (
            None if any(isinstance(node, AiJoin)
                        for node in graph.nodes)
            else payload.get("filter_limit")
        ),
        "joins_started": False,
        "kv_stats": {
            "retained_after_filters": 0,
            "retained_pages_after_filters": 0,
            "retained_prefix_tokens_after_filters": 0,
            "join_anchor_hits": 0,
            "join_anchor_misses": 0,
        },
    }
    execution = state["gpu"].execution
    return ExecutionContext(
        runtimes=state["runtimes"],
        model_execution=execution,
        sources=sources,
        model_result=record_model_result,
        state=runtime_state,
        # an execution that runs node by node leaves the runner's chains out
        run_pipeline=(execution.execute_pipeline if execution.groups_operators
                      else None),
        deduplicate=payload.get("deduplicate", {}),
    )


def execution_measurements(context: ExecutionContext) -> dict:
    """Return one GPU's KV counters and memory report for the query so far."""
    gpu = context.state["gpu"]
    arena = gpu.arena
    return {
        "kv_manager": {
            **context.state["kv_stats"],
            "evicted_keys": arena.evicted_keys,
            "evicted_pages": arena.evicted_pages,
            "evicted_prefix_tokens": arena.evicted_prefix_tokens,
        },
        "memory": memory_report(gpu),
    }


# the tables of a runtime payload, each a mapping from alias
_PAYLOAD_TABLES = ("documents", "texts", "columns", "source_rows")


def payload_tables(tables: dict) -> dict:
    """Return the documents, texts, and columns of ``region_tables``' tables."""
    payload = {name: {} for name in _PAYLOAD_TABLES}
    for name, table in tables.items():
        kind, alias = name.split(":", 1)
        payload[kind][alias] = (
            documents_from_table(table) if kind == "documents"
            else table.column("text") if kind == "texts"
            else table.column("row") if kind == "source_rows" else table)
    return payload


def serve_region(state, payload, graph: PhysicalGraph, end, wanted,
                 hinted: dict | None = None) -> None:
    """Serve a begun region's stages on one GPU, then say what the GPU ran.

    Args:
        state: The GPU executor's state for the query.
        payload: The query's runtime payload.
        graph: The region's graph.
        end: The worker's ``WorkerEnd``, which has read the region.
        wanted: The (node, port) pairs the region's FinishedRows reads.
        hinted: Each scan alias's table name, for the tables the
            region's hint holds rows of.
    """
    context = execution_context(state, payload, graph)
    torch, arena = state["gpu"].torch, state["gpu"].arena
    try:
        with torch.inference_mode():
            results = serve_stages(
                end, context, compute_subgraph(graph), payload["stages"],
                wanted, hinted, payload_first_rows(payload) if hinted else None)
        torch.cuda.synchronize()
        # the region is timed until its end, so the files fall inside it
        arena.write_trace_files()
    finally:
        for key in arena.resident_keys():
            arena.free_key(key)
    end.ran(*ran_message(results, execution_measurements(context)))


def answer_region(hint, sinks, question: str):
    """Answer a worker's question from the region's hint and listeners.

    Args:
        hint: The region's ``InputHint``, or None.
        sinks: The progress sink and the answer sink, each or None.
        question: The question, as JSON.

    Returns:
        The answer as JSON text, or a table for a table's kept rows.
    """
    found = json.loads(question)
    if "hint" in found:
        return answer_hint(hint, found)
    for sink, name in zip(sinks, ("progress", "answer"), strict=True):
        if name in found and sink is not None:
            sink(*found[name])
    return "null"


def region_report(graph: PhysicalGraph, ran: list, wall: float, device, *,
                  gpus: int, gpu_timing: bool = False) -> dict:
    """Report a region's run from what each worker ran.

    Rust merges the workers' outputs and metrics
    (``quail::report::region_report``).

    Args:
        graph: The region's graph.
        ran: Per worker, what it ran as ``ran_message`` wrote it.
        wall: Seconds the region took.
        device: The GPUs' device spec.
        gpus: How many GPUs.
        gpu_timing: Whether to report GPU seconds and chunks.

    Returns:
        The report, with the run's output tables under "_outputs".
    """
    text, tables = _quail_native.region_report(
        graph.to_json(), ran, wall, device.usd_per_hour, gpus, gpu_timing)
    report = json.loads(text)
    outputs = {}
    for table in tables:
        metadata = dict(table.schema.metadata)
        ref = PortRef(metadata.pop(b"quail.node").decode(),
                      metadata.pop(b"quail.port").decode())
        outputs[ref] = table.replace_schema_metadata(metadata or None)
    report["_outputs"] = outputs
    return report
