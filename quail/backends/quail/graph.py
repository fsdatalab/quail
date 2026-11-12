"""Quail runtimes for the typed model graph."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time

import _quail_native
import pyarrow as pa

from quail.backends.quail.executor.memory import memory_report, merge_memory_reports
from quail.backends.quail.executor.operators.filter import stop_keys
from quail.backends.quail.retention import apply_retention
from quail.common.pairs import FIRST_DOCUMENT, columns_key
from quail.execution.finished import finished_rows
from quail.execution.metrics import NodeResult, scalar_node_metrics
from quail.execution.runner import (
    ExecutionContext,
    GenericRunner,
    compute_subgraph,
)
from quail.execution.stages import merged_run, serve_stages
from quail.execution.types import export_physical_outputs
from quail.physical import (
    AiClassify,
    AiExtract,
    AiFilter,
    AiInference,
    AiJoin,
    AiMap,
    AiSaeFeatures,
    AiScore,
    Barrier,
    PhysicalGraph,
    Scan,
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


def execute_single_graph(state, payload, graph: PhysicalGraph) -> dict:
    """Execute one Quail model graph on one GPU executor."""
    context = execution_context(state, payload, graph)
    torch = state["gpu"].torch
    arena = state["gpu"].arena
    started = time.perf_counter()
    with torch.inference_mode():
        result = GenericRunner().run(compute_subgraph(graph), context)
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    arena.write_trace_files()
    for key in arena.resident_keys():
        arena.free_key(key)
    report = graph_report(graph, result, wall, state["device"],
                          gpu_timing=state.get("gpu_timing", False))
    measurements = execution_measurements(context)
    report.update(kv_manager=measurements["kv_manager"], **measurements["memory"])
    return report


def serve_region(state, payload, graph: PhysicalGraph, server, wanted) -> dict:
    """Serve a region's stages on one GPU until the region ends.

    Returns:
        The results of each node the GPU ran under "results", beside the
        GPU's KV counters and memory report.
    """
    context = execution_context(state, payload, graph)
    torch, arena = state["gpu"].torch, state["gpu"].arena
    try:
        with torch.inference_mode():
            results = serve_stages(server, context, compute_subgraph(graph),
                                   payload["stages"], wanted)
        torch.cuda.synchronize()
        # execute_staged_graph times the region, so the files fall inside it
        arena.write_trace_files()
    finally:
        for key in arena.resident_keys():
            arena.free_key(key)
    return {"results": results, **execution_measurements(context)}


def execute_staged_graph(payload, graph: PhysicalGraph, gpus: int, serve,
                         device) -> dict:
    """Run a region's stages on one server per GPU and report the run.

    Each row a stage finishes goes to the region's FinishedRows, when
    the region reports to one.

    Args:
        payload: The query's runtime payload, with the plan's stages
            under "stages".
        graph: The region's graph.
        gpus: How many GPUs, one server each.
        serve: Callable(paths, wanted) that starts a server listening at
            each socket path and returns an owned handle. Its ``finish``
            waits for each server's result; ``abort`` stops every server
            without waiting indefinitely for a coordinator connection.
            wanted lists the (node, port) pairs FinishedRows reads.
        device: The GPUs' device spec.
    """
    compute = compute_subgraph(graph)
    rows = finished_rows()
    wanted = [] if rows is None else [
        (node.node_id, port.name) for node in compute.nodes
        for port in node.outputs if rows.wants(node.node_id, port.name)]

    def report(table: pa.Table) -> None:
        if rows is not None:
            rows.report(table)

    # No listener exists until Python preparation and native decoding succeed.
    graph_json = compute.to_json()
    stages_json = json.dumps(payload["stages"])
    groupings = stage_groupings(graph, payload)
    _quail_native.validate_stages(graph_json, stages_json, groupings)
    started = time.perf_counter()
    # the tables the stages share are files beside the sockets; in
    # /dev/shm a file is memory only
    memory = "/dev/shm" if os.path.isdir("/dev/shm") else None
    with tempfile.TemporaryDirectory(dir=memory, prefix="quail-") as directory:
        paths = [os.path.join(directory, f"gpu{index}.sock") for index in range(gpus)]
        servers = serve(paths, wanted)
        try:
            _quail_native.run_stages(graph_json, stages_json, groupings, paths, report)
            outputs = servers.finish()
        except BaseException:
            # A listener may never have received its coordinator connection.
            # Receiving its result here would wait forever in accept().
            with contextlib.suppress(BaseException):
                servers.abort()
            raise
    wall = time.perf_counter() - started
    result = merged_run(compute, [output["results"] for output in outputs])
    report = graph_report(graph, result, wall, device, gpus=gpus,
                          gpu_timing=payload.get("gpu_timing", False))
    report.update(
        kv_manager={name: _summed([output["kv_manager"][name] for output in outputs])
                    for name in outputs[0]["kv_manager"]},
        **merge_memory_reports([output["memory"] for output in outputs]))
    return report


def _summed(values: list):
    """Sum counters, and counters by alias alias by alias."""
    if isinstance(values[0], dict):
        return {key: sum(value.get(key, 0) for value in values)
                for key in set().union(*values)}
    return sum(values)


def stage_groupings(graph: PhysicalGraph, payload) -> dict:
    """Return, by alias, lists of each document's group for the stages' hash.

    One list holds each document's first copy of its text, one each stop
    key's group numbers; Rust's ``run_stages`` keys a document by the
    smallest id it is grouped with.
    """
    columns = payload.get("columns", {})
    scans = {node.alias: node.num_documents for node in graph.nodes
             if isinstance(node, Scan)}
    groupings = {alias: [columns[alias].column(FIRST_DOCUMENT).to_pylist()]
                 for alias in payload.get("deduplicate", {})}
    for node in graph.nodes:
        if isinstance(node, AiFilter) and node.stop_key:
            groupings.setdefault(node.alias, []).append(
                list(stop_keys(payload, node).of(range(scans[node.alias]))))
    return groupings


def graph_report(graph, result, wall, device, *, gpus=1, gpu_timing=False):
    """Collect graph outputs, execution metrics, and GPU cost."""
    filters, joins = model_answers(graph, result)
    report = {
        "filters": filters,
        "joins": joins,
        "_outputs": export_physical_outputs(compute_subgraph(graph), result),
        "wall_s": round(wall, 2),
        "fresh_tokens": result.metrics.fresh_tokens,
        "cached_tokens": result.metrics.cached_tokens,
        "evaluated_documents": result.metrics.evaluated_documents,
        "evaluated_document_pairs": result.metrics.evaluated_document_pairs,
        "usd_per_query": (
            None if device.usd_per_hour is None
            else wall / 3600 * gpus * device.usd_per_hour
        ),
        "backend_metrics": {"inferences": [
            dict(value.metrics.extension)
            for node_id, value in result.nodes.items()
            if isinstance(graph.node(node_id), AiInference)
        ]},
        "node_metrics": scalar_node_metrics(result.nodes),
        "executed_join_plan": executed_join_plan(graph),
    }
    report.update(throughput(graph, result.metrics, wall))
    if gpu_timing:
        report["gpu_s"] = round(result.metrics.gpu_s, 3)
        report["chunks"] = result.metrics.chunks
    return report


def throughput(graph, metrics, seconds: float) -> dict:
    """Return the run's throughput under one key.

    A run that evaluated document pairs reports evaluated pairs per
    second. Any other run reports input document rows, summed over
    every scan and counted before filters, per second.
    """
    seconds = max(seconds, 1e-9)
    if metrics.evaluated_document_pairs:
        return {
            "document_pairs_per_second":
                metrics.evaluated_document_pairs / seconds,
        }
    documents = sum(
        node.num_documents for node in graph.nodes if isinstance(node, Scan)
    )
    return {"documents_per_second": documents / seconds}


def model_answers(graph, result) -> tuple[dict, list]:
    """Collect predicate answers in execution order."""
    filters, joins = {}, []
    for node in graph.topological_nodes():
        if isinstance(node, AiFilter):
            filters.setdefault(node.alias, []).append(
                result.nodes[node.node_id].outputs[f"filter_answers:{node.alias}"])
        elif isinstance(node, AiJoin):
            joins.append(result.nodes[node.node_id].outputs[
                f"join_answers:{node.written_position}"])
    return {alias: pa.concat_tables(tables) for alias, tables in filters.items()}, joins


# the fields the report shows of each executed join and barrier
_JOIN_FIELDS = {
    AiJoin.type_name: (
        "written_position", "exec_index", "anchor", "partners", "semantics",
        "selectivity", "expected_rows", "anchor_frame_tokens", "pair_tail_tokens",
        "anchor_resident", "tuple_tokens", "pairs_from", "keep_kv", "attention",
        "share_prefixes"),
    Barrier.type_name: ("next_anchor", "aliases"),
}


def executed_join_plan(graph) -> list[dict]:
    """Describe the join nodes executed from the saved graph."""
    return [
        {"type": node.type_name, "id": node.node_id,
         **{name: getattr(node, name) for name in _JOIN_FIELDS[node.type_name]}}
        for node in graph.topological_nodes()
        if node.type_name in _JOIN_FIELDS
    ]
