"""Quail runtimes for the typed model graph.

One GPU runs a region's graph. Several GPUs each run it over their
shard of one table, the region's split table, and read every other
table whole, so each row of the region is built on the GPU holding its
document of the split table.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pyarrow as pa

from quail.backends.quail.executor.memory import memory_report, merge_memory_reports
from quail.backends.quail.executor.operators.filter import stop_keys
from quail.backends.quail.retention import apply_retention
from quail.common.pairs import FIRST_DOCUMENT, columns_key
from quail.execution.finished import report_outputs
from quail.execution.metrics import (
    NodeMetrics,
    NodeResult,
    RunResult,
    scalar_node_metrics,
)
from quail.execution.runner import (
    ExecutionContext,
    GenericRunner,
    compute_subgraph,
)
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
    PortRef,
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

    The arena is emptied first: nothing survives from the last query. A
    payload's ``shard``, alias -> document ids, limits the scans of
    those tables to the ids.
    """
    arena = state["gpu"].arena
    for key in arena.resident_keys():
        arena.free_key(key)
    arena.reset_stats()
    # the peak memory reported is this query's, not the process's
    state["gpu"].torch.cuda.reset_peak_memory_stats()
    retention = payload.get("retention", {})
    apply_retention(arena, retention, retention.get("initial", {}))
    shard = payload.get("shard", {})
    sources = {alias: shard.get(alias, range(len(tokens)))
               for alias, tokens in state["documents"].items()}
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


def _run(state, payload, graph: PhysicalGraph) -> tuple:
    """Run a region on one GPU; return its RunResult, wall seconds, and context."""
    context = execution_context(state, payload, graph)
    torch = state["gpu"].torch
    arena = state["gpu"].arena
    started = time.perf_counter()
    with torch.inference_mode():
        result = GenericRunner().run(compute_subgraph(graph), context)
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    for key in arena.resident_keys():
        arena.free_key(key)
    return result, wall, context


def execute_single_graph(state, payload, graph: PhysicalGraph) -> dict:
    """Execute one Quail model graph on one GPU executor."""
    result, wall, context = _run(state, payload, graph)
    report = graph_report(graph, result, wall, state["device"],
                          gpu_timing=state.get("gpu_timing", False))
    measurements = execution_measurements(context)
    report.update(kv_manager=measurements["kv_manager"], **measurements["memory"])
    return report


def run_shard(state, payload, graph: PhysicalGraph) -> dict:
    """Run a region on one of several GPUs, over its shard of the split table.

    Returns:
        {"outputs": PortRef -> Arrow table, "metrics": node id ->
        NodeMetrics, "kv_manager": ..., "memory": ...}.
    """
    result, _, context = _run(state, payload, graph)
    return {"outputs": export_physical_outputs(compute_subgraph(graph), result),
            "metrics": {node_id: found.metrics
                        for node_id, found in result.nodes.items()},
            **execution_measurements(context)}


def region_shards(payload, graph: PhysicalGraph, gpus: int) -> list[dict]:
    """Each GPU's documents of the region's split table, as {alias: ids}.

    The split table is the anchor of the region's first AI join, else
    the table of its first model operator. The documents of one stop key
    group, or the copies of one text in a deduplicated table, go to one
    GPU. A region whose split table is a partner of an anti join runs
    on one GPU.

    Args:
        payload: The query's settings, documents, and value columns.
        graph: The region's physical graph.
        gpus: The GPU count.
    """
    nodes = graph.topological_nodes()
    joins = [node for node in nodes if isinstance(node, AiJoin)]
    readers = [node for node in nodes if node.document_aliases]
    if not readers or any(join.semantics == "anti" and joins[0].anchor in join.partners
                          for join in joins):
        return [{}]
    alias = joins[0].anchor if joins else readers[0].document_aliases[0]
    documents = payload["documents"][alias]
    groups = next((stop_keys(payload, node).of(range(len(documents)))
                   for node in nodes if isinstance(node, AiFilter)
                   and node.alias == alias and node.stop_key), None)
    if groups is None and alias in payload.get("deduplicate", {}):
        groups = payload["columns"][alias].column(FIRST_DOCUMENT).to_pylist()
    if groups is None:
        scan = next(node for node in nodes
                    if isinstance(node, Scan) and node.alias == alias)
        return [{alias: shard} for shard in scan.shards]
    # each group goes whole to the GPU with the fewest tokens so far
    tokens = {}
    for group, document in zip(groups, documents):
        tokens[group] = tokens.get(group, 0) + len(document)
    loads, placed = [0] * gpus, {}
    for group in sorted(tokens, key=lambda group: -tokens[group]):
        placed[group] = loads.index(min(loads))
        loads[placed[group]] += tokens[group]
    owners = np.asarray([placed[group] for group in groups])
    return [{alias: np.flatnonzero(owners == gpu)} for gpu in range(gpus)]


def execute_sharded_graph(payload, graph: PhysicalGraph, gpus: int,
                          round_fn, device) -> dict:
    """Run a region on several GPUs, each over its shard of the split table.

    Each GPU runs the region as one GPU does, with run_shard, and its
    rows go to the region's FinishedRows. A port's output is the union
    of the GPUs' rows, each row once; a node's metrics are their sum.

    Args:
        payload: The query's settings, documents, and value columns.
        graph: The region's physical graph.
        gpus: The GPU count.
        round_fn: Callable("execute", payloads) -> each GPU's run_shard
            result, one payload per GPU.
        device: The device spec, for the run's cost.
    """
    shards = region_shards(payload, graph, gpus)
    started = time.perf_counter()
    runs = round_fn("execute", [{**payload, "shard": shard} for shard in shards])
    wall = time.perf_counter() - started
    compute = compute_subgraph(graph)
    nodes = {}
    for node in compute.nodes:
        parts = [NodeResult({port.name: run["outputs"][ref] for port in node.outputs
                             if (ref := PortRef(node.node_id, port.name))
                             in run["outputs"]},
                            run["metrics"][node.node_id]) for run in runs]
        for part in parts:
            report_outputs(node, part)
        metrics = sum((part.metrics for part in parts), NodeMetrics())
        nodes[node.node_id] = NodeResult(
            {name: union_rows([part.outputs[name] for part in parts])
             for name in parts[0].outputs},
            replace(metrics, wall_s=max(part.metrics.wall_s for part in parts)))
    root = compute.root
    result = RunResult(nodes[root.node_id].outputs[root.port], nodes,
                       sum((node.metrics for node in nodes.values()), NodeMetrics()))
    report = graph_report(graph, result, wall, device, gpus=gpus,
                          gpu_timing=payload.get("gpu_timing", False))
    counters = {}
    for run in runs:
        for name, value in run["kv_manager"].items():
            if isinstance(value, dict):
                counts = counters.setdefault(name, {})
                for alias, count in value.items():
                    counts[alias] = counts.get(alias, 0) + count
            else:
                counters[name] = counters.get(name, 0) + value
    report.update(kv_manager=counters,
                  **merge_memory_reports([run["memory"] for run in runs]))
    return report


def union_rows(tables: list) -> pa.Table:
    """The rows of the tables, the first row of each set of document ids."""
    table = pa.concat_tables(tables, promote_options="permissive")
    ids = [field.name for field in table.schema
           if b"quail.alias" in (field.metadata or {})]
    if len(tables) == 1 or not ids:
        return table
    first = (table.select(ids)
             .append_column("position", pa.array(np.arange(table.num_rows)))
             .group_by(ids, use_threads=False).aggregate([("position", "min")]))
    return table.take(np.sort(first["position_min"].to_numpy()))


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
