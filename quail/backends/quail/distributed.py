"""Partition model graph groups across workers running the shared executor."""

import time

import numpy as np

from quail.backends.quail.coordinator import keyed_shards, merge_results, shard_input
from quail.backends.quail.executor.operators.filter import stop_groups
from quail.backends.quail.graph import graph_report
from quail.catalog.pairs import columns_key
from quail.catalog.tokens import select_documents
from quail.execution.ids import id_positions
from quail.execution.rows import Rows
from quail.execution.runner import ExecutionContext, GenericRunner, compute_subgraph
from quail.physical import AiFilter, AiJoin, PhysicalGraph, Scan
from quail.planner import balanced_shards


class DistributedQuailExecution:
    """Place graph groups near resident KV and merge their typed results."""

    def __init__(self, payload, graph, gpu_count, round_fn):
        self.payload = payload
        self.graph = graph
        self.gpu_count = gpu_count
        self.round_fn = round_fn
        self.docs = payload["docs"]
        self.owners = {alias: np.full(len(docs), -1, dtype=np.int32)
                       for alias, docs in self.docs.items()}
        for node in graph.nodes:
            if isinstance(node, Scan) and len(node.shards) == gpu_count:
                self._place(node.alias, node.shards)
        for node in graph.nodes:
            if isinstance(node, AiFilter) and node.stop_key:
                docs = self.docs[node.alias]
                self._place(node.alias, keyed_shards(
                    stop_groups(node, payload.get("columns", {}).get(node.alias),
                                range(len(docs))),
                    [len(document) for document in docs], gpu_count))
        self.last_compute = compute_subgraph(graph).nodes[-1].node_id
        self.limit = (None if any(isinstance(node, AiJoin) for node in graph.nodes)
                      else payload.get("filter_limit"))
        self.started = False
        self.outputs = []

    def _place(self, alias, shards):
        for worker, shard in enumerate(shards):
            self.owners[alias][np.asarray(shard, dtype=np.int64)] = worker

    def _shards(self, alias, ids):
        owners = self.owners[alias]
        missing = ids[owners[ids] < 0]
        shards, _ = balanced_shards(
            [len(self.docs[alias][document]) for document in missing], self.gpu_count)
        self._place(alias, [missing[np.asarray(shard, dtype=np.int64)]
                            for shard in shards])
        return [ids[owners[ids] == worker] for worker in range(self.gpu_count)]

    def execute(self, node, inputs):
        return self._execute((node,), {node.node_id: inputs}, pipeline=False)[
            node.node_id]

    def execute_pipeline(self, nodes, inputs, context):
        """Run the same pipeline on each worker's document shard."""
        return self._execute(nodes, inputs, pipeline=True)

    def _execute(self, nodes, inputs, *, pipeline):
        first = nodes[0]
        alias = (first.document_aliases or first.spec.aliases)[0]
        source = next(value.ids[alias] if isinstance(value, Rows) else value
                      for port in first.inputs
                      if (value := inputs[first.node_id][port.name]) is not None
                      and (port.source.port == f"ids:{alias}"
                           or isinstance(value, Rows) and alias in value.ids))
        ids = id_positions(source)
        final = any(node.node_id == self.last_compute for node in nodes)
        subs = []
        for shard in self._shards(alias, ids):
            subs.append({
                "nodes": nodes, "pipeline": pipeline, "final": final,
                "limit": self.limit if final else None,
                "inputs": {
                    node.node_id: {
                        port.name: shard_input(port, inputs[node.node_id][port.name],
                                               alias, shard)
                        for port in node.inputs if port.name in inputs[node.node_id]}
                    for node in nodes},
            })
        if not self.started:
            query = {**self.payload, "docs": {
                alias: select_documents(documents, range(len(documents)))
                for alias, documents in self.docs.items()}}
            for sub in subs:
                sub["query"] = query
        started = time.perf_counter()
        outputs = self.round_fn("execute", subs)
        wall = time.perf_counter() - started
        self.started = True
        self.outputs = outputs
        for worker, output in enumerate(outputs):
            for key, document in output.get("retained", ()):
                alias_key = key[1] if isinstance(key, tuple) and key[0] == "extract" \
                    else key
                if alias_key in self.owners:
                    self.owners[alias_key][document] = worker
        return merge_results(nodes, outputs, alias, ids, wall)

    def report(self):
        """Sum the latest query counters and report peak memory per GPU."""
        totals = {}
        for output in self.outputs:
            for name, value in output["kv_manager"].items():
                if isinstance(value, dict):
                    counts = totals.setdefault(name, {})
                    for alias, count in value.items():
                        counts[alias] = counts.get(alias, 0) + count
                else:
                    totals[name] = totals.get(name, 0) + value
        return {"kv_manager": totals,
                "peak_gib": max((output["peak_gib"] for output in self.outputs),
                                default=0.0)}


def execute_distributed_graph(payload, graph: PhysicalGraph, gpu_count: int,
                              round_fn, device, runtimes,
                              registry) -> dict:
    """Execute a graph with model groups partitioned across GPU workers."""
    execution = DistributedQuailExecution(
        payload, graph, gpu_count, round_fn)
    sources = {alias: range(len(documents))
               for alias, documents in payload["docs"].items()}
    sources.update({columns_key(alias): table
                    for alias, table in payload.get("columns", {}).items()})
    context = ExecutionContext(
        runtimes=runtimes, model_execution=execution, sources=sources,
        functions=registry.functions, run_pipeline=execution.execute_pipeline)
    started = time.perf_counter()
    result = GenericRunner().run(compute_subgraph(graph), context)
    report = graph_report(graph, result, time.perf_counter() - started, device,
                          workers=gpu_count,
                          gpu_timing=payload.get("gpu_timing", False))
    return {**report, **execution.report()}
