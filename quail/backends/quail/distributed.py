"""Run a graph's model groups on GPU workers that share the single-GPU executor.

Each group of model nodes, a pipeline or one node, runs on every
worker over that worker's share of the group's documents. A document
stays on the worker that holds its KV: its scan shard, its stop-key
group, or the worker that retained it in an earlier group. The
workers run the same executor as a single GPU does; the coordinator
only shards inputs and merges outputs.
"""

import time

import numpy as np

from quail.backends.quail.coordinator import keyed_shards, merge_results, shard_input
from quail.backends.quail.executor.memory import merge_memory_reports
from quail.backends.quail.executor.operators.filter import stop_groups
from quail.backends.quail.graph import graph_report
from quail.catalog.pairs import columns_key
from quail.catalog.tokens import select_documents
from quail.execution.ids import id_positions
from quail.execution.rows import Rows
from quail.execution.runner import ExecutionContext, GenericRunner, compute_subgraph
from quail.physical import AiFilter, AiJoin, AiTopK, PhysicalGraph, Scan
from quail.planner import balanced_shards


class DistributedQuailExecution:
    """Place each graph group's documents on the workers holding their KV."""

    def __init__(self, payload, graph, gpus, round_fn):
        self.payload = payload
        self.graph = graph
        self.gpus = gpus
        self.round_fn = round_fn
        self.documents = payload["documents"]
        # alias -> the worker of each document, or -1 before it is placed
        self.owners = {alias: np.full(len(documents), -1, dtype=np.int32)
                       for alias, documents in self.documents.items()}
        for node in graph.nodes:
            if isinstance(node, Scan) and len(node.shards) == gpus:
                self._place(node.alias, node.shards)
        for node in graph.nodes:
            if isinstance(node, AiFilter) and node.stop_key:
                # a per-key stop needs a key's documents on one worker
                documents = self.documents[node.alias]
                self._place(node.alias, keyed_shards(
                    stop_groups(node, payload.get("columns", {}).get(node.alias),
                                range(len(documents))),
                    [len(document) for document in documents], gpus))
        self.last_compute = compute_subgraph(graph).nodes[-1].node_id
        self.limit = (
            None if any(isinstance(node, (AiJoin, AiTopK))
                        for node in graph.nodes)
            else payload.get("filter_limit"))
        self.started = False
        self.outputs = []

    def _place(self, alias, shards):
        for worker, shard in enumerate(shards):
            self.owners[alias][np.asarray(shard, dtype=np.int64)] = worker

    def _shards(self, alias, ids):
        """Return each worker's documents among ids, placing the unplaced ones."""
        owners = self.owners[alias]
        missing = ids[owners[ids] < 0]
        shards, _ = balanced_shards(
            [len(self.documents[alias][document]) for document in missing], self.gpus)
        self._place(alias, [missing[np.asarray(shard, dtype=np.int64)]
                            for shard in shards])
        return [ids[owners[ids] == worker] for worker in range(self.gpus)]

    def execute(self, node, inputs):
        return self._execute((node,), {node.node_id: inputs}, pipeline=False)[
            node.node_id]

    def execute_pipeline(self, nodes, inputs, context):
        """Run one pipeline on every worker's share of its documents."""
        return self._execute(nodes, inputs, pipeline=True)

    def _execute(self, nodes, inputs, *, pipeline):
        first = nodes[0]
        alias = (first.document_aliases or first.spec.aliases)[0]
        source = next(
            value.ids[alias] if isinstance(value, Rows) else value
            for port in first.inputs
            if (value := inputs[first.node_id].get(port.name)) is not None
            and (port.source.port == f"rows:{alias}"
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
            # the first round carries the query; a worker binds it once
            query = {**self.payload, "documents": {
                alias: select_documents(documents, range(len(documents)))
                for alias, documents in self.documents.items()}}
            for sub in subs:
                sub["query"] = query
        started = time.perf_counter()
        outputs = self.round_fn("execute", subs)
        wall = time.perf_counter() - started
        self.started = True
        self.outputs = outputs
        for worker, output in enumerate(outputs):
            for key, document in output.get("retained", ()):
                retained = key[1] if isinstance(key, tuple) and key[0] == "extract" \
                    else key
                if retained in self.owners:
                    self.owners[retained][document] = worker
        return merge_results(nodes, outputs, alias, ids, wall)

    def report(self) -> dict:
        """Sum the workers' KV counters and merge their memory reports."""
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
                **merge_memory_reports([output["memory"] for output in self.outputs])}


def execute_distributed_graph(payload, graph: PhysicalGraph, gpus: int,
                              round_fn, device, runtimes, registry) -> dict:
    """Execute one typed Quail graph across GPU child processes."""
    execution = DistributedQuailExecution(payload, graph, gpus, round_fn)
    sources = {alias: range(len(documents))
               for alias, documents in payload["documents"].items()}
    for alias, table in payload.get("columns", {}).items():
        sources[columns_key(alias)] = table
    context = ExecutionContext(
        runtimes=runtimes,
        model_execution=execution,
        sources=sources,
        functions=registry.functions,
        run_pipeline=execution.execute_pipeline,
    )
    started = time.perf_counter()
    result = GenericRunner().run(compute_subgraph(graph), context)
    report = graph_report(graph, result, time.perf_counter() - started, device,
                          gpus=gpus, gpu_timing=payload.get("gpu_timing", False))
    report.update(execution.report())
    return report
