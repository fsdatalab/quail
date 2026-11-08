"""Run a region's stages on several GPUs, one partition per GPU.

Rust's ``quail::stages`` runs partition p of each stage on the server of
GPU p: a process, or a test's thread, holding every document and one
execution context for the region. Over a Unix socket, Rust sends a
stage's input rows; the server runs the stage's nodes with the
single-GPU runner and sends back the rows later stages read and the
rows it finished for the region's FinishedRows.
"""

from __future__ import annotations

import json
import traceback

import _quail_native
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from quail.execution.finished import reporting
from quail.execution.metrics import NodeMetrics, NodeResult, RunResult
from quail.execution.rows import Rows
from quail.execution.runner import ExecutionContext, GenericRunner
from quail.execution.types import output_table
from quail.physical import PhysicalGraph, PortRef, Scan, ValueType


def serve_stages(server, context: ExecutionContext, graph: PhysicalGraph,
                 wanted) -> dict:
    """Run each stage Rust sends until the region ends.

    Args:
        server: The ``StageServer`` listening for the region's stages.
        context: The region's execution context on this GPU.
        graph: The region's compute graph.
        wanted: The (node, port) pairs the region's FinishedRows reads.

    Returns:
        Node id -> NodeResult, of Arrow tables, of each node this server ran.
    """
    stages = json.loads(_quail_native.region_stages(graph.to_json()))
    wanted = {tuple(pair) for pair in wanted}
    results = {}
    while (request := server.next_stage()) is not None:
        stage, tables = request
        try:
            outputs, finished, ran = run_stage(context, graph, stages[stage], tables,
                                               wanted)
        except Exception:
            server.fail(traceback.format_exc())
            continue
        results.update(ran)
        server.reply(outputs, finished)
    return results


class _StageRows:
    """The rows a stage's nodes finish, kept for the region's FinishedRows."""

    def __init__(self, nodes, wanted):
        self.nodes, self.wanted, self.tables = nodes, wanted, []

    def wants(self, node: str, port: str) -> bool:
        return node in self.nodes and (node, port) in self.wanted

    def report(self, table: pa.Table) -> None:
        self.tables.append(table)


def run_stage(context: ExecutionContext, graph: PhysicalGraph, stage: dict,
              tables: list, wanted: set) -> tuple[list, list, dict]:
    """Run one stage's nodes over its input rows.

    Args:
        context: The region's execution context on this GPU.
        graph: The region's compute graph.
        stage: The stage, as ``region_stages`` writes it.
        tables: Each stage input's rows, in the stage's input order.
        wanted: The (node, port) pairs the region's FinishedRows reads.

    Returns:
        The tables of the stage's outputs, in its output order; the
        tables its nodes finished, tagged with node and port; and node
        id -> NodeResult of the scans it read and the nodes it ran.
    """
    by_id = {node.node_id: node for node in graph.nodes}
    inside, outside = set(stage["nodes"]), set()
    pending = list(inside)
    while pending:
        for port in by_id[pending.pop()].inputs:
            if port.source.node_id not in inside | outside:
                outside.add(port.source.node_id)
                pending.append(port.source.node_id)
    # a node outside the stage counts as run; only the ports the stage
    # reads hold rows
    initial = {PortRef(node_id, port.name): None for node_id in outside
               for port in by_id[node_id].outputs}
    for spec, table in zip(stage["inputs"], tables, strict=True):
        node = by_id[spec["node_id"]]
        port = next(port for port in node.outputs if port.name == spec["port"])
        initial[PortRef(node.node_id, port.name)] = _port_value(
            node, port, table, context)
    last = by_id[stage["nodes"][-1]]
    subgraph = PhysicalGraph(
        tuple(node for node in graph.topological_nodes()
              if node.node_id in inside | outside),
        PortRef(last.node_id, last.outputs[0].name))
    # the documents a scan sent here are this GPU's share of its rows
    scans = [spec["node_id"] for spec in stage["inputs"]
             if isinstance(by_id[spec["node_id"]], Scan)]
    rows = _StageRows({*inside, *scans}, wanted)
    with reporting(rows if wanted else None):
        result = GenericRunner().run(subgraph, context, initial_outputs=initial)
    ran = {}
    for node_id in [*scans, *stage["nodes"]]:
        ports = {port.name: port for port in by_id[node_id].outputs}
        found = result.nodes[node_id]
        ran[node_id] = NodeResult(
            {name: output_table(by_id[node_id], ports[name], value)
             for name, value in found.outputs.items()}, found.metrics)
    outputs = [ran[node_id].outputs[port] for node_id, port in stage["outputs"]]
    return outputs, rows.tables, ran


def _port_value(node, port, table: pa.Table, context: ExecutionContext):
    """Return an input table as the value the reading node takes."""
    if isinstance(node, Scan):
        # the scan's own tests still apply to the documents sent here
        scanned = context.runtimes[node.runtime_key].execute(node, {}, context)
        ids = scanned.outputs[port.name].ids[node.alias]
        sent = table.column(node.alias).cast(ids.type)
        return Rows({node.alias: ids.filter(pc.is_in(ids, value_set=sent))})
    if port.value_type is not ValueType.ROWS:
        return table
    names = _id_columns(table)
    table = distinct_rows(table, names)
    return Rows({name: table.column(name) for name in names},
                table.drop_columns(names))


def _id_columns(table: pa.Table) -> list[str]:
    """Return the names of a table's document id columns."""
    return [field.name for field in table.schema
            if field.metadata and b"quail.alias" in field.metadata]


def distinct_rows(table: pa.Table, names: list[str]) -> pa.Table:
    """Keep the first row of each combination of the named id columns.

    A partner's survivors reach a later stage from every GPU that pruned
    them, so the same ids can arrive more than once.
    """
    if table.num_rows == 0 or not names:
        return table
    positions = table.select(names).append_column(
        "position", pa.array(np.arange(table.num_rows)))
    first = positions.group_by(names, use_threads=False).aggregate(
        [("position", "min")]).column("position_min").to_numpy()
    return table if len(first) == table.num_rows else table.take(np.sort(first))


def merged_run(graph: PhysicalGraph, parts: list[dict]) -> RunResult:
    """Merge each GPU's node results into one run of the region.

    A port's rows are every GPU's rows, each set of ids once; a node's
    metrics are the sum over the GPUs.

    Args:
        graph: The region's compute graph.
        parts: Per GPU, node id -> NodeResult of Arrow tables.
    """
    nodes = {}
    for node in graph.nodes:
        found = [part[node.node_id] for part in parts if node.node_id in part]
        if not found:
            continue
        outputs = {}
        for port in node.outputs:
            table = pa.concat_tables([result.outputs[port.name] for result in found],
                                     promote_options="permissive")
            outputs[port.name] = (distinct_rows(table, _id_columns(table))
                                  if port.value_type is ValueType.ROWS else table)
        nodes[node.node_id] = NodeResult(
            outputs, sum((result.metrics for result in found), NodeMetrics()))
    root = graph.root
    return RunResult(nodes[root.node_id].outputs[root.port], nodes,
                     sum((result.metrics for result in nodes.values()), NodeMetrics()))

