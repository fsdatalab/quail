"""Run a region's stages on several GPUs, one partition per GPU.

Rust's ``quail::stages`` cuts a region's graph into stages and runs
partition p of each stage on the server of GPU p: a process, or for
tests a thread, that holds every document and one execution context
for the region. Rust sends the server a stage's input rows as Arrow IPC
over a Unix socket; the server runs the stage's nodes with the
single-GPU runner and sends back the rows later stages read and the
rows the stage finished for the region's FinishedRows. Each frame is
an 8-byte little-endian length and its bytes: a JSON header, then
Arrow IPC streams.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import tempfile
import traceback
from collections.abc import Callable, Iterable

import _quail_native
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from quail.execution.finished import finished_rows, reporting
from quail.execution.metrics import NodeMetrics, NodeResult, RunResult
from quail.execution.rows import Rows
from quail.execution.runner import ExecutionContext, GenericRunner
from quail.execution.types import output_table
from quail.physical import PhysicalGraph, PortRef, Scan, ValueType


def run_partitions(graph: PhysicalGraph, keys: dict, gpus: int,
                   serve: Callable) -> list[dict]:
    """Run a region's stages on one server per GPU and return their results.

    The rows each stage finishes go to the region's FinishedRows, when
    the region reports to one.

    Args:
        graph: The region's compute graph.
        keys: Alias -> each document's key, for a table whose documents
            go to a GPU by a key other than their ids.
        gpus: How many GPUs, one server each.
        serve: Callable(paths, wanted) that starts one server per
            socket path, each listening before the call returns, and
            returns a callable that waits for every server and returns
            each one's results: node id -> NodeResult of Arrow tables,
            and its measurements. wanted lists the (node, port) pairs
            the region's FinishedRows reads.
    """
    rows = finished_rows()
    wanted = [] if rows is None else [
        (node.node_id, port.name) for node in graph.nodes
        for port in node.outputs if rows.wants(node.node_id, port.name)]

    def report(data: bytes) -> None:
        rows.report(pa.ipc.open_stream(data).read_all())

    with tempfile.TemporaryDirectory() as directory:
        paths = [os.path.join(directory, f"gpu{index}.sock") for index in range(gpus)]
        results = serve(paths, wanted)
        try:
            _quail_native.run_stages(graph.to_json(), keys, paths,
                                     report if rows is not None else _ignore)
        except BaseException:
            # the servers ended too; a failure on them is in this error
            try:
                results()
            except Exception:
                pass
            raise
        return results()


def _ignore(data: bytes) -> None:
    """Take a finished table no FinishedRows reads."""


def listening(path: str) -> socket.socket:
    """Return a Unix socket listening at path for the region's stages."""
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(path)
    listener.listen(1)
    return listener


def serve_stages(listener: socket.socket, context: ExecutionContext,
                 graph: PhysicalGraph, wanted: Iterable) -> dict:
    """Run each stage Rust sends until the region ends.

    Args:
        listener: The listening socket Rust connects to.
        context: The region's execution context on this GPU.
        graph: The region's compute graph.
        wanted: The (node, port) pairs the region's FinishedRows reads.

    Returns:
        Node id -> NodeResult of every node this server ran, its
        outputs as Arrow tables.
    """
    stages = json.loads(_quail_native.region_stages(graph.to_json()))
    wanted = {tuple(pair) for pair in wanted}
    results = {}
    connection, _ = listener.accept()
    with connection, connection.makefile("rb") as reader, \
            connection.makefile("wb") as writer:
        while True:
            header = json.loads(_read_frame(reader))
            if header.get("end"):
                return results
            stage = stages[header["stage"]]
            tables = [pa.ipc.open_stream(_read_frame(reader)).read_all()
                      for _ in stage["inputs"]]
            try:
                outputs, finished, ran = run_stage(context, graph, stage, tables,
                                                   wanted)
            except Exception:
                _write_frame(writer, json.dumps(
                    {"error": traceback.format_exc()}).encode())
                writer.flush()
                continue
            results.update(ran)
            _write_frame(writer, json.dumps(
                {"outputs": len(outputs), "finished": len(finished)}).encode())
            for table in [*outputs, *finished]:
                _write_frame(writer, ipc_bytes(table))
            writer.flush()


class _StageRows:
    """Keep the rows a stage's nodes finish, for the region's FinishedRows."""

    def __init__(self, nodes, wanted):
        self.nodes = nodes
        self.wanted = wanted
        self.tables = []

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
        id -> NodeResult of the stage's nodes.
    """
    by_id = {node.node_id: node for node in graph.nodes}
    inside = set(stage["nodes"])
    outside = set()
    pending = list(inside)
    while pending:
        for port in by_id[pending.pop()].inputs:
            source = port.source.node_id
            if source not in inside and source not in outside:
                outside.add(source)
                pending.append(source)
    # a node outside the stage counts as run; only the ports the stage
    # reads hold rows
    initial = {PortRef(node_id, port.name): None for node_id in outside
               for port in by_id[node_id].outputs}
    for spec, table in zip(stage["inputs"], tables, strict=True):
        node = by_id[spec["node_id"]]
        port = next(port for port in node.outputs if port.name == spec["port"])
        initial[PortRef(node.node_id, port.name)] = _port_value(
            node, port, table, context)
    nodes = tuple(node for node in graph.topological_nodes()
                  if node.node_id in inside or node.node_id in outside)
    last = by_id[stage["nodes"][-1]]
    subgraph = PhysicalGraph(nodes, PortRef(last.node_id, last.outputs[0].name))
    # the documents a scan sent here are this GPU's share of its rows
    scans = [spec["node_id"] for spec in stage["inputs"]
             if isinstance(by_id[spec["node_id"]], Scan)]
    rows = _StageRows({*inside, *scans}, wanted)
    with reporting(rows if wanted else None):
        result = GenericRunner().run(subgraph, context, initial_outputs=initial)
    ran = {}
    for node_id in [*scans, *stage["nodes"]]:
        node, found = by_id[node_id], result.nodes[node_id]
        ports = {port.name: port for port in node.outputs}
        ran[node_id] = NodeResult(
            {name: output_table(node, ports[name], value)
             for name, value in found.outputs.items()}, found.metrics)
    outputs = [ran[node_id].outputs[port] for node_id, port in stage["outputs"]]
    return outputs, rows.tables, ran


def _port_value(node, port, table: pa.Table, context: ExecutionContext):
    """Return an input table as the value the reading node takes."""
    if isinstance(node, Scan):
        # the scan's own tests still apply to the documents it sends here
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

    A partner's survivors reach a later stage from every GPU that
    pruned them, so the same ids can arrive more than once.
    """
    if table.num_rows == 0 or not names:
        return table
    positions = pa.table({**{name: table.column(name) for name in names},
                          "position": np.arange(table.num_rows)})
    first = positions.group_by(names, use_threads=False).aggregate(
        [("position", "min")]).column("position_min").to_numpy()
    if len(first) == table.num_rows:
        return table
    return table.take(np.sort(first))


def merged_run(graph: PhysicalGraph, parts: list[dict]) -> RunResult:
    """Merge each GPU's node results into one run of the region.

    A port's rows are every GPU's rows, each set of ids once; a node's
    metrics are the sum over the GPUs.

    Args:
        graph: The region's compute graph.
        parts: Per GPU, node id -> NodeResult of Arrow tables.
    """
    nodes = {}
    total = NodeMetrics()
    for node in graph.nodes:
        found = [part[node.node_id] for part in parts if node.node_id in part]
        if not found:
            continue
        outputs = {}
        for port in node.outputs:
            table = pa.concat_tables(
                [result.outputs[port.name] for result in found],
                promote_options="permissive")
            outputs[port.name] = (distinct_rows(table, _id_columns(table))
                                  if port.value_type is ValueType.ROWS else table)
        metrics = sum((result.metrics for result in found), NodeMetrics())
        nodes[node.node_id] = NodeResult(outputs, metrics)
        total = total + metrics
    root = graph.root
    return RunResult(nodes[root.node_id].outputs[root.port], nodes, total)


def ipc_bytes(table: pa.Table) -> bytes:
    """Return a table as Arrow IPC stream bytes."""
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _write_frame(writer, data: bytes) -> None:
    writer.write(struct.pack("<Q", len(data)))
    writer.write(data)


def _read_frame(reader) -> bytes:
    (length,) = struct.unpack("<Q", reader.read(8))
    data = reader.read(length)
    if len(data) != length:
        raise EOFError("the region's connection closed inside a frame")
    return data

