"""Run a region's stages on several GPUs, one partition per GPU.

Rust's ``quail_python::stages`` runs partition p of each stage on the server of
GPU p: a process, or a test's thread, holding every document and one
execution context for the region. Rust sends a stage's input rows; the
worker runs the stage's nodes with the single-GPU runner, sends each
table it finishes for the region's FinishedRows as soon as it has it,
and sends back the rows later stages read. The region's hint stays in
the region's process: the worker asks it over the same connection.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from quail.execution.finished import finished_table, first_rows, reporting
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.rows import Rows
from quail.execution.runner import ExecutionContext, GenericRunner
from quail.execution.types import output_table
from quail.physical import PhysicalGraph, PortRef, Scan, ValueType

# The longest a server goes without asking whether the hint tightened.
# The hint only lets the model skip rows, so a late answer costs work,
# not rows.
_HINT_SECONDS = 0.005


class ServedHint:
    """The region's ``InputHint``, asked over a worker's connection.

    Args:
        server: The ``WorkerEnd`` of the running region.
        tables: The names of the tables the hint holds rows of.
    """

    def __init__(self, server, tables):
        self._server = server
        self._tables = list(tables)
        self._generation = None
        self._asked = 0.0

    def _ask(self, question: str, **arguments):
        return self._server.ask(json.dumps({"hint": question, **arguments}))

    def tables(self) -> list[str]:
        return self._tables

    def generation(self) -> int:
        now = time.monotonic()
        if self._generation is None or now - self._asked >= _HINT_SECONDS:
            self._generation = json.loads(self._ask("generation"))
            self._asked = now
        return self._generation

    def keep(self, table: str):
        return self._ask("keep", table=table).column("keep").to_numpy(
            zero_copy_only=False)

    def reads_answer(self, answer: str) -> bool:
        return json.loads(self._ask("reads_answer", answer=answer))

    def rejects_answer(self, answer: str, prefix: str, **completion) -> bool:
        return json.loads(self._ask("rejects_answer", answer=answer,
                                    prefix=prefix, completion=completion))


def answer_hint(hint, question: dict):
    """Answer one question a ``ServedHint`` asked, from the region's hint.

    Returns:
        The answer as JSON text, or a table for a table's kept rows.
    """
    asked = question["hint"]
    if asked == "keep":
        return pa.table({"keep": pa.array(hint.keep(question["table"]), pa.bool_())})
    if asked == "generation":
        return json.dumps(hint.generation())
    if asked == "reads_answer":
        return json.dumps(hint.reads_answer(question["answer"]))
    return json.dumps(hint.rejects_answer(
        question["answer"], question["prefix"], **question["completion"]))


def serve_stages(server, context: ExecutionContext, graph: PhysicalGraph,
                 stages: list, wanted, hinted: dict | None = None,
                 first: dict | None = None) -> dict:
    """Run each stage Rust sends until the region ends.

    Args:
        server: The ``WorkerEnd`` that has read the region.
        context: The region's execution context on this GPU.
        graph: The region's compute graph.
        stages: The plan's stages, as its "stages" setting holds them.
        wanted: The (node, port) pairs the region's FinishedRows reads;
            None when no one takes the region's rows as they finish.
        hinted: Each scan alias's table name, for the tables the
            region's hint holds rows of; None when it has no hint.
        first: For each alias the plan deduplicates, each row's first
            row with the same document.

    Returns:
        Node id -> NodeResult, of Arrow tables, of each node this server ran.
    """
    reports = wanted is not None
    wanted = {tuple(pair) for pair in wanted or ()}
    hint = None if hinted is None else ServedHint(server, set(hinted.values()))
    results = {}
    server.ready()
    while (message := server.next()) is not None and message[0] == "stage":
        _, stage, tables = message
        try:
            rows = _StageRows(server, wanted)
            with reporting(rows if reports else None, hint, hinted, first):
                outputs, ran = run_stage(context, graph, stages[stage], tables, rows)
        except Exception:
            server.fail(traceback.format_exc())
            continue
        results.update(ran)
        server.outputs(outputs)
    return results


def plain(value):
    """Return a numpy value as the Python value JSON writes."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON")


def ran_message(results: dict, measurements: dict) -> tuple[str, list]:
    """Return what a worker ran as the JSON and tables it sends.

    Args:
        results: Node id -> NodeResult, of Arrow tables.
        measurements: The worker's KV counters and memory report.

    Returns:
        The measurements and each node's metrics as JSON, and each
        node's output tables, tagged with node and port.
    """
    tables = [finished_table(node_id, port, table)
              for node_id, result in results.items()
              for port, table in result.outputs.items()]
    metrics = {node_id: asdict(result.metrics)
               for node_id, result in results.items()}
    return json.dumps({**measurements, "metrics": metrics}, default=plain), tables


def payload_first_rows(payload: dict) -> dict:
    """Each deduplicated alias's first rows, from a runtime payload."""
    return first_rows(payload.get("columns", {}), payload.get("deduplicate", {}))


class _StageRows:
    """Sends the rows a stage's nodes finish to the region's FinishedRows."""

    def __init__(self, server, wanted):
        self.server, self.wanted, self.nodes = server, wanted, set()

    def wants(self, node: str, port: str) -> bool:
        return node in self.nodes and (node, port) in self.wanted

    def report(self, table: pa.Table) -> None:
        self.server.finished(table)


def run_stage(context: ExecutionContext, graph: PhysicalGraph, stage: dict,
              tables: list, rows: _StageRows) -> tuple[list, dict]:
    """Run one stage's nodes over its input rows.

    Args:
        context: The region's execution context on this GPU.
        graph: The region's compute graph.
        stage: The stage, as the plan's "stages" setting holds it.
        tables: Each stage input's rows, in the stage's input order.
        rows: Where the stage's nodes report the rows they finish.

    Returns:
        The tables of the stage's outputs, in its output order, and
        node id -> NodeResult of the scans it read and the nodes it ran.
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
    scanned = {}
    for spec, table in zip(stage["inputs"], tables, strict=True):
        node = by_id[spec["node_id"]]
        port = next(port for port in node.outputs if port.name == spec["port"])
        started = time.perf_counter()
        value = _port_value(node, port, table, context)
        initial[PortRef(node.node_id, port.name)] = value
        if isinstance(node, Scan):
            scanned[node.node_id] = NodeMetrics(
                output_rows=len(value), wall_s=time.perf_counter() - started)
    last = by_id[stage["nodes"][-1]]
    subgraph = PhysicalGraph(
        tuple(node for node in graph.topological_nodes()
              if node.node_id in inside | outside),
        PortRef(last.node_id, last.outputs[0].name))
    # the documents a scan sent here are this GPU's share of its rows
    scans = [spec["node_id"] for spec in stage["inputs"]
             if isinstance(by_id[spec["node_id"]], Scan)]
    rows.nodes = {*inside, *scans}
    result = GenericRunner().run(subgraph, context, initial_outputs=initial,
                                 initial_metrics=scanned)
    ran = {}
    for node_id in [*scans, *stage["nodes"]]:
        ports = {port.name: port for port in by_id[node_id].outputs}
        found = result.nodes[node_id]
        ran[node_id] = NodeResult(
            {name: output_table(by_id[node_id], ports[name], value)
             for name, value in found.outputs.items()}, found.metrics)
    outputs = [ran[node_id].outputs[port] for node_id, port in stage["outputs"]]
    return outputs, ran


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
