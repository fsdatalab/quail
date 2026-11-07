"""Runtimes for the relational nodes: scan, project, and limit."""

from __future__ import annotations

from collections.abc import Sized

import pyarrow as pa

from quail.common.pairs import columns_key
from quail.execution.ids import document_ids
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.result import QueryResult
from quail.execution.rows import Rows, as_table
from quail.physical import Limit, Project, Scan


class ScanRuntime:
    """Read a prepared source registered by document alias."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Scan):
            raise TypeError(type(node).__name__)
        if node.input_id not in context.sources:
            raise KeyError(
                f"no prepared source for input {node.input_id!r}"
            )
        value = document_ids(
            _tested_documents(node, context) if node.predicates
            else context.sources[node.input_id])
        return NodeResult({f"rows:{node.alias}": Rows({node.alias: value})},
                          NodeMetrics(output_rows=len(value)))


def _tested_documents(node, context) -> pa.Array:
    """Return the positions of the documents that pass a scan's regular predicates."""
    from quail.common.selection import selected_rows

    columns = context.sources.get(columns_key(node.alias))
    if columns is None:
        raise KeyError(
            f"the regular predicates of {node.alias!r} need its value columns")
    return selected_rows(columns, node.predicates)


def _num_rows(value) -> int | None:
    """Rows of a table or sequence, or None when counting would run a plan."""
    if isinstance(value, QueryResult):
        return value.known_num_rows()
    return len(value) if isinstance(value, Sized) else None


class ProjectRuntime:
    """Run the configured result projection.

    The first input is the relation to project; any others are label
    tables whose columns the projection may name.
    """

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Project):
            raise TypeError(type(node).__name__)
        if not inputs:
            raise ValueError("Project needs one input")
        first, *rest = node.inputs
        value = as_table(inputs[first.name])
        labels = {port.name: as_table(inputs[port.name]) for port in rest}
        if context.project is not None:
            value = (context.project(node, value, labels) if labels
                     else context.project(node, value))
        # a projection keeps its rows; a lazy result's count is the
        # input's until the result runs
        rows = _num_rows(inputs[first.name]) or 0
        output_rows = _num_rows(value)
        return NodeResult({"rows": value}, NodeMetrics(
            input_rows=rows, output_rows=rows if output_rows is None else output_rows))


class LimitRuntime:
    """Limit a table, batch, or Python sequence."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Limit):
            raise TypeError(type(node).__name__)
        if len(inputs) != 1:
            raise ValueError("Limit needs one input")
        value = next(iter(inputs.values()))

        rows = _num_rows(value) or 0
        if isinstance(value, QueryResult):
            value = value.with_limit(node.count)
        else:
            value = value[:node.count]
        return NodeResult({"rows": value}, NodeMetrics(
            input_rows=rows, output_rows=_num_rows(value) or 0))
