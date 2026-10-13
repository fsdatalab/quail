"""Runtimes for the relational nodes: scan, project, limit, aggregate, and sort."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

from quail.catalog.pairs import columns_key
from quail.execution.ids import document_ids
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.result import QueryResult
from quail.physical import Aggregate, Limit, Project, Scan, Sort


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
        return NodeResult({f"ids:{node.alias}": value},
                          NodeMetrics(output_rows=len(value)))


def _tested_documents(node, context) -> pa.Array:
    """Return the positions of the documents that pass a scan's regular predicates."""
    from quail.execution.selection import selected_rows
    from quail.logical import ColumnRef, RegularPredicate

    columns = context.sources.get(columns_key(node.alias))
    if columns is None:
        raise KeyError(
            f"the regular predicates of {node.alias!r} need its value columns")
    predicates = [
        RegularPredicate(ColumnRef(node.alias, "", column), comparison, value)
        for column, comparison, value in node.predicates]
    values = {column: columns.column(column)
              for column, _, _ in node.predicates}
    return selected_rows(values, predicates)


def _row_count(value) -> int | None:
    """Rows of a table or sequence, or None when counting would run a plan."""
    if isinstance(value, QueryResult):
        return value.known_count()
    if hasattr(value, "num_rows"):
        return int(value.num_rows)
    if hasattr(value, "__len__"):
        return len(value)
    return None


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
        value = inputs[first.name]
        labels = {port.name: inputs[port.name] for port in rest}
        if context.project is not None:
            value = (context.project(node, value, labels) if labels
                     else context.project(node, value))
        rows = _row_count(value) or 0
        return NodeResult({"rows": value},
                          NodeMetrics(input_rows=rows, output_rows=rows))


class LimitRuntime:
    """Limit a table, batch, or Python sequence."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Limit):
            raise TypeError(type(node).__name__)
        if len(inputs) != 1:
            raise ValueError("Limit needs one input")
        value = next(iter(inputs.values()))

        rows = _row_count(value) or 0
        if isinstance(value, QueryResult):
            value = value.with_limit(node.count)
        elif hasattr(value, "slice"):
            value = value.slice(0, node.count)
        else:
            value = value[:node.count]
        return NodeResult({"rows": value}, NodeMetrics(
            input_rows=rows, output_rows=_row_count(value) or 0))


def _comparable(column, clause: str):
    """Return a column as a type Arrow can sort or group by."""
    if pa.types.is_dictionary(column.type):
        column = column.cast(column.type.value_type)
    if pa.types.is_nested(column.type):
        raise TypeError(
            f"{clause} cannot compare values of type {column.type}")
    return column


def _distinct_indices(table: pa.Table, columns) -> pa.Array:
    """Return the first row of each distinct value of the columns, in order."""
    keyed = pa.table({
        **{name: _comparable(table.column(name), "DISTINCT")
           for name in columns},
        "__row": pa.array(np.arange(table.num_rows), pa.int64()),
    })
    firsts = keyed.group_by(list(columns), use_threads=False).aggregate(
        [("__row", "min")]).column("__row_min")
    return pc.take(firsts, pc.sort_indices(firsts))


_AGGREGATE_KERNELS = {
    "count": "count", "count_distinct": "count_distinct", "sum": "sum",
    "avg": "mean", "min": "min", "max": "max",
}
_HAVING_COMPARE = {
    "=": pc.equal, "<>": pc.not_equal, "<": pc.less, "<=": pc.less_equal,
    ">": pc.greater, ">=": pc.greater_equal,
}


def _rows_table(value) -> pa.Table:
    """Return a node's rows input as one Arrow table."""
    if isinstance(value, QueryResult):
        return value.collect()
    if isinstance(value, pa.RecordBatch):
        return pa.Table.from_batches([value])
    if isinstance(value, pa.Table):
        return value
    raise TypeError(f"expected rows, got {type(value).__name__}")


def aggregate_table(table: pa.Table, node) -> pa.Table:
    """Group a table by the node's keys and compute its aggregates.

    Dictionary key columns are grouped by their values. A null argument
    value is left out of every aggregate but count(*). Groups come out
    in the order their first rows appear.
    """
    arguments = list(dict.fromkeys((*node.keys, *(
        argument for _, _, argument in node.aggregates
        if argument is not None))))
    # Arrow derives aggregate output names from input names. Use private
    # names throughout the intermediate table to avoid user-name collisions.
    names = {name: f"column_{index}" for index, name in enumerate(arguments)}
    keyed = {names[name]: _comparable(table.column(name), "an aggregate")
             for name in arguments}
    keyed["row_position"] = pa.array(np.arange(table.num_rows), pa.int64())
    sources = {}
    specs = []
    for _, function, argument in node.aggregates:
        kernel = "count_all" if argument is None else _AGGREGATE_KERNELS[function]
        if (argument, kernel) not in sources:
            sources[(argument, kernel)] = len(node.keys) + len(specs)
            specs.append(([], "count_all") if argument is None
                         else (names[argument], kernel))
    grouped = pa.table(keyed).group_by(
        [names[key] for key in node.keys], use_threads=False
    ).aggregate(specs + [("row_position", "min")])
    order = pc.sort_indices(grouped.column(len(node.keys) + len(specs)))
    columns = {name: grouped.column(index)
               for index, name in enumerate(node.keys)}
    for name, function, argument in node.aggregates:
        kernel = "count_all" if argument is None else _AGGREGATE_KERNELS[function]
        columns[name] = grouped.column(sources[(argument, kernel)])
    result = pa.table({name: pc.take(column, order)
                       for name, column in columns.items()})
    mask = None
    for name, comparison, value in node.having:
        test = _HAVING_COMPARE[comparison](result.column(name), value)
        mask = test if mask is None else pc.and_kleene(mask, test)
    if mask is not None:
        result = result.filter(pc.fill_null(mask, False))
    return result.select(list(node.columns))


class AggregateRuntime:
    """Group and aggregate a table or a materialized result."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Aggregate):
            raise TypeError(type(node).__name__)
        if len(inputs) != 1:
            raise ValueError("Aggregate needs one input")
        table = _rows_table(next(iter(inputs.values())))
        result = aggregate_table(table, node)
        return NodeResult({"rows": QueryResult.from_table(result)}, NodeMetrics(
            input_rows=table.num_rows, output_rows=result.num_rows))


class SortRuntime:
    """Sort, deduplicate, and bound a table or a materialized result."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Sort):
            raise TypeError(type(node).__name__)
        if len(inputs) != 1:
            raise ValueError("Sort needs one input")
        table = _rows_table(next(iter(inputs.values())))
        rows = table.num_rows
        if node.distinct:
            table = table.take(_distinct_indices(table, node.columns))
        if node.keys:
            keyed = pa.table({
                column: _comparable(table.column(column), "ORDER BY")
                for column, _, _ in node.keys})
            indices = pc.sort_indices(keyed, sort_keys=[
                (column, "descending" if descending else "ascending",
                 "at_start" if nulls_first else "at_end")
                for column, descending, nulls_first in node.keys])
            table = table.take(indices)
        table = table.slice(node.offset, node.fetch).select(list(node.columns))
        return NodeResult({"rows": QueryResult.from_table(table)}, NodeMetrics(
            input_rows=rows, output_rows=table.num_rows))
