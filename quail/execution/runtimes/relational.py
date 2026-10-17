"""Runtimes for the relational nodes: scan, project, limit, aggregate, and sort."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

from quail.catalog.pairs import columns_key
from quail.execution.aggregates import prepare_aggregate
from quail.execution.ids import document_ids
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.result import QueryResult
from quail.execution.rows import Rows
from quail.logical.expressions import hash_values
from quail.logical.predicates import compare_literal
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
    from quail.catalog.selection import selected_rows

    columns = context.sources.get(columns_key(node.alias))
    if columns is None:
        raise KeyError(
            f"the regular predicates of {node.alias!r} need its value columns")
    values = {predicate.column.column: columns.column(predicate.column.column)
              for predicate in node.predicates}
    return selected_rows(values, node.predicates)


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
        value = _as_table(inputs[first.name])
        labels = {port.name: _as_table(inputs[port.name]) for port in rest}
        if context.project is not None:
            value = (context.project(node, value, labels) if labels
                     else context.project(node, value))
        # A lazy projection may not know its output count until it runs.
        rows = _row_count(inputs[first.name]) or 0
        output_rows = _row_count(value)
        return NodeResult({"rows": value}, NodeMetrics(
            input_rows=rows,
            output_rows=rows if output_rows is None else output_rows))


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
    values = {f"column_{index}": hash_values(
        _comparable(table.column(name), "DISTINCT"), nan_equal=True)
              for index, name in enumerate(columns)}
    keys = list(values)
    values["row_position"] = pa.array(np.arange(table.num_rows), pa.int64())
    firsts = pa.table(values).group_by(keys, use_threads=False).aggregate(
        [("row_position", "min")]).column(len(keys))
    return pc.take(firsts, pc.sort_indices(firsts))


_AGGREGATE_KERNELS = {
    "count": "count", "count_distinct": "count_distinct", "sum": "sum",
    "avg": "mean", "min": "min", "max": "max",
}


def _as_table(value):
    """Rows as their table; any other value as it is."""
    return value.table() if isinstance(value, Rows) else value


def _rows_table(value) -> pa.Table:
    """Return a node's rows input as one Arrow table."""
    if isinstance(value, Rows):
        return value.table()
    if isinstance(value, QueryResult):
        return value.collect()
    if isinstance(value, pa.RecordBatch):
        return pa.Table.from_batches([value])
    if isinstance(value, pa.Table):
        return value
    raise TypeError(f"expected rows, got {type(value).__name__}")


def _aggregate_kernel(function, values):
    """Choose an aggregate kernel from the function and its input values."""
    if values is None:
        return "count_all"
    if function == "count_distinct" and values.null_count == len(values):
        return "count"
    return _AGGREGATE_KERNELS[function]


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
    hash_names = {name: f"key_{index}" for index, name in enumerate(dict.fromkeys((
        *node.keys, *(argument for _, function, argument in node.aggregates
                      if function == "count_distinct"))))}
    keyed.update({key: hash_values(keyed[names[name]], nan_equal=True)
                  for name, key in hash_names.items()})
    keyed["row_position"] = pa.array(np.arange(table.num_rows), pa.int64())
    aggregates = [
        (name, argument, _aggregate_kernel(
            function, None if argument is None else keyed[names[argument]]))
        for name, function, argument in node.aggregates]
    sources = {}
    scalars = {}
    finalizers = {}
    specs = []
    for _, argument, kernel in aggregates:
        key = (argument, kernel)
        if key in sources or key in scalars:
            continue
        source = hash_names if kernel == "count_distinct" else names
        column = None if argument is None else source[argument]
        if column is not None:
            values, kernel, finish = prepare_aggregate(
                keyed[column], kernel, grouped=bool(node.keys))
            if kernel is None:
                scalars[key] = finish(None)
                continue
            if finish is not None:
                finalizers[len(node.keys) + len(specs)] = finish
            column = f"aggregate_{len(specs)}"
            keyed[column] = values
        sources[key] = len(node.keys) + len(specs)
        specs.append(([], "count_all") if argument is None else (column, kernel))
    grouped = pa.table(keyed).group_by(
        [hash_names[key] for key in node.keys], use_threads=False
    ).aggregate(specs + [("row_position", "min")])
    firsts = grouped.column(len(node.keys) + len(specs))
    order = pc.sort_indices(firsts)
    for index, finish in finalizers.items():
        grouped = grouped.set_column(index, grouped.column_names[index],
                                     finish(grouped.column(index)))
    columns = {name: pc.take(keyed[names[name]], firsts) for name in node.keys}
    for name, argument, kernel in aggregates:
        key = (argument, kernel)
        columns[name] = (scalars[key] if key in scalars
                         else grouped.column(sources[key]))
    result = pa.table({name: pc.take(column, order)
                       for name, column in columns.items()})
    mask = None
    for name, comparison, value in node.having:
        test = compare_literal(result.column(name), comparison, value)
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
        stop = None if node.fetch is None else node.offset + node.fetch
        table = table[node.offset:stop].select(list(node.columns))
        return NodeResult({"rows": QueryResult.from_table(table)}, NodeMetrics(
            input_rows=rows, output_rows=table.num_rows))
