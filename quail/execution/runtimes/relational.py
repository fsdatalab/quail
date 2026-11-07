"""Runtimes for the relational nodes: scan, project, limit, aggregate, sort."""

from __future__ import annotations

from collections.abc import Iterator, Sized

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

from quail.common.pairs import columns_key
from quail.common.selection import equality_keys
from quail.execution.ids import document_ids
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.result import QueryResult
from quail.execution.rows import Rows, as_table
from quail.physical import Limit, Project, Scan, Sort


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
        if node.expressions is not None:
            batches = [pa.RecordBatch.from_arrays(
                [batch.column(expression) if isinstance(expression, str)
                 else expression.evaluate(batch, qualified=True)
                 for expression in node.expressions], names=list(node.columns))
                if node.expressions else batch.select([])
                for batch in row_batches(value)]
            value = QueryResult.from_table(pa.Table.from_batches(batches))
        elif context.project is not None:
            value = (context.project(node, value, labels) if labels
                     else context.project(node, value))
        # a projection keeps its rows; a lazy result's count is the
        # input's until the result runs
        rows = _num_rows(inputs[first.name]) or 0
        output_rows = _num_rows(value)
        return NodeResult({"rows": value}, NodeMetrics(
            input_rows=rows, output_rows=rows if output_rows is None else output_rows))


def row_batches(value) -> Iterator[pa.RecordBatch]:
    """Yield a node's rows input one record batch at a time.

    An empty input yields one empty batch, which carries the schema.
    """
    value = as_table(value)
    if isinstance(value, QueryResult):
        reader = value.execute_stream()
        schema = reader.schema
        batches = iter(reader)
    elif isinstance(value, pa.RecordBatch):
        schema, batches, reader = value.schema, iter([value]), None
    elif isinstance(value, pa.Table):
        schema, batches, reader = value.schema, iter(value.to_batches()), None
    else:
        raise TypeError(f"expected rows, got {type(value).__name__}")
    try:
        empty = True
        for batch in batches:
            empty = False
            yield batch
        if empty:
            yield pa.RecordBatch.from_pylist([], schema=schema)
    finally:
        if reader is not None:
            reader.close()


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
    # internal names stay clear of the user's, and are read by position
    keyed = {f"key_{index}": equality_keys(
        _comparable(table.column(name), "DISTINCT"), nan_equal=True)
        for index, name in enumerate(columns)}
    keys = list(keyed)
    keyed["row_position"] = pa.array(np.arange(table.num_rows), pa.int64())
    firsts = pa.table(keyed).group_by(keys, use_threads=False).aggregate(
        [("row_position", "min")]).column(len(keys))
    return pc.take(firsts, pc.sort_indices(firsts))


def _rows_table(value) -> pa.Table:
    """Return a node's rows input as one Arrow table."""
    value = as_table(value)
    if isinstance(value, QueryResult):
        return value.collect()
    if isinstance(value, pa.RecordBatch):
        return pa.Table.from_batches([value])
    if isinstance(value, pa.Table):
        return value
    raise TypeError(f"expected rows, got {type(value).__name__}")


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
