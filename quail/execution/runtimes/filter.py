"""The Filter runtime: documents on the GPU executor, result rows on the coordinator."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

from quail.catalog.pairs import columns_key
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.result import QueryResult, answer_table, filter_answer_table
from quail.execution.rows import Rows
from quail.execution.runtimes.relational import row_batches
from quail.physical import Filter


class FilterRuntime:
    """Keep the rows that pass the predicate.

    A filter with aliases reads a score, label, or answer table and also
    returns its answers; one without reads result rows.
    """

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Filter):
            raise TypeError(type(node).__name__)
        values = [inputs[port.name] for port in node.inputs]
        if not node.aliases:
            if len(values) != 1:
                raise ValueError("a Filter on result rows needs one input")
            return filter_result_rows(node, values[0])
        rows, others = values[0], values[1:]
        if all(name in rows.column_names for name in node.tested_columns):
            return filter_scores(node, rows)
        sources = context.sources.get(columns_key(node.aliases[0]))
        return filter_scores(node, rows, tested_table(
            node, rows, lambda name, ids: _read(name, ids, node, others, sources)))


class ResultColumns:
    """Read result row columns by name, including fields of struct columns.

    The Project packs each extraction's answers into a struct column,
    so ``lease.ends`` is the field ``ends`` of the column ``lease`` when
    no column has that name.
    """

    def __init__(self, batch: pa.RecordBatch):
        self.batch = batch

    def column(self, name: str):
        if name in self.batch.schema.names:
            return self.batch.column(name)
        struct, _, field = name.rpartition(".")
        if struct in self.batch.schema.names:
            return pc.struct_field(self.batch.column(struct), field)
        raise KeyError(f"the result rows have no column {name!r}")


def filter_result_rows(node, value) -> NodeResult:
    """Keep the result rows whose predicate holds, one batch at a time."""
    num_rows, batches = 0, []
    for batch in row_batches(value):
        num_rows += batch.num_rows
        kept = node.predicate.accepts(ResultColumns(batch), qualified=True)
        batches.append(batch.filter(kept).select(list(node.columns)))
    table = pa.Table.from_batches(batches)
    return NodeResult({"rows": QueryResult.from_table(table)}, NodeMetrics(
        input_rows=num_rows, output_rows=table.num_rows))


def _filter_answer_table(node: Filter, rows: Rows, answers) -> pa.Table:
    """Build one answer row per document using the filter answer schema."""
    alias = node.aliases[0]
    return filter_answer_table(
        alias, rows.ids[alias],
        np.full(rows.num_rows, node.written_position, dtype=np.int32), answers)


def tested_table(node, rows: Rows, read) -> pa.Table:
    """Return every column a document Filter tests, one row per row of rows.

    Args:
        node: The Filter.
        rows: The rows it filters; their columns are read as they are.
        read: (column, document ids) -> the column's values for those
            documents, for a column the rows lack: another operator's
            label or answer, or a source column.
    """
    ids = rows.ids[node.aliases[0]]
    return pa.table({
        name: (rows.column(name) if name in rows.column_names
               else read(name, ids))
        for name in node.tested_columns})


def _read(name: str, ids, node, others, sources):
    """Read a column for documents from other operators' rows or the source table."""
    alias = node.aliases[0]
    for other in others:
        if name in other.column_names:
            positions = pc.index_in(ids, value_set=other.ids[alias])
            return pc.take(other.column(name), positions)
    if sources is not None and name in sources.column_names:
        return pc.take(sources.column(name), ids)
    raise KeyError(f"{node.node_id!r} reads {name!r}, which no input holds")


def filter_scores(node, rows: Rows, tested: pa.Table | None = None) -> NodeResult:
    """Apply a filter to rows of scores, labels, or answers.

    Args:
        node: The Filter.
        rows: The rows it filters.
        tested: The columns the predicate reads, one row per row of
            rows, when the rows lack some of them.
    """
    answers = node.predicate.accepts(rows if tested is None else tested)
    filtered = rows.filter(answers)
    if len(node.aliases) == 1:
        answer_name = f"filter_answers:{node.aliases[0]}"
        answer_relation = _filter_answer_table(node, rows, answers)
    else:
        left, right = node.aliases
        answer_name = f"join_answers:{node.written_position}"
        answer_relation = answer_table(
            {left: rows.ids[left], right: rows.ids[right]},
            answers,
            "join_answers",
            metadata={
                "written_position": node.written_position,
                "anchor": left,
                "partners": right,
                "semantics": "full",
                "comparison": node.predicate.comparison,
                "threshold": node.predicate.value,
            },
        )
    outputs = {node.rows_port: filtered, answer_name: answer_relation}
    return NodeResult(
        outputs,
        NodeMetrics(input_rows=rows.num_rows, output_rows=filtered.num_rows),
    )
