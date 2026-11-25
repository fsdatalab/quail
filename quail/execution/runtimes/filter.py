"""The Filter runtime: a condition on documents, on the GPU executor."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

from quail.common.expressions import accepts
from quail.common.pairs import columns_key
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.result import answer_table, filter_answer_table
from quail.execution.rows import Rows
from quail.physical import Filter


class FilterRuntime:
    """Keep the rows that pass the predicate.

    It reads a score, label, or answer table and also returns its answers.
    """

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Filter):
            raise TypeError(type(node).__name__)
        values = [inputs[port.name] for port in node.inputs]
        rows, others = values[0], values[1:]
        if all(name in rows.column_names for name in node.tested_columns):
            return filter_scores(node, rows)
        sources = context.sources.get(columns_key(node.aliases[0]))
        return filter_scores(node, rows, tested_table(
            node, rows, lambda name, ids: _read(name, ids, node, others, sources)))


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
    answers = accepts(node.predicate, rows if tested is None else tested)
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
                "predicate": str(node.predicate),
            },
        )
    outputs = {node.rows_port: filtered, answer_name: answer_relation}
    return NodeResult(
        outputs,
        NodeMetrics(input_rows=rows.num_rows, output_rows=filtered.num_rows),
    )
