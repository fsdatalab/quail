"""The Filter runtime: documents on the GPU executor, result rows on the coordinator."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.result import QueryResult, answer_table, filter_answer_table
from quail.execution.rows import Rows
from quail.execution.runtimes.relational import row_batches
from quail.logical import column_name
from quail.physical import Filter


class FilterRuntime:
    """Keep the rows that pass the predicate.

    A filter with aliases reads a score, label, or answer table and also
    returns its answers; one without reads result rows.
    """

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Filter):
            raise TypeError(type(node).__name__)
        if len(inputs) != 1:
            raise ValueError("Filter needs one input")
        value = next(iter(inputs.values()))
        if not node.aliases:
            return filter_result_rows(node, value)
        return filter_scores(node, value)


def filter_result_rows(node, value) -> NodeResult:
    """Keep the result rows whose predicate holds, one batch at a time."""
    name = column_name(node.predicate.column)
    num_rows, batches = 0, []
    for batch in row_batches(value):
        num_rows += batch.num_rows
        # the predicate reads its column by the bare column name
        tested = pa.RecordBatch.from_arrays(
            [batch.column(name)], names=[node.predicate.column.column])
        batches.append(batch.filter(node.predicate.accepts(tested))
                       .select(list(node.columns)))
    table = pa.Table.from_batches(batches)
    return NodeResult({"rows": QueryResult.from_table(table)}, NodeMetrics(
        input_rows=num_rows, output_rows=table.num_rows))


def _filter_answer_table(node: Filter, rows: Rows, answers) -> pa.Table:
    """Build one answer row per document using the filter answer schema."""
    alias = node.aliases[0]
    return filter_answer_table(
        alias, rows.ids[alias],
        np.full(rows.num_rows, node.written_position, dtype=np.int32), answers)


def filter_scores(node, rows: Rows) -> NodeResult:
    """Apply a filter to rows of scores, labels, or answers."""
    answers = node.predicate.accepts(rows)
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
