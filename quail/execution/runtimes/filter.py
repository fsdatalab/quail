"""The Filter runtime: documents on the GPU executor, result rows on the coordinator."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

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
        if len(inputs) != 1:
            raise ValueError("Filter needs one input")
        value = next(iter(inputs.values()))
        if not node.aliases:
            return filter_result_rows(node, value)
        return filter_scores(node, value)


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
