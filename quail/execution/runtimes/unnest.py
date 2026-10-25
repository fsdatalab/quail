"""The Unnest runtime: one row per element of list columns."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.result import QueryResult
from quail.execution.runtimes.relational import row_batches
from quail.physical import Unnest


class UnnestRuntime:
    """Unnest the node's list columns, one record batch at a time."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Unnest):
            raise TypeError(type(node).__name__)
        if len(inputs) != 1:
            raise ValueError("Unnest needs one input")
        num_rows, batches = 0, []
        for batch in row_batches(next(iter(inputs.values()))):
            num_rows += batch.num_rows
            batches.append(unnest_batch(batch, node))
        table = pa.Table.from_batches(batches)
        return NodeResult({"rows": QueryResult.from_table(table)}, NodeMetrics(
            input_rows=num_rows, output_rows=table.num_rows))


def unnest_batch(batch: pa.RecordBatch, node) -> pa.RecordBatch:
    """Return one row per element of the node's lists, zipped by position.

    Each row repeats as many times as its longest list, with one take of
    the other columns. A list as long as the longest is its flattened
    values, without a copy; a shorter one is taken with a null past its
    end. With ``outer``, a row with no elements counts as one null
    element. An alias is the same array as its field column.
    """
    lists = [batch.column(name) for name in node.lists]
    lengths = [pc.fill_null(pc.list_value_length(values), 0).to_numpy()
               for values in lists]
    longest = np.maximum.reduce(lengths)
    if node.outer:
        longest = np.maximum(longest, 1)
    parents = np.repeat(np.arange(batch.num_rows), longest)
    columns = {name: values.take(parents) for name, values in zip(
        batch.schema.names, batch.columns) if name not in node.lists}
    # each output row's position within its parent row's elements
    position = None
    for name, values, length in zip(node.lists, lists, lengths):
        elements = pc.list_flatten(values)
        if not np.array_equal(length, longest):
            if position is None:
                position = np.arange(len(parents)) - np.repeat(
                    np.cumsum(longest) - longest, longest)
            starts = np.cumsum(length) - length
            elements = elements.take(pa.array(
                starts[parents] + position, mask=position >= length[parents]))
        for field, column in zip(elements.type.names, elements.flatten()):
            columns[f"{name}.{field}"] = column
    for field, name in node.aliases:
        columns[name] = columns[field]
    return pa.RecordBatch.from_arrays(
        [columns[name] for name in node.columns], names=list(node.columns))
