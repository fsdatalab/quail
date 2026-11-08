"""Runtimes for the relational nodes: scan and limit."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from quail.common.pairs import columns_key
from quail.execution.ids import document_ids
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.rows import Rows
from quail.physical import Limit, Scan


class ScanRuntime:
    """Read a prepared source registered by document alias."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Scan):
            raise TypeError(type(node).__name__)
        if node.input_id not in context.sources:
            raise KeyError(
                f"no prepared source for input {node.input_id!r}"
            )
        value = document_ids(context.sources[node.input_id])
        if node.predicates:
            tested = document_ids(_tested_documents(node, context))
            # a GPU's shard holds some of the table's documents
            value = pc.filter(tested, pc.is_in(tested, value_set=value))
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


class LimitRuntime:
    """Keep the first rows of a relation."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Limit):
            raise TypeError(type(node).__name__)
        if len(inputs) != 1:
            raise ValueError("Limit needs one input")
        value = next(iter(inputs.values()))
        rows = len(value)
        kept = min(rows, node.count)
        value = (value.take(range(kept)) if isinstance(value, Rows)
                 else value[:kept])
        return NodeResult({"rows": value}, NodeMetrics(
            input_rows=rows, output_rows=kept))
