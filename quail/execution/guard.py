"""Ask a guarded column node about its guard's rows, and answer the rest NULL."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pyarrow as pa
import pyarrow.compute as pc

from quail.common.expressions import accepts
from quail.common.pairs import columns_key
from quail.execution.metrics import NodeResult
from quail.execution.rows import Rows, rows_of


@dataclass
class Guarded:
    """A guarded node's rows, before and after its guard.

    Attributes:
        node: The guarded column node.
        port: The name of its rows input.
        rows: The rows it read.
        asked: Whether its guard is TRUE for each row it read.
    """

    node: object
    port: str
    rows: Rows
    asked: pa.Array

    def inputs(self, inputs: dict) -> dict:
        """Return the node's inputs with only the rows its guard accepts."""
        return {**inputs, self.port: self.rows.filter(self.asked)}

    def expanded(self, result: NodeResult) -> NodeResult:
        """Give every row the node read its answer, or NULL where not asked.

        Args:
            result: The node's result over the rows its guard accepts.
        """
        port = self.node.rows_port
        answered = result.outputs[port]
        alias = self.node.spec.aliases[0]
        positions = pc.index_in(self.rows.ids[alias],
                                value_set=answered.ids[alias])
        # a row asked and given no answer row was dropped by the node
        kept = pc.or_(pc.invert(self.asked), pc.is_valid(positions))
        rows = self.rows
        for name in answered.column_names:
            if name not in rows.column_names:
                rows = rows.with_column(name, pc.take(answered.column(name), positions))
        rows = rows.filter(kept)
        metrics = replace(result.metrics, input_rows=self.rows.num_rows,
                          output_rows=rows.num_rows)
        return NodeResult({**result.outputs, port: rows}, metrics)


def guarded(node, inputs: dict, context) -> Guarded | None:
    """Evaluate a column node's guard over the rows it reads.

    Args:
        node: The node to run.
        inputs: Its input port values.
        context: The execution context; its sources hold each table's
            column values by document id.

    Returns:
        The rows and which of them the model is asked about, or None for
        a node without a guard.
    """
    guard = getattr(node, "guard", None)
    if guard is None:
        return None
    (port,) = [port.name for port in node.inputs]
    alias = node.spec.aliases[0]
    rows = rows_of(inputs[port], alias)
    columns = context.sources.get(columns_key(alias))
    if columns is None:
        raise KeyError(f"{node.node_id!r} reads its guard from {alias!r}'s columns")
    return Guarded(node, port, rows, accepts(guard, columns.take(rows.ids[alias])))
