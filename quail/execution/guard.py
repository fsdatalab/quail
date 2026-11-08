"""Ask a guarded model node about its guard's rows alone.

A guarded column node answers NULL for every other row. An AI filter in
an OR also passes each row where the OR's other terms, its alternative,
are TRUE, whether or not it asked the model about that row.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pyarrow as pa
import pyarrow.compute as pc

from quail.common.expressions import accepts
from quail.common.pairs import columns_key
from quail.execution.metrics import NodeResult
from quail.execution.rows import Rows, rows_of
from quail.physical import AiFilter


@dataclass
class Guarded:
    """A guarded node's rows, before and after its guard.

    Attributes:
        node: The guarded column node or AI filter.
        port: The name of its rows input.
        rows: The rows it read.
        asked: Whether its guard is TRUE for each row it read.
        passes: For an AI filter, whether its alternative is TRUE for
            each row it read; all false for any other node.
    """

    node: object
    port: str
    rows: Rows
    asked: pa.Array
    passes: pa.Array

    def inputs(self, inputs: dict) -> dict:
        """Return the node's inputs with only the rows its guard accepts."""
        asked = self.rows.filter(self.asked)
        if not isinstance(inputs[self.port], Rows):
            return {**inputs, self.port: asked.ids[asked.aliases[0]]}
        return {**inputs, self.port: asked}

    def expanded(self, result: NodeResult) -> NodeResult:
        """Give every row the node read its answer, or NULL where not asked.

        Args:
            result: The node's result over the rows its guard accepts.
        """
        if isinstance(self.node, AiFilter):
            return self._filtered(result)
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


    def _filtered(self, result: NodeResult) -> NodeResult:
        """Keep the asked rows that passed and the rows the alternative passes."""
        alias = self.node.alias
        port = f"rows:{alias}"
        survivors = result.outputs[port].ids[alias] if isinstance(
            result.outputs[port], Rows) else result.outputs[port]
        passed = pc.is_in(self.rows.ids[alias], value_set=survivors)
        kept = pc.or_(pc.and_(self.asked, passed), self.passes)
        rows = Rows({alias: self.rows.ids[alias]}).filter(kept)
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
    alternative = getattr(node, "alternative", None)
    if guard is None and alternative is None:
        return None
    (port,) = [port.name for port in node.inputs]
    alias = node.alias if isinstance(node, AiFilter) else node.spec.aliases[0]
    rows = rows_of(inputs[port], alias)
    columns = context.sources.get(columns_key(alias))
    if columns is None:
        raise KeyError(f"{node.node_id!r} reads its guard from {alias!r}'s columns")
    read = columns.take(rows.ids[alias])
    everyone = pa.array([True] * rows.num_rows, pa.bool_())
    asked = everyone if guard is None else accepts(guard, read)
    passes = (pc.invert(everyone) if alternative is None
              else accepts(alternative, read))
    return Guarded(node, port, rows, asked, passes)
