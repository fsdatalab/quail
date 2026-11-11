"""Ask a guarded model node about its guard's rows alone.

A guarded column node answers NULL for every other row. An AI filter in
an OR passes each row where the OR's plain terms, its alternative, are
TRUE, whether or not it asked the model about that row. When the OR has
several AI filters, the rows carry whether a term has passed them in
the OR's column, in whatever order the filters run, until the OR's last
filter returns the rows a term passed. A short-circuit filter asks only
the rows no earlier one passed.
An AI filter that returns its answer keeps every row it reads and adds
the answer column: TRUE or FALSE where it asked, NULL elsewhere.

A row whose document is NULL is guarded the same way, whatever the
plan: the model is not asked about it, a column node answers NULL for
it, and an AI filter does not pass it. An AI join reads no such row:
a pair with a NULL document does not match.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pyarrow as pa
import pyarrow.compute as pc

from quail.common.expressions import accepts
from quail.common.pairs import NULL_DOCUMENT, columns_key
from quail.execution.metrics import NodeResult
from quail.execution.rows import Rows, rows_of
from quail.physical import (
    AiClassify,
    AiExtract,
    AiFilter,
    AiJoin,
    AiMap,
    AiSaeFeatures,
    AiScore,
)

DOCUMENT_NODES = (AiFilter, AiClassify, AiExtract, AiMap, AiSaeFeatures, AiScore)


@dataclass
class Guarded:
    """A guarded node's rows, before and after its guard.

    Attributes:
        node: The guarded column node or AI filter.
        port: The name of its rows input.
        rows: The rows it read.
        asked: Whether the model is asked about each row it read: its
            guard is TRUE, and for a short-circuit filter no earlier
            term of its OR passed the row.
        passes: For an AI filter, whether a term of its OR has passed
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
            if self.node.answer is not None:
                return self._answered(result)
            if self.node.disjunction is None or self.node.closes_disjunction:
                return self._filtered(result)
            return self._undecided(result)
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


    def _answered(self, result: NodeResult) -> NodeResult:
        """Keep every row read, with the answer where asked and NULL elsewhere."""
        alias = self.node.alias
        port = f"rows:{alias}"
        survivors = result.outputs[port].ids[alias] if isinstance(
            result.outputs[port], Rows) else result.outputs[port]
        passed = pc.is_in(self.rows.ids[alias], value_set=survivors)
        unasked = pa.nulls(self.rows.num_rows, pa.bool_())
        answers = pc.if_else(self.asked, passed, unasked)
        rows = self.rows.with_column(self.node.answer, answers)
        metrics = replace(result.metrics, input_rows=self.rows.num_rows,
                          output_rows=rows.num_rows)
        return NodeResult({**result.outputs, port: rows}, metrics)

    def _undecided(self, result: NodeResult) -> NodeResult:
        """Keep every row read, with whether a term of the OR has passed it."""
        alias = self.node.alias
        port = f"rows:{alias}"
        survivors = result.outputs[port].ids[alias] if isinstance(
            result.outputs[port], Rows) else result.outputs[port]
        passed = pc.is_in(self.rows.ids[alias], value_set=survivors)
        decided = pc.or_(self.passes, pc.and_(self.asked, passed))
        name = self.node.disjunction
        values = self.rows.values
        if name in values.column_names:
            values = values.drop_columns([name])
        rows = Rows(self.rows.ids, values.append_column(name, decided))
        metrics = replace(result.metrics, input_rows=self.rows.num_rows,
                          output_rows=rows.num_rows)
        return NodeResult({**result.outputs, port: rows}, metrics)

    def _filtered(self, result: NodeResult) -> NodeResult:
        """Keep the asked rows that passed and the rows a term of the OR passed."""
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


def _document_alias(node) -> str | None:
    """Return the alias whose one document the node asks about, if it has one."""
    if isinstance(node, AiFilter):
        return node.alias
    if isinstance(node, DOCUMENT_NODES) and len(node.spec.aliases) == 1:
        return node.spec.aliases[0]
    return None


def reads_null_documents(node, context) -> bool:
    """Return whether the node asks about a table that has a NULL document."""
    alias = _document_alias(node)
    columns = None if alias is None else context.sources.get(columns_key(alias))
    return columns is not None and NULL_DOCUMENT in columns.column_names


def present_join_inputs(node, inputs: dict, context) -> dict:
    """Return an AI join's inputs without the rows whose document is NULL.

    An anti join keeps its anchors: one with a NULL document matches no
    partner, so the join returns it.

    Args:
        node: The node to run.
        inputs: Its input port values.
        context: The execution context.
    """
    if not isinstance(node, AiJoin):
        return inputs
    anti = not bool(node.keeps(True))
    found = dict(inputs)
    for port in node.inputs:
        value = inputs.get(port.name)
        named = port.source.port.partition(":")
        if value is None or named[0] not in ("rows", "pairs"):
            continue
        aliases = value.aliases if isinstance(value, Rows) else (named[2],)
        for alias in aliases:
            columns = context.sources.get(columns_key(alias))
            if (columns is None or NULL_DOCUMENT not in columns.column_names
                    or (anti and alias == node.anchor)):
                continue
            rows = rows_of(found[port.name], alias)
            present = pc.invert(pc.take(columns.column(NULL_DOCUMENT), rows.ids[alias]))
            kept = rows.filter(present)
            found[port.name] = kept if isinstance(value, Rows) else kept.ids[alias]
    return found


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
    answer = getattr(node, "answer", None)
    disjunction = getattr(node, "disjunction", None)
    absent = reads_null_documents(node, context)
    if (guard is None and alternative is None and answer is None
            and disjunction is None and not absent):
        return None
    (port,) = [port.name for port in node.inputs]
    alias = node.alias if isinstance(node, AiFilter) else node.spec.aliases[0]
    rows = rows_of(inputs[port], alias)
    read = _read(node, rows, alias, context, [
        condition for condition in (guard, alternative) if condition is not None])
    everyone = pa.array([True] * rows.num_rows, pa.bool_())
    asked = everyone if guard is None else accepts(guard, read)
    passes = (pc.invert(everyone) if alternative is None
              else pc.fill_null(accepts(alternative, read), False))
    if disjunction in rows.column_names:
        earlier = pc.fill_null(rows.column(disjunction), False)
        passes = pc.or_(passes, earlier)
        if node.short_circuit:
            asked = pc.and_(pc.fill_null(asked, False), pc.invert(earlier))
    if absent:
        null_documents = context.sources[columns_key(alias)].column(NULL_DOCUMENT)
        present = pc.invert(pc.take(null_documents, rows.ids[alias]))
        asked = pc.and_(pc.fill_null(asked, False), present)
    return Guarded(node, port, rows, asked, passes)


def _read(node, rows: Rows, alias: str, context, conditions) -> pa.Table:
    """Return the columns the conditions read, one row per row read.

    A column is a value the rows carry, or a column of the alias's table.
    """
    names = {ref.column for condition in conditions
             for ref in condition.references}
    carried = [name for name in rows.column_names if name in names]
    read = rows.values.select(carried)
    if names <= set(carried):
        return read if carried else pa.table({"_": pa.nulls(rows.num_rows)})
    columns = context.sources.get(columns_key(alias))
    if columns is None:
        raise KeyError(f"{node.node_id!r} reads its guard from {alias!r}'s columns")
    taken = columns.take(rows.ids[alias])
    for name in carried:
        taken = taken.append_column(name, read.column(name))
    return taken
