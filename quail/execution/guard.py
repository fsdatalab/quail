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
a pair with a NULL document does not match. A column node over pairs
is asked about the pairs of two documents and answers NULL for the rest.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
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

    def reports_as_run(self) -> bool:
        """Whether each row the node reports as it runs is a row of its result.

        True for a column node, an AI filter that closes its OR, and an
        AI filter outside an OR that returns its answer: each row the
        node reports stays in the result, which only adds rows, with a
        NULL answer where the model was not asked. An AI filter that
        leaves its OR undecided gives every row a column only its whole
        result has. A join reports pairs, which go with its whole result.
        """
        if isinstance(self.node, AiJoin):
            return False
        if not isinstance(self.node, AiFilter):
            return True
        if self.node.answer is not None:
            return self.node.disjunction is None
        return self.node.disjunction is None or self.node.closes_disjunction

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
        found = result.outputs[port]
        if isinstance(found, Rows) and self.node.answer in found.column_names:
            # the runtime wrote the answer of each row it asked
            positions = pc.index_in(self.rows.ids[alias], value_set=found.ids[alias])
            rows = self.rows
            for name in found.column_names:
                if name not in rows.column_names:
                    rows = rows.with_column(
                        name, pc.take(found.column(name), positions))
            metrics = replace(result.metrics, input_rows=self.rows.num_rows,
                              output_rows=rows.num_rows)
            return NodeResult({**result.outputs, port: rows}, metrics)
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


@dataclass
class NullPairs:
    """A pair column node's pairs with a NULL document, which it is not asked about.

    Attributes:
        node: The column node over two tables' rows.
        present: Each input port's rows that have a document.
        unasked: The pairs with a NULL document, with the values their
            rows carry.
    """

    node: object
    present: dict
    unasked: Rows

    def inputs(self, inputs: dict) -> dict:
        """Return the node's inputs with only the rows that have a document."""
        return {**inputs, **self.present}

    def reports_as_run(self) -> bool:
        """Whether each row the node reports as it runs is a row of its result.

        Always: the result adds the unasked pairs to the asked ones.
        """
        return True

    def expanded(self, result: NodeResult) -> NodeResult:
        """Add the unasked pairs to the node's rows, with NULL answers."""
        port = self.node.rows_port
        answered = result.outputs[port]
        unasked = Rows({alias: self.unasked.ids[alias] for alias in answered.aliases})
        for name in answered.column_names:
            carried = (self.unasked.column(name) if name in self.unasked.column_names
                       else pa.nulls(unasked.num_rows, answered.column(name).type))
            unasked = unasked.with_column(name, carried)
        rows = Rows.concat([answered, unasked])
        metrics = replace(result.metrics, output_rows=rows.num_rows)
        return NodeResult({**result.outputs, port: rows}, metrics)


def _null_pairs(node, inputs: dict, context) -> NullPairs | None:
    """Return the pairs a two-table column node is not asked about, if it has any."""
    if isinstance(node, (AiFilter, AiJoin)) or not isinstance(node, DOCUMENT_NODES):
        return None
    sides = [inputs.get(port.name) for port in node.inputs]
    if len(sides) != 2 or not all(
            isinstance(side, Rows) and len(side.aliases) == 1 for side in sides):
        return None
    absent = []
    for side in sides:
        (alias,) = side.aliases
        columns = context.sources.get(columns_key(alias))
        absent.append(
            np.zeros(side.num_rows, bool)
            if columns is None or NULL_DOCUMENT not in columns.column_names
            else pc.take(columns.column(NULL_DOCUMENT), side.ids[alias]).to_numpy(
                zero_copy_only=False).astype(bool))
    if not absent[0].any() and not absent[1].any():
        return None
    left = np.repeat(np.arange(sides[0].num_rows), sides[1].num_rows)
    right = np.tile(np.arange(sides[1].num_rows), sides[0].num_rows)
    unasked = absent[0][left] | absent[1][right]
    first, second = sides[0].take(left[unasked]), sides[1].take(right[unasked])
    pairs = Rows({**first.ids, **second.ids})
    for carried in (first, second):
        for name in carried.column_names:
            pairs = pairs.with_column(name, carried.column(name))
    present = {port.name: side.filter(pa.array(~mask))
               for port, side, mask in zip(node.inputs, sides, absent)}
    return NullPairs(node, present, pairs)


def _document_alias(node) -> str | None:
    """Return the alias whose one document the node asks about, if it has one."""
    if isinstance(node, AiFilter):
        return node.alias
    if isinstance(node, DOCUMENT_NODES) and len(node.spec.aliases) == 1:
        return node.spec.aliases[0]
    return None


def reads_null_documents(node, context) -> bool:
    """Return whether the node asks about a table that has a NULL document."""
    aliases = [_document_alias(node)]
    if aliases == [None] and isinstance(node, DOCUMENT_NODES) and not isinstance(
            node, AiFilter):
        aliases = node.spec.aliases
    return any(
        alias is not None
        and (columns := context.sources.get(columns_key(alias))) is not None
        and NULL_DOCUMENT in columns.column_names
        for alias in aliases)


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


def guarded(node, inputs: dict, context, *, runtime_answers: bool = False,
            answers: Mapping[str, Rows] | None = None) -> Guarded | NullPairs | None:
    """Evaluate a column node's guard over the rows it reads.

    Args:
        node: The node to run.
        inputs: Its input port values.
        context: The execution context; its sources hold each table's
            column values by document id.
        runtime_answers: Whether the runtime writes an AI filter's answer
            column itself, keeping every row it reads, as Quail's GPU
            executor does.
        answers: The rows that hold each answer column the query's
            nodes have returned so far, by column name.

    Returns:
        The rows and which of them the model is asked about, or None for
        a node without a guard.
    """
    if (pairs := _null_pairs(node, inputs, context)) is not None:
        return pairs
    guard = getattr(node, "guard", None)
    alternative = getattr(node, "alternative", None)
    # a runtime that writes a filter's answer column needs no guard for it
    answer = None if runtime_answers else getattr(node, "answer", None)
    disjunction = getattr(node, "disjunction", None)
    absent = _document_alias(node) is not None and reads_null_documents(node, context)
    if (guard is None and alternative is None and answer is None
            and disjunction is None and not absent):
        return None
    (port,) = [port.name for port in node.inputs]
    alias = node.alias if isinstance(node, AiFilter) else node.spec.aliases[0]
    rows = rows_of(inputs[port], alias)
    read = _read(node, rows, alias, context, [
        condition for condition in (guard, alternative) if condition is not None],
        answers or {})
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


def _read(node, rows: Rows, alias: str, context, conditions,
          answers: Mapping[str, Rows]) -> pa.Table:
    """Return the columns the conditions read, one row per row read.

    A column is a value the rows carry, an answer an earlier node
    returned for the same documents, or a column of the alias's table.
    """
    names = {ref.column for condition in conditions
             for ref in condition.references}
    read = rows.values.select([name for name in rows.column_names if name in names])
    for name in sorted(names - set(read.column_names)):
        answered = answers.get(name)
        if answered is None or alias not in answered.aliases:
            continue
        # the answering node asked about every row this node reads, or
        # answered NULL where it did not
        positions = pc.index_in(rows.ids[alias], value_set=answered.ids[alias])
        read = read.append_column(name, pc.take(answered.column(name), positions))
    if names <= set(read.column_names):
        return read if read.column_names else pa.table({"_": pa.nulls(rows.num_rows)})
    columns = context.sources.get(columns_key(alias))
    if columns is None:
        raise KeyError(f"{node.node_id!r} reads its guard from {alias!r}'s columns")
    taken = columns.take(rows.ids[alias])
    for name in read.column_names:
        taken = taken.append_column(name, read.column(name))
    return taken
