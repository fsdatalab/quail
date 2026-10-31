"""Place each part of a WHERE or ON condition where its columns exist.

The front ends bind every condition that asks no model into a tree of
RegularPredicate, Not, And, and Or. place_conditions normalizes each
tree, splits it at AND, and puts every part at the lowest operator that
has all of its columns, as Spark's predicate pushdown does:

- A part over one table's source columns runs at that table's Scan,
  before any model call.
- A part over one table's computed columns, its labels or its answers
  and spans, and maybe its source columns, runs right after the last
  operator it reads, so the next operator skips the documents it
  rejects. The filter reads each column by document ID.
- Any other part, such as one over two tables, over a label and an
  answer, or over an unnested field, runs on the result rows, after the
  model calls; the Project returns the columns it reads.

For example, ``WHERE NOT (r.stars > 3 OR p.price < 10) AND r.lang =
'en'`` becomes three parts: ``NOT r.stars > 3`` and ``r.lang = 'en'``
at the Scan of ``r``, and ``NOT p.price < 10`` at the Scan of ``p``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

from quail.logical.expressions import (
    Alias,
    ColumnRef,
    CompileError,
    GetField,
    Not,
    RegularPredicate,
    ScalarFunction,
    column_name,
    column_references,
    conjuncts,
    normalize,
)


@dataclass(frozen=True)
class PlacedConditions:
    """The parts of a query's conditions, by where they run.

    Attributes:
        scans: alias -> the conditions on the table's source columns.
        computed: alias -> (condition, selectivity) per condition on
            the table's labels, or on its answers and spans, and maybe
            its source columns.
        rows: The conditions on result rows.
    """

    scans: dict = field(default_factory=dict)
    computed: dict = field(default_factory=dict)
    rows: tuple = ()


def place_conditions(conditions, computed: Mapping[tuple[str, str], str],
                     unnested=()) -> PlacedConditions:
    """Split conditions at AND and place each part where its columns exist.

    Args:
        conditions: (condition, selectivity) pairs, in written order.
            A selectivity is a HINT on the whole condition, which then
            stays one part.
        computed: (alias, column) -> (kind, name) of the operator
            computing it, for every classification's label column
            ("label") and every extraction's answer and span column
            ("extract").
        unnested: The aliases of UNNEST relations.

    Raises:
        CompileError: A HINT is on a part decided without the model's
            answers, at the scan or on the result rows.
    """
    scans, tested, rows = {}, {}, []
    for condition, selectivity in conditions:
        condition = normalize(condition)
        parts = conjuncts(condition) if selectivity is None else (condition,)
        for part in parts:
            part.validate()
            references = [reference for test in part.tests()
                          for reference in column_references(test.expression)]
            aliases = {reference.alias for reference in references}
            owners = {computed.get((reference.alias, reference.column))
                      for reference in references}
            if len(aliases) == 1 and owners == {None} and not aliases & set(unnested):
                if selectivity is not None:
                    raise CompileError(
                        f"HINT estimates a test of a label or an answer; "
                        f"{part} tests a source column, applied at the scan")
                scans.setdefault(aliases.pop(), []).append(part)
            elif (len(aliases) == 1 and not aliases & set(unnested)
                  and len({kind for kind, _ in owners - {None}}) == 1):
                tested.setdefault(aliases.pop(), []).append((part, selectivity))
            else:
                if selectivity is not None:
                    raise CompileError(
                        f"HINT estimates a test that runs before a model "
                        f"call; {part} runs on the result rows, after every "
                        f"model call")
                rows.append(part)
    return PlacedConditions(
        {alias: tuple(found) for alias, found in scans.items()},
        {alias: tuple(found) for alias, found in tested.items()},
        tuple(rows))


def result_row_condition(condition, packing):
    """Rewrite a condition to read result rows, where struct columns hold answers.

    The Project packs each extraction's answer into a field of a struct
    column, such as ``lease.ends`` into the field ``ends`` of ``lease``.
    A test of the answer then reads ``ColumnRef("lease", "", "ends")``,
    named ``lease.ends`` as the result rows name it.

    Args:
        condition: A condition over source, extraction, and unnested
            field columns.
        packing: (struct column, ((field, member column), ...)) pairs,
            as Project.packing() returns them.
    """
    members = {member: (struct, name) for struct, names in packing
               for name, member in names}

    def reference(expression):
        if isinstance(expression, ColumnRef) and expression.column in members:
            struct, name = members[expression.column]
            return ColumnRef(struct, "", name)
        if isinstance(expression, GetField):
            return replace(expression, input=reference(expression.input))
        if isinstance(expression, ScalarFunction):
            return replace(expression,
                           args=tuple(reference(arg) for arg in expression.args))
        if isinstance(expression, Alias):
            return replace(expression, expression=reference(expression.expression))
        return expression

    def rewrite(node):
        if isinstance(node, RegularPredicate):
            return replace(node, expression=reference(node.expression))
        if isinstance(node, Not):
            return Not(rewrite(node.operand))
        return type(node)(tuple(rewrite(operand) for operand in node.operands))

    return rewrite(condition)


def result_columns_read(condition) -> tuple[str, ...]:
    """Return the result row columns a rewritten condition reads, by name."""
    return tuple(dict.fromkeys(
        column_name(reference) for test in condition.tests()
        for reference in column_references(test.expression)))
