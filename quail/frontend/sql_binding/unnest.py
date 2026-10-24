"""Bind ``FROM t, UNNEST(AI.SAE_FEATURES(...)) AS f`` and the tests of its fields.

The UNNEST names one AI.SAE_FEATURES call. The call becomes a projected
feature list column ``f``, and the query reads its elements as the
columns ``f.feature`` and ``f.strength``. A WHERE test of one of them
keeps the elements it accepts.
"""

from __future__ import annotations

from sqlglot import exp

from quail.frontend.sql_binding.predicates import parse_plain_test
from quail.logical import ColumnRef, CompileError, RegularPredicate
from quail.logical.expressions import UNNEST_FIELDS


def take_unnest(tree: exp.Select) -> tuple[str, exp.Expression] | None:
    """Remove the UNNEST from FROM and return its alias and its call.

    Returns:
        (alias, AI.SAE_FEATURES call), or None when FROM has no UNNEST.

    Raises:
        CompileError: FROM has more than one UNNEST, the UNNEST has no
            alias, or it reads anything but one AI.SAE_FEATURES call.
    """
    joins = [join for join in tree.args.get("joins") or ()
             if isinstance(join.this, exp.Unnest)]
    if not joins:
        return None
    if len(joins) > 1:
        raise CompileError("a query reads at most one UNNEST")
    unnest = joins[0].this
    calls = unnest.expressions
    if len(calls) != 1 or not (isinstance(calls[0], exp.Anonymous)
                               and str(calls[0].this).upper() == "AI_SAE_FEATURES"):
        raise CompileError("UNNEST reads one AI.SAE_FEATURES call")
    alias = unnest.args.get("alias")
    names = ([alias.this] if alias is not None and alias.this is not None
             else list(alias.columns) if alias is not None else [])
    if len(names) != 1:
        raise CompileError("UNNEST(AI.SAE_FEATURES(...)) needs one AS name")
    joins[0].pop()
    return names[0].name, calls[0]


def unnest_field(node, alias: str | None) -> str | None:
    """Return the field a column reads from the UNNEST, or None.

    Raises:
        CompileError: The column reads the UNNEST but names no field.
    """
    if alias is None or not isinstance(node, exp.Column) or node.table != alias:
        return None
    if node.name not in UNNEST_FIELDS:
        raise CompileError(
            f"UNNEST {alias!r} has the fields {alias}.feature and "
            f"{alias}.strength, got {node.sql()}")
    return node.name


def bind_unnest_test(b, term, alias: str | None) -> RegularPredicate | None:
    """Return a WHERE term's test of an UNNEST field, or None.

    Raises:
        CompileError: The term mixes an UNNEST field with other columns,
            or is not a test of one field against literals.
    """
    if alias is None:
        return None
    columns = list(term.find_all(exp.Column))
    reads = [column for column in columns if column.table == alias]
    if not reads:
        return None
    if len(reads) != len(columns):
        raise CompileError(
            f"a WHERE test of UNNEST {alias!r} reads its fields only, got "
            f"{term.sql()}")
    test = parse_plain_test(term, b.dialect)
    if (test is None or test.function is not None
            or not isinstance(test.operand, exp.Column)):
        raise CompileError(
            f"a WHERE test of UNNEST {alias!r} compares one field with "
            f"literals, got {term.sql()}")
    field = unnest_field(test.operand, alias)
    predicate = RegularPredicate(ColumnRef(alias, "", field), test.comparison,
                                 test.value)
    predicate.validate()
    return predicate
