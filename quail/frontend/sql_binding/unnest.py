"""Bind the UNNEST relations of FROM and the tests of their field columns.

``FROM t, UNNEST(AI.SAE_FEATURES(...)) AS f`` names one list ``f``,
whose elements the query reads as the columns ``f.feature`` and
``f.strength``. ``UNNEST(a, b) AS u(x, y)`` zips two lists, named by
the column aliases ``x`` and ``y``. Each UNNEST in FROM is one logical
Unnest above the ones before it, so two of them return the product of
their lists.
"""

from __future__ import annotations

from sqlglot import exp

from quail.frontend.sql_binding.predicates import parse_plain_test
from quail.logical import (
    FEATURE_LIST,
    ColumnRef,
    CompileError,
    RegularPredicate,
    column_name,
)


def take_unnests(tree: exp.Select) -> tuple[tuple[tuple[str, ...], tuple], ...]:
    """Remove each UNNEST from FROM and return its list names and calls.

    Returns:
        One (names, AI.SAE_FEATURES calls) pair per UNNEST, in FROM
        order.

    Raises:
        CompileError: An UNNEST is an outer join, has an ON condition or
            WITH ORDINALITY, reads anything but AI.SAE_FEATURES calls, or
            does not name each list.
    """
    taken = []
    for join in [join for join in tree.args.get("joins") or ()
                 if isinstance(join.this, exp.Unnest)]:
        unnest = join.this
        if join.side or join.args.get("on") or join.args.get("using"):
            raise CompileError(
                f"UNNEST is a cross join: write FROM t, UNNEST(...) AS f; "
                f"{join.sql()} is not supported")
        if unnest.args.get("offset"):
            raise CompileError(
                "UNNEST ... WITH ORDINALITY or WITH OFFSET is not supported")
        calls = tuple(unnest.expressions)
        if not calls or not all(
                isinstance(call, exp.Anonymous)
                and str(call.this).upper() == "AI_SAE_FEATURES" for call in calls):
            raise CompileError(
                f"UNNEST reads AI.SAE_FEATURES calls, got {unnest.sql()}")
        alias = unnest.args.get("alias")
        columns = [column.name for column in alias.columns] if alias else []
        if not columns and alias is not None and alias.this is not None:
            columns = [alias.this.name]
        if len(columns) != len(calls):
            raise CompileError(
                f"UNNEST names each list it reads: AS f for one list, or "
                f"AS u(x, y) for two; got {unnest.sql()}")
        join.pop()
        taken.append((tuple(columns), calls))
    return tuple(taken)


def list_fields() -> tuple[str, ...]:
    """Return the field names of one element of a feature list."""
    return tuple(FEATURE_LIST.value_type.names)


def unnest_field(node, unnested: dict) -> str | None:
    """Return the name of the field column a column reads, or None.

    Args:
        node: A parsed expression.
        unnested: Each unnested list's name and its field names.

    Raises:
        CompileError: The column reads an unnested list but names no
            field of it.
    """
    if not isinstance(node, exp.Column) or node.table not in unnested:
        return None
    fields = unnested[node.table]
    if node.name not in fields:
        raise CompileError(
            f"the unnested list {node.table!r} has the fields "
            f"{', '.join(f'{node.table}.{field}' for field in fields)}, "
            f"got {node.sql()}")
    return f"{node.table}.{node.name}"


def field_reference(b, node):
    """Return the expression later clauses read an unnested field by, or None.

    A field with a SELECT name is read by its first Alias; any other by
    its ColumnRef.

    Raises:
        CompileError: The column reads an unnested list but names no
            field of it.
    """
    name = unnest_field(node, b.unnested)
    if name is None:
        return None
    return next((alias for alias in b.unnest_aliases
                 if column_name(alias.expression) == name),
                ColumnRef(node.table, "", node.name))


def bind_unnest_test(b, term) -> RegularPredicate | None:
    """Return a WHERE term's test of an unnested field column, or None.

    Raises:
        CompileError: The term mixes a field column with other columns,
            or is not a test of one field against literals.
    """
    columns = list(term.find_all(exp.Column))
    reads = [column for column in columns if column.table in b.unnested]
    if not reads:
        return None
    if len(reads) != len(columns) or len({column.sql() for column in reads}) != 1:
        raise CompileError(
            f"a WHERE test of an unnested list reads one field column, got "
            f"{term.sql()}")
    test = parse_plain_test(term, b.dialect)
    if (test is None or test.function is not None
            or not isinstance(test.operand, exp.Column)):
        raise CompileError(
            f"a WHERE test of an unnested list compares one field with "
            f"literals, got {term.sql()}")
    unnest_field(test.operand, b.unnested)
    predicate = RegularPredicate(
        ColumnRef(test.operand.table, "", test.operand.name), test.comparison,
        test.value)
    predicate.validate()
    return predicate
