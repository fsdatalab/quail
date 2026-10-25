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

from quail.logical import CompileError


def take_unnests(tree: exp.Select) -> tuple[tuple[tuple[str, ...], tuple, bool], ...]:
    """Remove each UNNEST from FROM and return its list names and calls.

    ``LEFT JOIN UNNEST(...) AS f``, with no ON or with ON TRUE, keeps a
    row with no elements as one row of nulls.

    Returns:
        One (names, AI.SAE_FEATURES calls, outer) triple per UNNEST, in
        FROM order; outer is whether the UNNEST is a LEFT JOIN.

    Raises:
        CompileError: An UNNEST reads anything but AI.SAE_FEATURES
            calls, or does not name each list.
    """
    taken = []
    for join in [join for join in tree.args.get("joins") or ()
                 if isinstance(join.this, exp.Unnest)]:
        outer = join.side == "LEFT"
        on = join.args.get("on")
        if (join.side not in ("", "LEFT")
                or join.kind not in (("", "OUTER") if outer else ("", "CROSS"))
                or (on is not None and not (outer and on == exp.true()))
                or any(join.args.get(key) for key in ("using", "method"))):
            raise CompileError(
                "UNNEST supports comma joins, CROSS JOIN, and LEFT JOIN with "
                "no ON or ON TRUE")
        unnest = join.this
        if unnest.args.get("offset"):
            raise CompileError("UNNEST does not support WITH ORDINALITY or OFFSET")
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
        for column in tree.find_all(exp.Column):
            if alias is not None and column.db == alias.name and (
                    column.table in columns):
                column.set("db", None)
        join.pop()
        taken.append((tuple(columns), calls, outer))
    return tuple(taken)
