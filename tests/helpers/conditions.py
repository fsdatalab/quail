"""Conditions for hand-built physical plans, written as SQL."""

import pyarrow as pa

from quail.common.expressions import parse_expression


def condition(sql: str, columns: dict, alias: str = "", provider: str = ""):
    """Return the Expression of a SQL condition over typed columns.

    For example, ``condition("stars > 3", {"stars": pa.int64()}, "r")``
    is the test a Filter of the reviews ``r`` makes.

    Args:
        sql: The condition, such as ``stars > 3``; a column whose name
            has a dot is written in double quotes.
        columns: Each column's name and Arrow type.
        alias: The table alias each column belongs to.
        provider: The table each column is read from.
    """
    rows = pa.schema(list(columns.items())).empty_table()
    return parse_expression(sql, rows, alias, provider)
