"""The expressions of a plan's operators, and their evaluation by DataFusion.

The expression classes are Rust's, from ``_quail_native``: a value is a
ColumnRef, a GetField of one, an Alias, or an Expression, which holds a
DataFusion expression; a condition is an Expression. ``evaluate`` hands
one to DataFusion with the rows it reads; Python never reads an
Expression's form.
"""

from __future__ import annotations

import _quail_native
import pyarrow as pa
from _quail_native import (
    Alias,
    ColumnRef,
    Expression,
    GetField,
    ModelCall,
    Prompt,
    ScoreTest,
)
from pyarrow import compute as pc

__all__ = [
    "Alias", "ColumnRef", "Expression", "GetField", "ModelCall", "Prompt",
    "ScoreTest",
]

DEFAULT_SELECTIVITY = 0.2
SCORE_COMPARISONS = ("<", "<=", ">", ">=")
# the text before each label of a classification prompt
LABEL_PREFIX = " "
PROBABILITIES_SUFFIX = "_probabilities"
# an "extract" call's second result column: the answer's character
# span in the document
SPAN_SUFFIX = "_span"
# the Arrow type of one feature list
FEATURE_LIST = pa.list_(
    pa.struct([("feature", pa.int32()), ("strength", pa.float32())]))


class CompileError(ValueError):
    """Raised when a query is malformed or outside the language."""


def effective_selectivity(selectivity: float | None) -> float:
    """Return the selectivity the planner uses: the given one or the default."""
    return DEFAULT_SELECTIVITY if selectivity is None else selectivity


def parse_expression(sql: str, rows, alias: str = "", provider: str = ""):
    """Return the Expression of SQL text over the columns of rows.

    For example, ``parse_expression("stars > 3", reviews, "r", "reviews")``
    is the condition a scan of reviews as ``r`` tests.

    Args:
        sql: A SQL value or condition, such as ``stars > 3``.
        rows: A table or record batch whose schema types the columns.
        alias: The table alias each column belongs to.
        provider: The table each column is read from.

    Raises:
        CompileError: The text is not SQL over the rows' columns.
    """
    found, message = _quail_native.parse_expression(sql, rows, alias, provider)
    if message is not None:
        raise CompileError(message)
    return _quail_native.decode_plan_value(found)


def evaluate(expression, rows, *, qualified: bool = False) -> pa.Array:
    """Evaluate a value or condition over rows with DataFusion.

    Args:
        expression: A ColumnRef, GetField, Alias, or Expression.
        rows: A table or record batch holding the columns it reads, or
            an object that reads them with ``column(name)``, such as a
            scan's input.
        qualified: Whether the rows name each column with its table
            alias, as result rows do.

    Returns:
        One value per row; a condition's are booleans, null where SQL's
        three-valued logic leaves a row unknown.

    Raises:
        CompileError: The rows lack a column the expression reads, or
            DataFusion cannot evaluate it on the columns' types.
    """
    if not isinstance(rows, (pa.Table, pa.RecordBatch)):
        names = dict.fromkeys(
            f"{reference.alias}.{reference.column}"
            if qualified and reference.alias else reference.column
            for reference in expression.references)
        rows = pa.table({name: rows.column(name) for name in names})
    values, message = _quail_native.evaluate(expression, rows, qualified)
    if message is not None:
        raise CompileError(message)
    return values


def accepted_rows(mask) -> pa.Array:
    """Return a condition's mask with every null rejected, as one array."""
    mask = pc.fill_null(mask, False)
    return mask.combine_chunks() if isinstance(mask, pa.ChunkedArray) else mask


def accepts(condition, rows, *, qualified: bool = False) -> pa.Array:
    """Return the rows a condition accepts, with every null rejected."""
    return accepted_rows(evaluate(condition, rows, qualified=qualified))
