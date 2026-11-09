"""The expressions of a plan's operators, and their evaluation by DataFusion.

The expression classes are Rust's, from ``_quail_native``: a value is a
ColumnRef, a GetField of one, an Alias, or an Expression, which holds a
DataFusion expression; a condition is an Expression. Each is built for
DataFusion once, when it is read from a plan, and its ``evaluate(rows)``
method only evaluates it; Python never reads an Expression's form.
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


class SimilarCallWarning(UserWarning):
    """Two AI calls of a query differ by a typo, so the model answers each."""


def effective_selectivity(selectivity: float | None) -> float:
    """Return the selectivity the planner uses: the given one or the default."""
    return DEFAULT_SELECTIVITY if selectivity is None else selectivity


def parse_expression(sql: str, rows, alias: str = "", provider: str = ""):
    """Return the Expression of SQL text over the columns of rows.

    The Expression is built for evaluation once, here.

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
    return found


def accepted_rows(mask) -> pa.Array:
    """Return a condition's mask with every null rejected, as one array."""
    mask = pc.fill_null(mask, False)
    return mask.combine_chunks() if isinstance(mask, pa.ChunkedArray) else mask


def accepts(condition, rows, *, qualified: bool = False) -> pa.Array:
    """Return the rows a condition accepts, with every null rejected."""
    return accepted_rows(condition.evaluate(rows, qualified=qualified))
