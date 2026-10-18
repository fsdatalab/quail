"""The Arrow kernels that compare a column with literals, one per comparison.

RegularPredicate, in quail.logical.expressions, evaluates itself with
them.
"""

from __future__ import annotations

import math
from functools import partial

import pyarrow as pa
from pyarrow import compute as pc

VALUE_COMPARISONS = ("=", "<>", "<", "<=", ">", ">=")
TEXT_COMPARISONS = ("like", "ilike", "contains", "icontains", "starts with",
                    "ends with", "regex")
COLUMN_COMPARISONS = (
    *VALUE_COMPARISONS, "in", "is null", "is not null",
    *TEXT_COMPARISONS, *(f"not {comparison}" for comparison in TEXT_COMPARISONS))

COMPARISON_KERNELS = {
    "=": pc.equal, "<>": pc.not_equal,
    "<": pc.less, "<=": pc.less_equal,
    ">": pc.greater, ">=": pc.greater_equal,
    "like": pc.match_like,
    "ilike": partial(pc.match_like, ignore_case=True),
    "contains": pc.match_substring,
    "icontains": partial(pc.match_substring, ignore_case=True),
    "starts with": pc.starts_with,
    "ends with": pc.ends_with,
    "regex": pc.match_substring_regex,
}


def _numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _integer_bounds(kind) -> tuple[int, int]:
    signed = pa.types.is_signed_integer(kind)
    return (-2 ** (kind.bit_width - 1) if signed else 0,
            2 ** (kind.bit_width - int(signed)) - 1)


def _every_value(values, accepted: bool):
    """One outcome for every value that is not null."""
    return pc.if_else(pc.is_valid(values), accepted, None)


def _as_float(value) -> float:
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def compare_literal(values, comparison: str, value):
    """Compare a column with a literal, never rounding the column's values.

    An integer column keeps its precision against a float literal, and
    a literal outside the column's range decides the comparison on its
    own. A float column compared with an integer the floats cannot
    hold moves the bound to the nearest float.
    """
    if pa.types.is_null(values.type):
        return values.cast(pa.bool_())
    if _numeric(value) and pa.types.is_integer(values.type):
        if isinstance(value, float):
            if math.isnan(value):
                return _every_value(values, comparison == "<>")
            if math.isfinite(value):
                if comparison in ("=", "<>") and not value.is_integer():
                    return _every_value(values, comparison == "<>")
                value = (math.ceil(value) if comparison in ("<", ">=")
                         else math.floor(value))
        lower, upper = _integer_bounds(values.type)
        if value < lower:
            return _every_value(values, comparison in (">", ">=", "<>"))
        if value > upper:
            return _every_value(values, comparison in ("<", "<=", "<>"))
        return COMPARISON_KERNELS[comparison](
            values, pa.scalar(value, type=values.type))
    if _numeric(value) and pa.types.is_floating(values.type):
        rounded = _as_float(value)
        if isinstance(value, int) and rounded != value:
            if comparison in ("=", "<>"):
                return _every_value(values, comparison == "<>")
            comparison = (("<=" if rounded < value else "<")
                          if comparison.startswith("<") else
                          (">" if rounded < value else ">="))
        return COMPARISON_KERNELS[comparison](values, pa.scalar(rounded, pa.float64()))
    return COMPARISON_KERNELS[comparison](values, pa.scalar(value))


def membership(values, literals):
    """Test membership with the literals the column's type can hold."""
    if pa.types.is_null(values.type):
        return pc.is_valid(values)
    literals = tuple(value for value in literals if value is not None)
    if literals and all(_numeric(value) for value in literals):
        if pa.types.is_integer(values.type):
            lower, upper = _integer_bounds(values.type)
            literals = [int(value) for value in literals if lower <= value <= upper
                        and (isinstance(value, int) or value.is_integer())]
            return pc.is_in(values, value_set=pa.array(literals, type=values.type))
        if pa.types.is_floating(values.type):
            literals = [_as_float(value) for value in literals
                        if isinstance(value, float) or _as_float(value) == value]
            return pc.is_in(values.cast(pa.float64()),
                            value_set=pa.array(literals, pa.float64()))
    return pc.is_in(values, value_set=pa.array(list(literals)))


def accepted_rows(mask) -> pa.Array:
    """Return a predicate's mask with every null rejected, as one array."""
    mask = pc.fill_null(mask, False)
    return mask.combine_chunks() if isinstance(mask, pa.ChunkedArray) else mask
