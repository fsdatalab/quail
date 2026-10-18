"""Column predicates: a column tested against literals, decided without the model.

A column predicate tests a source column at the scan, or a column a
classification or an extraction computed, in a Filter. It evaluates
itself over Arrow values with one compute function per comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Any, ClassVar, Optional

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

from quail.logical.expressions import (
    ColumnRef,
    CompileError,
    ScalarFunction,
    column_name,
    column_references,
    decode_expression,
    encode_expression,
    evaluate_expression,
)

VALUE_COMPARISONS = ("=", "<>", "<", "<=", ">", ">=")
TEXT_COMPARISONS = ("like", "ilike", "contains", "icontains", "starts with",
                    "ends with", "regex")
COLUMN_COMPARISONS = (
    *VALUE_COMPARISONS, "in", "is null", "is not null",
    *TEXT_COMPARISONS, *(f"not {comparison}" for comparison in TEXT_COMPARISONS))

_COMPARE = {
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


def literal_value(value):
    """Return a literal as a plain Python value: a NumPy scalar becomes one."""
    if isinstance(value, tuple):
        return tuple(literal_value(item) for item in value)
    return value.item() if isinstance(value, np.generic) else value


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
        return _COMPARE[comparison](values, pa.scalar(value, type=values.type))
    if _numeric(value) and pa.types.is_floating(values.type):
        rounded = _as_float(value)
        if isinstance(value, int) and rounded != value:
            if comparison in ("=", "<>"):
                return _every_value(values, comparison == "<>")
            comparison = (("<=" if rounded < value else "<")
                          if comparison.startswith("<") else
                          (">" if rounded < value else ">="))
        return _COMPARE[comparison](values, pa.scalar(rounded, pa.float64()))
    return _COMPARE[comparison](values, pa.scalar(value))


def _membership(values, literals):
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


@dataclass(frozen=True)
class RegularPredicate:
    """A test of a value expression against literals, decided without the model.

    ``expression`` reads one source column, with fields or a function
    on it, such as the ``start`` field of an extraction's span. ``value``
    is the literal compared with, a tuple of literals for ``in``, and
    None for the null tests.
    """

    expression: Any
    comparison: str
    value: Any = None

    type_name: ClassVar[str] = "quail.regular_predicate"

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", literal_value(self.value))

    @property
    def column(self) -> ColumnRef:
        """The one source column the expression reads."""
        (column,) = column_references(self.expression)
        return column

    @property
    def function(self) -> Optional[str]:
        """The function applied last, or None."""
        return (self.expression.function
                if isinstance(self.expression, ScalarFunction) else None)

    def aliases(self) -> tuple[str, ...]:
        return (self.column.alias,)

    @property
    def text(self) -> bool:
        """Whether the comparison tests text against a pattern."""
        return self.comparison.removeprefix("not ") in TEXT_COMPARISONS

    def validate(self) -> None:
        if len(column_references(self.expression)) != 1:
            raise CompileError("a column predicate tests one column")
        self.expression.validate()
        if self.comparison not in COLUMN_COMPARISONS:
            raise CompileError(
                f"unsupported column comparison {self.comparison!r}; "
                f"supported: {', '.join(COLUMN_COMPARISONS)}"
            )
        if self.comparison == "in":
            if not isinstance(self.value, tuple) or not self.value:
                raise CompileError("IN needs a non-empty list of literals")
            if len(set(self.value)) != len(self.value):
                raise CompileError(f"{self.name} IN lists a value twice")
        elif self.comparison.startswith("is"):
            if self.value is not None:
                raise CompileError("a null test takes no value")
        elif self.text:
            if not isinstance(self.value, str):
                raise CompileError(f"{self.comparison} compares with a string")
        elif self.value is None or isinstance(self.value, (tuple, list)):
            raise CompileError(f"{self.comparison} compares with one literal")

    def validate_text_column(self) -> None:
        """Check the literals of a test over a text column, such as an answer.

        Raises:
            CompileError: The text is compared with a number or boolean,
                or its length with a string.
        """
        self.validate()
        if self.comparison.startswith("is"):
            return
        values = self.value if isinstance(self.value, tuple) else (self.value,)
        if self.function is None and any(not isinstance(v, str) for v in values):
            raise CompileError(
                f"{self.name} is text; {self.comparison} compares it with a string")
        if self.function == "length" and any(
                isinstance(v, (str, bool)) for v in values):
            raise CompileError(f"{self.name} compares with a number")

    def evaluate(self, columns) -> pa.ChunkedArray:
        """Return a boolean array of the rows the predicate accepts.

        A null compared with a literal is neither accepted nor rejected
        and comes back null; accepts() treats it as rejected.

        Args:
            columns: The table, record batch, or rows holding the
                tested column, read with ``column(name)``.

        Raises:
            CompileError: The literal cannot be compared with the
                column's values, such as a string against a number
                column.
        """
        values = evaluate_expression(self.expression, columns)
        try:
            comparison = self.comparison.removeprefix("not ")
            if comparison == "is null":
                mask = pc.is_null(values)
            elif comparison == "is not null":
                mask = pc.is_valid(values)
            elif comparison == "in":
                mask = _membership(values, self.value)
            elif isinstance(self.value, str):
                mask = _COMPARE[comparison](values, self.value)
            else:
                mask = compare_literal(values, comparison, self.value)
            return pc.invert(mask) if comparison != self.comparison else mask
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError,
                pa.ArrowTypeError, TypeError, ValueError, OverflowError) as error:
            raise CompileError(
                f"{self} cannot be evaluated on a column of type "
                f"{values.type}: {error}") from error

    def accepts(self, columns) -> pa.Array:
        """Return the rows the predicate accepts, with every null rejected."""
        return accepted_rows(self.evaluate(columns))

    def to_dict(self) -> dict:
        return {"expression": encode_expression(self.expression),
                "comparison": self.comparison,
                "value": (list(self.value) if isinstance(self.value, tuple)
                          else self.value)}

    @classmethod
    def from_dict(cls, value: dict) -> "RegularPredicate":
        """Decode a predicate from its serialized mapping."""
        literal = value["value"]
        return cls(decode_expression(value["expression"]), str(value["comparison"]),
                   tuple(literal) if isinstance(literal, list) else literal)

    @property
    def name(self) -> str:
        """The tested expression as written."""
        return column_name(self.expression)

    def __str__(self) -> str:
        if self.comparison == "in":
            return f"{self.name} IN ({', '.join(repr(v) for v in self.value)})"
        if self.comparison.startswith("is"):
            return f"{self.name} {self.comparison.upper()}"
        return f"{self.name} {self.comparison} {self.value!r}"


def accepted_rows(mask) -> pa.Array:
    """Return a predicate's mask with every null rejected, as one array."""
    mask = pc.fill_null(mask, False)
    return mask.combine_chunks() if isinstance(mask, pa.ChunkedArray) else mask
