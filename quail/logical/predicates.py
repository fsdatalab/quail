"""Column predicates: a column tested against literals, decided without the model.

A column predicate tests a source column at the scan, or a column a
classification or an extraction computed, in a Filter. It evaluates
itself over Arrow values with one compute function per comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, ClassVar, Optional

import pyarrow as pa
from pyarrow import compute as pc

from quail.logical.expressions import ColumnRef, CompileError

VALUE_COMPARISONS = ("=", "<>", "<", "<=", ">", ">=")
TEXT_COMPARISONS = ("like", "ilike", "contains", "icontains", "starts with",
                    "ends with", "regex")
COLUMN_COMPARISONS = (
    *VALUE_COMPARISONS, "in", "is null", "is not null",
    *TEXT_COMPARISONS, *(f"not {comparison}" for comparison in TEXT_COMPARISONS))
# functions applied to the column's values before the comparison
COLUMN_FUNCTIONS = ("length",)

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
_FUNCTIONS = {"length": pc.utf8_length}


@dataclass(frozen=True)
class RegularPredicate:
    """A test of one column against literals, decided without the model.

    ``value`` is the literal compared with, a tuple of literals for
    ``in``, and None for the null tests. ``path`` names the struct
    fields read under the column, such as ``("start",)`` of an
    extraction's span. ``function`` is applied to the values first.
    """

    column: ColumnRef
    comparison: str
    value: Any = None
    path: tuple = ()  # tuple[str, ...]
    function: Optional[str] = None

    type_name: ClassVar[str] = "quail.regular_predicate"

    def aliases(self) -> tuple[str, ...]:
        return (self.column.alias,)

    @property
    def text(self) -> bool:
        """Whether the comparison tests text against a pattern."""
        return self.comparison.removeprefix("not ") in TEXT_COMPARISONS

    def validate(self) -> None:
        if not isinstance(self.column, ColumnRef):
            raise CompileError("a column predicate tests a column reference")
        if self.comparison not in COLUMN_COMPARISONS:
            raise CompileError(
                f"unsupported column comparison {self.comparison!r}; "
                f"supported: {', '.join(COLUMN_COMPARISONS)}"
            )
        if self.function is not None and self.function not in COLUMN_FUNCTIONS:
            raise CompileError(
                f"unsupported column function {self.function!r}; "
                f"supported: {', '.join(COLUMN_FUNCTIONS)}")
        if any(not isinstance(field, str) or not field for field in self.path):
            raise CompileError("a column path names struct fields")
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

    def evaluate(self, values) -> pa.ChunkedArray:
        """Return a boolean array of the rows the predicate accepts.

        A null compared with a literal is neither accepted nor rejected
        and comes back null; accepts() treats it as rejected.

        Args:
            values: The column's values, an Arrow array or chunked array.

        Raises:
            CompileError: The literal cannot be compared with the
                column's values, such as a string against a number
                column.
        """
        try:
            for field in self.path:
                values = pc.struct_field(values, field)
            if self.function is not None:
                values = _FUNCTIONS[self.function](values)
            comparison = self.comparison.removeprefix("not ")
            if comparison == "is null":
                mask = pc.is_null(values)
            elif comparison == "is not null":
                mask = pc.is_valid(values)
            elif comparison == "in":
                mask = pc.is_in(values, value_set=pa.array(
                    list(self.value), values.type))
            elif isinstance(self.value, str):
                mask = _COMPARE[comparison](values, self.value)
            else:
                mask = _COMPARE[comparison](values, pa.scalar(self.value, values.type))
            return pc.invert(mask) if comparison != self.comparison else mask
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError,
                pa.ArrowTypeError, TypeError, ValueError, OverflowError) as error:
            raise CompileError(
                f"{self} cannot be evaluated on a column of type "
                f"{values.type}: {error}") from error

    def accepts(self, values) -> pa.Array:
        """Return the rows the predicate accepts, with every null rejected."""
        return accepted_rows(self.evaluate(values))

    def to_dict(self) -> dict:
        return {"column": vars(self.column), "comparison": self.comparison,
                "value": (list(self.value) if isinstance(self.value, tuple)
                          else self.value),
                "path": list(self.path), "function": self.function}

    @classmethod
    def from_dict(cls, value: dict) -> "RegularPredicate":
        """Decode a predicate from its serialized mapping."""
        literal = value["value"]
        return cls(ColumnRef(**value["column"]), str(value["comparison"]),
                   tuple(literal) if isinstance(literal, list) else literal,
                   tuple(str(field) for field in value.get("path", ())),
                   value.get("function"))

    @property
    def name(self) -> str:
        """The tested column as written, with its path and function."""
        name = ".".join((self.column.alias, self.column.column, *self.path))
        return f"{self.function}({name})" if self.function else name

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
