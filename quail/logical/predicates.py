"""Column predicates: a column tested against literals, decided without the model.

A column predicate tests a source column at the scan, or a column a
classification or an extraction computed, in a Filter. The planner
and the executor evaluate every comparison with one Arrow compute
function (quail.catalog.selection).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from quail.logical.expressions import ColumnRef, CompileError

VALUE_COMPARISONS = ("=", "<>", "<", "<=", ">", ">=")
TEXT_COMPARISONS = ("like", "ilike", "contains", "icontains", "starts with",
                    "ends with", "regex")
COLUMN_COMPARISONS = (
    *VALUE_COMPARISONS, "in", "is null", "is not null",
    *TEXT_COMPARISONS, *(f"not {comparison}" for comparison in TEXT_COMPARISONS))
# functions applied to the column's values before the comparison
COLUMN_FUNCTIONS = ("length",)


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
