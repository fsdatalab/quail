"""Expression specs for the Python builder: columns, prompts, predicates, aggregates."""

from dataclasses import dataclass
from typing import Optional

from quail.logical import CompileError


class ExpressionSpec:
    """A query expression evaluated over rows, without a Python truth value."""

    def __bool__(self):
        raise TypeError(
            "query expressions have no Python truth value; combine where() "
            "conditions with & or pass them as separate arguments")


@dataclass(frozen=True, eq=False)
class ColSpec(ExpressionSpec):
    """A column reference, with the struct fields and function read on it.

    Attributes:
        alias: The table alias or result struct, or None when unqualified.
        column: The column or struct field name.
        path: Struct fields read under the column, such as ``("start",)``
            of an extraction's span.
        function: A function applied to the values, such as ``"length"``.
    """

    alias: Optional[str]
    column: str
    path: tuple = ()
    function: Optional[str] = None

    def require_plain(self) -> None:
        """Reject transformations where only a column reference is supported.

        Raises:
            CompileError: The expression reads a field or applies a function.
        """
        if self.path or self.function is not None:
            raise CompileError(
                "this position needs a plain source column, without "
                "functions or fields")

    def __eq__(self, other):
        """``col("c.url") == col("e.url")`` is a join condition.

        Compared with a literal, it is a regular predicate for ``where()``.
        """
        if isinstance(other, ColSpec):
            return EqualsSpec(self, other)
        return PredicateSpec(self, "=", other)

    def __ne__(self, other):
        return PredicateSpec(self, "<>", other)

    def __lt__(self, other):
        return PredicateSpec(self, "<", other)

    def __le__(self, other):
        return PredicateSpec(self, "<=", other)

    def __gt__(self, other):
        return PredicateSpec(self, ">", other)

    def __ge__(self, other):
        return PredicateSpec(self, ">=", other)

    def isin(self, values) -> "PredicateSpec":
        return PredicateSpec(self, "in", tuple(values))

    def is_null(self) -> "PredicateSpec":
        return PredicateSpec(self, "is null")

    def is_not_null(self) -> "PredicateSpec":
        return PredicateSpec(self, "is not null")

    def like(self, pattern: str) -> "PredicateSpec":
        """SQL LIKE: ``%`` matches any text and ``_`` one character."""
        return PredicateSpec(self, "like", pattern)

    def ilike(self, pattern: str) -> "PredicateSpec":
        """LIKE ignoring case."""
        return PredicateSpec(self, "ilike", pattern)

    def contains(self, text: str, ignore_case: bool = False) -> "PredicateSpec":
        return PredicateSpec(self, "icontains" if ignore_case else "contains", text)

    def startswith(self, text: str) -> "PredicateSpec":
        return PredicateSpec(self, "starts with", text)

    def endswith(self, text: str) -> "PredicateSpec":
        return PredicateSpec(self, "ends with", text)

    def rlike(self, pattern: str) -> "PredicateSpec":
        """Whether a regular expression (RE2 syntax) matches anywhere."""
        return PredicateSpec(self, "regex", pattern)

    def field(self, name: str) -> "ColSpec":
        """Read a field of a struct column, such as a span's ``start``."""
        self._require_no_function()
        return ColSpec(self.alias, self.column, (*self.path, name))

    def length(self) -> "ColSpec":
        """The number of characters of each value."""
        self._require_no_function()
        return ColSpec(self.alias, self.column, self.path, "length")

    def _require_no_function(self) -> None:
        if self.function is not None:
            raise CompileError(
                "a column function cannot be followed by another function "
                "or a field access")

    def asc(self) -> "SortSpec":
        return SortSpec(self)

    def desc(self) -> "SortSpec":
        return SortSpec(self, descending=True)

    def nulls_first(self) -> "SortSpec":
        return SortSpec(self, nulls="first")

    def nulls_last(self) -> "SortSpec":
        return SortSpec(self, nulls="last")

    def __hash__(self):
        return hash((self.alias, self.column, self.path, self.function))


@dataclass(frozen=True)
class SortSpec:
    """A sort key for ``order_by()``: ``col("r.stars").desc().nulls_last()``.

    Attributes:
        col: The column or projected name to sort by.
        descending: Whether to sort from the largest value down.
        nulls: ``"first"`` or ``"last"``, or None for the default: nulls
            last when ascending and first when descending.
    """

    col: ColSpec
    descending: bool = False
    nulls: Optional[str] = None

    def asc(self) -> "SortSpec":
        return SortSpec(self.col, False, self.nulls)

    def desc(self) -> "SortSpec":
        return SortSpec(self.col, True, self.nulls)

    def nulls_first(self) -> "SortSpec":
        return SortSpec(self.col, self.descending, "first")

    def nulls_last(self) -> "SortSpec":
        return SortSpec(self.col, self.descending, "last")


@dataclass(frozen=True)
class EqualsSpec(ExpressionSpec):
    """An unresolved ``col(...) == col(...)`` join condition."""
    left: ColSpec
    right: ColSpec


@dataclass(frozen=True)
class PredicateSpec(ExpressionSpec):
    """An unresolved regular predicate for ``where()``.

    Attributes:
        col: The tested column.
        comparison: One of quail.logical.COLUMN_COMPARISONS.
        value: The literal or literals compared with.
        selectivity: Estimated fraction of documents the test keeps,
            for a test on a label or an answer; None uses 0.2.
    """
    col: ColSpec
    comparison: str
    value: object = None
    selectivity: Optional[float] = None

    def __invert__(self) -> "PredicateSpec":
        """``~col(...).like(...)``: the negated text test."""
        if self.comparison.startswith("not "):
            comparison = self.comparison[4:]
        elif self.comparison in ("is null", "is not null"):
            comparison = "is not null" if self.comparison == "is null" else "is null"
        else:
            comparison = f"not {self.comparison}"
        return PredicateSpec(self.col, comparison, self.value, self.selectivity)

    def hint(self, *, selectivity: float) -> "PredicateSpec":
        """``col(...).like(...).hint(selectivity=0.4)``: the fraction kept."""
        return PredicateSpec(self.col, self.comparison, self.value, selectivity)

    def __and__(self, other) -> "AndSpec":
        return AndSpec((self,)) & other

    def tests(self) -> tuple:
        """The tests combined with AND: this one."""
        return (self,)


@dataclass(frozen=True)
class AndSpec(ExpressionSpec):
    """Tests combined with AND: ``(col("a") > 1) & col("b").is_null()``."""
    parts: tuple

    def __and__(self, other) -> "AndSpec":
        if not isinstance(other, (PredicateSpec, AndSpec)):
            raise TypeError(f"& combines tests of columns, got {other!r}")
        return AndSpec((*self.parts, *other.tests()))

    def tests(self) -> tuple:
        """The tests combined with AND, in written order."""
        return self.parts


@dataclass(frozen=True)
class PromptSpec:
    template: str
    cols: tuple


@dataclass(frozen=True)
class AggSpec(ExpressionSpec):
    """An unresolved aggregate for ``agg()``."""
    function: str
    col: object = None    # ColSpec, a label or score name, or None

    def _test(self, comparison, value):
        return HavingSpec(self, comparison, value)

    def __eq__(self, other):
        return self._test("=", other)

    def __ne__(self, other):
        return self._test("<>", other)

    def __lt__(self, other):
        return self._test("<", other)

    def __le__(self, other):
        return self._test("<=", other)

    def __gt__(self, other):
        return self._test(">", other)

    def __ge__(self, other):
        return self._test(">=", other)

    def __hash__(self):
        return hash((self.function, self.col))


@dataclass(frozen=True)
class HavingSpec(ExpressionSpec):
    """An unresolved ``having()`` test of an aggregate against a number."""
    agg: AggSpec
    comparison: str
    value: object


def count(c=None) -> AggSpec:
    """``count()`` counts rows; ``count("r.id")`` counts non-null values."""
    return AggSpec("count", c)


def count_distinct(c) -> AggSpec:
    return AggSpec("count_distinct", c)


def sum_(c) -> AggSpec:
    return AggSpec("sum", c)


def avg(c) -> AggSpec:
    return AggSpec("avg", c)


def min_(c) -> AggSpec:
    return AggSpec("min", c)


def max_(c) -> AggSpec:
    return AggSpec("max", c)


def col(ref: str) -> ColSpec:
    """"r.review" or "review" (unqualified resolves when unambiguous)."""
    if "." in ref:
        alias, column = ref.split(".", 1)
        return ColSpec(alias=alias, column=column)
    return ColSpec(alias=None, column=ref)


def prompt(template: str, *cols: ColSpec) -> PromptSpec:
    return PromptSpec(template=template, cols=tuple(cols))
