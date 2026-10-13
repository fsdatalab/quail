"""Expression specs for the Python builder: columns, prompts, predicates, aggregates."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, eq=False)
class ColSpec:
    alias: Optional[str]
    column: str

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

    def asc(self) -> "SortSpec":
        return SortSpec(self)

    def desc(self) -> "SortSpec":
        return SortSpec(self, descending=True)

    def nulls_first(self) -> "SortSpec":
        return SortSpec(self, nulls="first")

    def nulls_last(self) -> "SortSpec":
        return SortSpec(self, nulls="last")

    def __hash__(self):
        return hash((self.alias, self.column))


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
class EqualsSpec:
    """An unresolved ``col(...) == col(...)`` join condition."""
    left: ColSpec
    right: ColSpec


@dataclass(frozen=True)
class PredicateSpec:
    """An unresolved regular predicate for ``where()``."""
    col: ColSpec
    comparison: str
    value: object = None


@dataclass(frozen=True)
class PromptSpec:
    template: str
    cols: tuple


@dataclass(frozen=True)
class AggSpec:
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
class HavingSpec:
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
