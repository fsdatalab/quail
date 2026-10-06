"""Expressions used by logical query operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Optional

DESCRIPTION_WORDS = 25
TASK_DESCRIPTION_WORDS = 50


class CompileError(ValueError):
    """Raised when a query is malformed or outside the language."""


@dataclass(frozen=True)
class ColumnRef:
    alias: str  # table alias in the query ("r")
    provider: str  # provider name in the catalog ("reviews")
    column: str  # column name ("review")

    type_name: ClassVar[str] = "quail.column_ref"


LABEL_PREFIX = " "


def column_name(column: ColumnRef | Alias) -> str:
    """Return the name used for a source or computed column."""
    return (
        f"{column.alias}.{column.column}"
        if isinstance(column, ColumnRef)
        else column.name
    )


@dataclass(frozen=True)
class Prompt:
    """Bound prompt text and token layout for one LLM call.

    Token counts are filled when a tokenizer is available; None means the
    planner must calculate them. A classification prompt keeps the text
    before each label in label_prefix. Its optional lettered prompt names
    each label with a distinct one-token letter for the letters rule.
    """

    template: str
    args: tuple  # tuple[ColumnRef, ...] in placeholder order
    preamble: str
    tail: str
    preamble_tokens: Optional[int] = None
    tail_tokens: Optional[int] = None
    frame: str = ""
    frame_tokens: Optional[int] = None
    # join prompts only: (alias, label_tokens, anchor_frame_tokens)
    # per placeholder, in order. Empty for filter prompts.
    labels: tuple = ()
    preamble_token_ids: tuple = ()
    tail_token_ids: tuple = ()
    label_token_ids: tuple = ()
    # the PROMPT_LAYOUTS name the prompt text was built with
    layout: str = "ai-if"
    # the text after a filter document or a join's last partner, one
    # entry per separately tokenized segment; empty means one segment
    tail_segments: tuple = ()
    label_prefix: str = LABEL_PREFIX
    letters: tuple = ()
    lettered: Optional["Prompt"] = None

    type_name: ClassVar[str] = "quail.prompt"


DEFAULT_SELECTIVITY = 0.2


SCORE_COMPARISONS = ("<", "<=", ">", ">=")


MODEL_CALL_KINDS = ("boolean", "score", "label")


def effective_selectivity(selectivity: Optional[float]) -> float:
    """Return the selectivity the planner uses: the given one or the default."""
    return DEFAULT_SELECTIVITY if selectivity is None else selectivity


PROBABILITIES_SUFFIX = "_probabilities"


@dataclass(frozen=True)
class ModelCall:
    """One prompt asked of every row; the model answers with a value.

    The AI functions of the language are all model calls. They differ
    only in ``kind``, the type of the answer.
    """

    prompt: Prompt
    kind: str = "boolean"
    # a "label" call only: the labels in written order, and one
    # description per label, empty for none
    labels: tuple = ()
    descriptions: tuple = ()
    # a "label" call only: whether the result also carries each label's
    # probability, as the column named after the label column plus
    # PROBABILITIES_SUFFIX
    probabilities: bool = False

    type_name: ClassVar[str] = "quail.model_call"

    def aliases(self) -> tuple[str, ...]:
        """Return the table aliases the prompt reads, in placeholder order."""
        return tuple(dict.fromkeys(ref.alias for ref in self.prompt.args))

    def validate(self) -> None:
        if self.kind not in MODEL_CALL_KINDS:
            raise CompileError(
                f"model call kind must be one of {MODEL_CALL_KINDS}, got {self.kind!r}"
            )
        if self.kind != "label":
            if self.labels or self.descriptions or self.probabilities:
                raise CompileError("only an AI.CLASSIFY call has labels")
            return
        validate_labels(self.labels, self.descriptions)


def validate_labels(labels: tuple, descriptions: tuple = ()) -> None:
    """Check an AI.CLASSIFY label list and its descriptions.

    Raises:
        CompileError: Fewer than two labels, an empty label, two labels
            equal ignoring case, or a description count that differs
            from the label count.
    """
    if len(labels) < 2:
        raise CompileError("AI.CLASSIFY needs at least two labels")
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise CompileError("every AI.CLASSIFY label is non-empty text")
    folded = [label.casefold() for label in labels]
    if len(set(folded)) != len(folded):
        raise CompileError(
            f"AI.CLASSIFY labels must differ ignoring case, got {list(labels)}"
        )
    if descriptions and len(descriptions) != len(labels):
        raise CompileError(
            "AI.CLASSIFY needs one description per label, empty for none"
        )
    for description in descriptions:
        if not isinstance(description, str):
            raise CompileError("an AI.CLASSIFY description is text or empty")
        if len(description.split()) > DESCRIPTION_WORDS:
            raise CompileError(
                f"an AI.CLASSIFY description has at most {DESCRIPTION_WORDS} "
                f"words, got {len(description.split())}"
            )


def validate_task_description(text: str) -> None:
    """Check an AI.CLASSIFY task description's length.

    Raises:
        CompileError: More than TASK_DESCRIPTION_WORDS words.
    """
    if len(text.split()) > TASK_DESCRIPTION_WORDS:
        raise CompileError(
            f"an AI.CLASSIFY task description has at most "
            f"{TASK_DESCRIPTION_WORDS} words, got {len(text.split())}"
        )


@dataclass(frozen=True)
class Compare:
    """A score compared with a threshold; answers yes or no."""

    call: ModelCall
    comparison: str
    threshold: float

    type_name: ClassVar[str] = "quail.compare"

    def validate(self) -> None:
        self.call.validate()
        if self.call.kind != "score":
            raise CompileError("only an AI.SCORE call compares with a threshold")
        if self.comparison not in SCORE_COMPARISONS:
            raise CompileError(f"unsupported AI.SCORE comparison {self.comparison!r}")
        if not 0.0 <= self.threshold <= 1.0:
            raise CompileError("AI.SCORE threshold must be between 0 and 1")


@dataclass(frozen=True)
class InList:
    """A column's value is one of the listed values, as SQL IN."""

    column: ColumnRef
    values: tuple  # tuple[str, ...]

    type_name: ClassVar[str] = "quail.in_list"

    def validate(self) -> None:
        # only a repeated value is an error; a value the column never
        # holds matches no row
        if len(set(self.values)) != len(self.values):
            raise CompileError(
                f"{self.column.alias}.{self.column.column} IN lists a value twice"
            )


@dataclass(frozen=True)
class Alias:
    """A model call returned as a named result column."""

    expression: ModelCall
    name: str

    type_name: ClassVar[str] = "quail.alias"

    def validate(self) -> None:
        self.expression.validate()
        if not self.name:
            raise CompileError("a projected expression needs a name")


def model_call(expression) -> ModelCall:
    """Return the model call inside a predicate or projected expression."""
    if isinstance(expression, ModelCall):
        return expression
    if isinstance(expression, Compare):
        return expression.call
    if isinstance(expression, Alias):
        return expression.expression
    raise TypeError(f"{type(expression).__name__} holds no model call")


def is_score(expression) -> bool:
    """Return whether the expression computes an AI.SCORE value."""
    return model_call(expression).kind == "score"


def is_label(expression) -> bool:
    """Return whether the expression computes an AI.CLASSIFY label."""
    return model_call(expression).kind == "label"


def has_score(plan) -> bool:
    """Return whether a logical plan computes an AI.SCORE value anywhere."""
    operators = plan.operators()
    return (
        any(
            is_score(predicate.expression)
            for predicates in operators.filters.values()
            for predicate in predicates
        )
        or any(is_score(join.predicate) for join in operators.joins)
        or any(
            isinstance(expression, Alias) and expression.expression.kind == "score"
            for expression in operators.projections
        )
    )


def validate_predicate(expression) -> None:
    """Check that an expression answers yes or no for every row."""
    if isinstance(expression, Compare):
        expression.validate()
        return
    if isinstance(expression, ModelCall):
        expression.validate()
        if expression.kind == "score":
            raise CompileError(
                "an AI.SCORE call is a predicate only when compared with a threshold"
            )
        if expression.kind == "label":
            raise CompileError(
                "an AI.CLASSIFY label is tested by a Filter on its "
                "column, not by a SemanticFilter"
            )
        return
    raise CompileError(
        f"a predicate is a model call or a comparison, got {type(expression).__name__}"
    )


@dataclass(frozen=True)
class FilterPredicate:
    """One predicate over a table's documents, with its selectivity hint."""

    expression: Any  # ModelCall | Compare
    selectivity: Optional[float] = None  # fraction of documents that
    #                                       pass; None means not given

    type_name: ClassVar[str] = "quail.filter_predicate"

    @property
    def prompt(self) -> Prompt:
        """The prompt the predicate asks."""
        return model_call(self.expression).prompt


COLUMN_COMPARISONS = ("=", "<>", "<", "<=", ">", ">=", "in", "is null", "is not null")


@dataclass(frozen=True)
class RegularPredicate:
    """A test of one source column against literals, decided without the model.

    ``value`` is the literal compared with, a tuple of literals for
    ``in``, and None for the null tests.
    """

    column: ColumnRef
    comparison: str
    value: Any = None

    type_name: ClassVar[str] = "quail.regular_predicate"

    def aliases(self) -> tuple[str, ...]:
        return (self.column.alias,)

    def validate(self) -> None:
        if not isinstance(self.column, ColumnRef):
            raise CompileError("a column predicate tests a column reference")
        if self.comparison not in COLUMN_COMPARISONS:
            raise CompileError(
                f"unsupported column comparison {self.comparison!r}; "
                f"supported: {', '.join(COLUMN_COMPARISONS)}"
            )
        if self.comparison == "in":
            if not isinstance(self.value, tuple) or not self.value:
                raise CompileError("IN needs a non-empty list of literals")
        elif self.comparison.startswith("is"):
            if self.value is not None:
                raise CompileError("a null test takes no value")
        elif self.value is None or isinstance(self.value, (tuple, list)):
            raise CompileError(f"{self.comparison} compares with one literal")

    def __str__(self) -> str:
        name = f"{self.column.alias}.{self.column.column}"
        if self.comparison == "in":
            return f"{name} IN ({', '.join(repr(v) for v in self.value)})"
        if self.comparison.startswith("is"):
            return f"{name} {self.comparison.upper()}"
        return f"{name} {self.comparison} {self.value!r}"


@dataclass(frozen=True)
class Equality:
    """One ordinary join condition: two columns of two tables are equal."""

    left: ColumnRef
    right: ColumnRef

    type_name: ClassVar[str] = "quail.equality"

    def aliases(self) -> tuple[str, str]:
        return self.left.alias, self.right.alias

    def __str__(self) -> str:
        return (
            f"{self.left.alias}.{self.left.column} = "
            f"{self.right.alias}.{self.right.column}"
        )


def _explain(expression) -> str:
    if isinstance(expression, Compare):
        return (
            f"{_explain(expression.call)} {expression.comparison} "
            f"{expression.threshold}"
        )
    if isinstance(expression, InList):
        return (
            f"{expression.column.alias}.{expression.column.column} "
            f"IN {list(expression.values)}"
        )
    if isinstance(expression, Alias):
        if expression.expression.kind == "label":
            # the SemanticClassify below computes the column
            return expression.name
        return f"{_explain(expression.expression)} AS {expression.name}"
    if expression.kind == "label":
        return f"AI.CLASSIFY({expression.prompt.template!r}, {list(expression.labels)})"
    function = "AI.SCORE" if expression.kind == "score" else "AI.IF"
    return f"{function}({expression.prompt.template!r})"


@dataclass(frozen=True)
class SortKey:
    """One ORDER BY term: a column or a projected expression and its direction."""

    expression: Any  # ColumnRef | Alias
    descending: bool = False
    nulls_first: bool = False

    type_name: ClassVar[str] = "quail.sort_key"

    @property
    def name(self) -> str:
        """Return the result column name the key sorts on."""
        if isinstance(self.expression, ColumnRef):
            return f"{self.expression.alias}.{self.expression.column}"
        return self.expression.name

    def validate(self) -> None:
        if isinstance(self.expression, (Alias, AggregateCall)):
            self.expression.validate()
        elif not isinstance(self.expression, ColumnRef):
            raise CompileError(
                f"a sort key is a column reference, a projected "
                f"expression, or an aggregate, got "
                f"{type(self.expression).__name__}"
            )

    def __str__(self) -> str:
        return (
            f"{self.name} {'DESC' if self.descending else 'ASC'} "
            f"NULLS {'FIRST' if self.nulls_first else 'LAST'}"
        )


AGGREGATE_FUNCTIONS = ("count", "count_distinct", "sum", "avg", "min", "max")


HAVING_COMPARISONS = ("=", "<>", "<", "<=", ">", ">=")


@dataclass(frozen=True)
class AggregateCall:
    """One aggregate over the rows of a group, returned under a name.

    ``argument`` names a projected column, or is None for ``count(*)``.
    """

    function: str
    argument: Optional[str]
    name: str

    type_name: ClassVar[str] = "quail.aggregate_call"

    def validate(self) -> None:
        if self.function not in AGGREGATE_FUNCTIONS:
            raise CompileError(
                f"unsupported aggregate {self.function!r}; supported: "
                f"{', '.join(AGGREGATE_FUNCTIONS)}"
            )
        if self.argument is None and self.function != "count":
            raise CompileError(f"{self.function}(*) is not an aggregate")
        if not self.name:
            raise CompileError("an aggregate needs a name")

    def __str__(self) -> str:
        function = (
            "count(distinct "
            if self.function == "count_distinct"
            else f"{self.function}("
        )
        return f"{self.name} = {function}{self.argument or '*'})"


@dataclass(frozen=True)
class HavingTest:
    """A comparison of one aggregate with a literal, keeping the group."""

    aggregate: AggregateCall
    comparison: str
    value: Any

    type_name: ClassVar[str] = "quail.having_test"

    def validate(self) -> None:
        self.aggregate.validate()
        if self.comparison not in HAVING_COMPARISONS:
            raise CompileError(
                f"HAVING compares with one of {HAVING_COMPARISONS}, got "
                f"{self.comparison!r}"
            )
        if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
            raise CompileError("HAVING compares an aggregate with a number")

    def __str__(self) -> str:
        return f"{self.aggregate.name} {self.comparison} {self.value!r}"


@dataclass(frozen=True)
class Aggregation:
    """GROUP BY keys, the aggregates over each group, and the HAVING tests.

    ``keys`` and each aggregate's argument name projected columns.
    ``output`` lists the result columns in SELECT order: keys and
    aggregate names. An aggregate only a HAVING test reads is absent
    from ``output``.
    """

    keys: tuple[str, ...]
    aggregates: tuple[AggregateCall, ...]
    output: tuple[str, ...]
    having: tuple[HavingTest, ...] = ()

    type_name: ClassVar[str] = "quail.aggregation"

    def validate(self, projected: tuple) -> None:
        """Check the keys, arguments, output, and tests against the projection.

        Args:
            projected: The names of the projected columns.
        """
        for key in self.keys:
            if key not in projected:
                raise CompileError(f"GROUP BY {key!r} is not a projected column")
        names = list(self.keys)
        for aggregate in self.aggregates:
            aggregate.validate()
            if aggregate.argument is not None and aggregate.argument not in projected:
                raise CompileError(
                    f"{aggregate} reads a column the projection does not have"
                )
            names.append(aggregate.name)
        if len(names) != len(set(names)):
            raise CompileError(
                f"GROUP BY keys and aggregate names must be unique, got {names}"
            )
        if not self.output:
            raise CompileError("an aggregation returns at least one column")
        for name in self.output:
            if name not in names:
                raise CompileError(
                    f"{name!r} is neither a GROUP BY key nor an aggregate"
                )
        if len(self.output) != len(set(self.output)):
            raise CompileError(f"output names must be unique, got {self.output}")
        for test in self.having:
            if not isinstance(test, HavingTest):
                raise CompileError(
                    f"a HAVING term is a HavingTest, got {type(test).__name__}"
                )
            test.validate()
            if test.aggregate not in self.aggregates:
                raise CompileError(
                    f"HAVING {test} tests an aggregate the aggregation does not compute"
                )

    def __str__(self) -> str:
        parts = []
        if self.keys:
            parts.append("group by " + ", ".join(self.keys))
        parts.append(", ".join(str(a) for a in self.aggregates))
        if self.having:
            parts.append("having " + " and ".join(str(t) for t in self.having))
        return "; ".join(parts)
