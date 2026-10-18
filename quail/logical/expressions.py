"""Expressions used by logical query operators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Optional

import pyarrow as pa
from pyarrow import compute as pc

DESCRIPTION_WORDS = 25
TASK_DESCRIPTION_WORDS = 50
# functions a column expression applies to a column's values
COLUMN_FUNCTIONS = ("length",)
_FUNCTIONS = {"length": pc.utf8_length}


class CompileError(ValueError):
    """Raised when a query is malformed or outside the language."""


@dataclass(frozen=True)
class ColumnRef:
    alias: str  # table alias in the query ("r")
    provider: str  # provider name in the catalog ("reviews")
    column: str  # column name ("review")

    type_name: ClassVar[str] = "quail.column_ref"


@dataclass(frozen=True)
class ColumnExpression:
    """A source column read with struct fields, a function, or an output name.

    ``path`` names the struct fields read under the column, and
    ``function`` the function applied to the values last, such as
    ``"length"``. ``name`` is the result column's name when the query
    gives one, as SQL's ``AS``. The expression stands wherever a column
    does: as a prompt's document, a join key, a column an apply()
    reads, or a result column.
    """

    reference: ColumnRef
    path: tuple = ()  # tuple[str, ...]
    function: Optional[str] = None
    name: Optional[str] = None

    type_name: ClassVar[str] = "quail.column_expression"

    @property
    def alias(self) -> str:
        return self.reference.alias

    @property
    def provider(self) -> str:
        return self.reference.provider

    def validate(self) -> None:
        if not isinstance(self.reference, ColumnRef):
            raise CompileError("a column expression reads a column reference")
        if self.function is not None and self.function not in COLUMN_FUNCTIONS:
            raise CompileError(
                f"unsupported column function {self.function!r}; "
                f"supported: {', '.join(COLUMN_FUNCTIONS)}")
        if any(not isinstance(field, str) or not field for field in self.path):
            raise CompileError("a column path names struct fields")

    def evaluate(self, values):
        """Return the expression's values over the source column's values."""
        return column_values(values, self.path, self.function,
                             column_name(self.reference))

    def __str__(self) -> str:
        written = column_expression_name(self.path, self.function,
                                         column_name(self.reference))
        return written if self.name is None else f"{written} AS {self.name}"


def column_values(values, path: tuple = (), function: Optional[str] = None,
                  name: str = ""):
    """Read struct fields under Arrow values and apply a function to them.

    Args:
        values: The source column's values, an Arrow array or chunked array.
        path: The struct fields to read, outermost first.
        function: The function applied last, one of COLUMN_FUNCTIONS.
        name: The column as written, for the error message.

    Raises:
        CompileError: The values have no such field, or the function
            does not take them.
    """
    read = name or "the column"
    for field in path:
        if not pa.types.is_struct(values.type):
            raise CompileError(f"{read} has no fields; it is {values.type}")
        if values.type.get_field_index(field) < 0:
            raise CompileError(
                f"{read} has no field {field!r}; its fields are "
                f"{[item.name for item in values.type]}")
        values = pc.struct_field(values, field)
        read = f"{read}.{field}"
    if function is not None:
        try:
            values = _FUNCTIONS[function](values)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError,
                pa.ArrowTypeError, KeyError, TypeError) as error:
            raise CompileError(
                f"{function}({read}) cannot be evaluated on {values.type}: "
                f"{error}") from error
    return values


def column_expression_name(path: tuple, function: Optional[str],
                           column: str = "") -> str:
    """Return a column with its fields and function as written."""
    name = ".".join(part for part in (column, *path) if part)
    return f"{function}({name})" if function else name


LABEL_PREFIX = " "


def column_name(column: str | ColumnRef | ColumnExpression | Alias, *,
                qualified: bool = True) -> str:
    """Return the name used for a source or computed column.

    Args:
        column: A column name, reference, expression, or projected
            expression.
        qualified: Whether a reference carries its table alias.
    """
    if isinstance(column, str):
        return column
    if isinstance(column, ColumnRef):
        return f"{column.alias}.{column.column}" if qualified else column.column
    if isinstance(column, ColumnExpression) and column.name is None:
        return column_expression_name(
            column.path, column.function,
            column_name(column.reference, qualified=qualified))
    return column.name


def column_reference(expression: ColumnRef | ColumnExpression) -> ColumnRef:
    """Return the source column an expression reads."""
    return (expression.reference if isinstance(expression, ColumnExpression)
            else expression)


def source_column(column: str | ColumnRef | ColumnExpression) -> str:
    """Return the name of the source column a document column reads."""
    return column if isinstance(column, str) else column_reference(column).column


def evaluate_column(column: str | ColumnRef | ColumnExpression, values):
    """Return a document column's values over its source column's values."""
    return (column.evaluate(values) if isinstance(column, ColumnExpression)
            else values)


def encode_column(expression: ColumnRef | ColumnExpression) -> dict:
    """Encode a column reference or expression for a physical plan."""
    return asdict(expression)


def decode_column(value: dict) -> ColumnRef | ColumnExpression:
    """Decode what encode_column wrote."""
    if "reference" in value:
        return ColumnExpression(ColumnRef(**value["reference"]),
                                tuple(value.get("path", ())), value.get("function"),
                                value.get("name"))
    return ColumnRef(**value)


@dataclass(frozen=True)
class Prompt:
    """The text of one model call, laid out for a model's prompt format.

    The planner lowers the text to token ids with the session's
    tokenizer; see quail.planner.prompt_lowering. A classification
    prompt keeps the text before each label in label_prefix and what
    rendered its label list, so lowering can render it again with one
    letter per label.

    Attributes:
        template: The template as the query names the predicate.
        args: Column references in placeholder order.
        preamble: The text before the document.
        tail: The text after the document, or after a join's last
            partner, with the document placeholder.
        frame: The text after the anchor document that every pair shares.
        frame_text: The text whose token count is the frame's; empty
            when the prompt has no frame.
        kind: filter, join, label, extract, or score.
        layout: The PROMPT_LAYOUTS name the text was built with.
        tail_segments: The tail after the placeholder, one entry per
            separately tokenized segment; empty means one segment.
        label_texts: Per placeholder of a join or a classification of
            joined rows, (alias, its label text, its frame text).
        label_prefix: The text before a scored letter or label.
        categories: A classification's labels, in query order.
        descriptions: Their descriptions, in label order.
        task_description: A classification's extra instructions.
        turn: The model's chat text around the prompt.
        source: The classification template as written.
        letters: The letters naming the labels of a lettered prompt.
    """

    template: str
    args: tuple  # tuple[ColumnRef, ...] in placeholder order
    preamble: str
    tail: str
    frame: str = ""
    frame_text: str = ""
    kind: str = "filter"
    layout: str = "ai-if"
    tail_segments: tuple = ()
    label_texts: tuple = ()
    label_prefix: str = LABEL_PREFIX
    categories: tuple = ()
    descriptions: tuple = ()
    task_description: str = ""
    turn: tuple = ("", "")
    source: str = ""
    letters: tuple = ()

    type_name: ClassVar[str] = "quail.prompt"


DEFAULT_SELECTIVITY = 0.2


SCORE_COMPARISONS = ("<", "<=", ">", ">=")


MODEL_CALL_KINDS = ("boolean", "score", "label", "extract")


def effective_selectivity(selectivity: Optional[float]) -> float:
    """Return the selectivity the planner uses: the given one or the default."""
    return DEFAULT_SELECTIVITY if selectivity is None else selectivity


PROBABILITIES_SUFFIX = "_probabilities"
# an "extract" call's second result column: the answer's character
# span in the document
SPAN_SUFFIX = "_span"


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
    # an "extract" call only: the question the answer is copied for
    question: str = ""

    type_name: ClassVar[str] = "quail.model_call"

    def aliases(self) -> tuple[str, ...]:
        """Return the table aliases the prompt reads, in placeholder order."""
        return tuple(dict.fromkeys(ref.alias for ref in self.prompt.args))

    def validate(self) -> None:
        if self.kind not in MODEL_CALL_KINDS:
            raise CompileError(
                f"model call kind must be one of {MODEL_CALL_KINDS}, got {self.kind!r}"
            )
        if self.kind != "label" and (
            self.labels or self.descriptions or self.probabilities
        ):
            raise CompileError("only an AI.CLASSIFY call has labels")
        if self.kind != "extract" and self.question:
            raise CompileError("only an AI.EXTRACT call has a question")
        if self.kind == "label":
            validate_labels(self.labels, self.descriptions)
        elif self.kind == "extract":
            validate_question(self.question)


def validate_question(question: str) -> None:
    """Check an AI.EXTRACT question: text with at least one word.

    Raises:
        CompileError: The question is empty or not a string.
    """
    if not isinstance(question, str) or not question.strip():
        raise CompileError("AI.EXTRACT needs a question with at least one word")


@dataclass(frozen=True)
class ExtractStruct:
    """Several AI.EXTRACT fields of one table, returned as struct columns.

    Each field is its own extraction, projected as an Alias named
    ``name.field``. The result packs the fields' answers into the
    struct column ``name`` and their spans into ``name`` plus
    SPAN_SUFFIX, each with one struct field per extraction field.

    Attributes:
        name: The struct column's name.
        fields: The field names, in written order.
    """

    name: str
    fields: tuple[str, ...]

    type_name: ClassVar[str] = "quail.extract_struct"

    def member(self, field: str) -> str:
        """Return the name of one field's extraction."""
        return f"{self.name}.{field}"

    @property
    def members(self) -> tuple[str, ...]:
        """Return the names of the fields' extractions, in field order."""
        return tuple(self.member(field) for field in self.fields)

    def packing(self) -> tuple:
        """Return the answer and span struct columns as Project packs them.

        Returns:
            (struct column, ((field, member column), ...)) for the
            answers, then for the spans.
        """
        return tuple(
            (self.name + suffix,
             tuple((field, self.member(field) + suffix) for field in self.fields))
            for suffix in ("", SPAN_SUFFIX))

    def validate(self) -> None:
        if not self.name or "." in self.name:
            raise CompileError(
                f"an extraction needs a column name without a dot, "
                f"got {self.name!r}")
        if not self.fields:
            raise CompileError(f"AI.EXTRACT {self.name!r} needs at least one field")
        for field in self.fields:
            if not isinstance(field, str) or not field or "." in field:
                raise CompileError(
                    f"an AI.EXTRACT field needs a name without a dot, "
                    f"got {field!r}")
        if len(set(self.fields)) != len(self.fields):
            raise CompileError(
                f"AI.EXTRACT {self.name!r} names a field twice: {list(self.fields)}")


def packed_columns(columns, packing) -> tuple[str, ...]:
    """Return result column names after packing members into struct columns.

    Each struct column takes the place of its first member; its other
    members leave the result.

    Args:
        columns: The column names before packing.
        packing: (struct column, ((field, member column), ...)) pairs.
    """
    owner = {member: name for name, fields in packing for _, member in fields}
    out = []
    for column in columns:
        name = owner.get(column, column)
        if name not in out:
            out.append(name)
    return tuple(out)


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
        if expression.kind == "extract":
            raise CompileError(
                "an AI.EXTRACT answer is a result column, not a predicate"
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


@dataclass(frozen=True)
class Equality:
    """One ordinary join condition: two columns of two tables are equal."""

    left: ColumnRef | ColumnExpression
    right: ColumnRef | ColumnExpression

    type_name: ClassVar[str] = "quail.equality"

    def aliases(self) -> tuple[str, str]:
        return self.left.alias, self.right.alias

    def __str__(self) -> str:
        return f"{column_name(self.left)} = {column_name(self.right)}"


def _explain(expression) -> str:
    if isinstance(expression, Compare):
        return (
            f"{_explain(expression.call)} {expression.comparison} "
            f"{expression.threshold}"
        )
    if isinstance(expression, Alias):
        if expression.expression.kind in ("label", "extract"):
            # the SemanticClassify or SemanticExtract below computes
            # the column
            return expression.name
        return f"{_explain(expression.expression)} AS {expression.name}"
    if expression.kind == "label":
        return f"AI.CLASSIFY({expression.prompt.template!r}, {list(expression.labels)})"
    if expression.kind == "extract":
        return f"AI.EXTRACT({expression.prompt.template!r}, {expression.question!r})"
    function = "AI.SCORE" if expression.kind == "score" else "AI.IF"
    return f"{function}({expression.prompt.template!r})"


@dataclass(frozen=True)
class SortKey:
    """One ORDER BY term: a column or a projected expression and its direction."""

    expression: Any  # ColumnRef | ColumnExpression | Alias | AggregateCall
    descending: bool = False
    nulls_first: bool = False

    type_name: ClassVar[str] = "quail.sort_key"

    @property
    def name(self) -> str:
        """Return the result column name the key sorts on."""
        return column_name(self.expression)

    def validate(self) -> None:
        if isinstance(self.expression, (Alias, AggregateCall, ColumnExpression)):
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

    def __post_init__(self) -> None:
        value = self.value
        object.__setattr__(self, "value", value.item() if hasattr(value, "item")
                           and not isinstance(value, (int, float)) else value)

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
