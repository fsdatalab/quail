"""Logical plan nodes and the plan builder."""

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Optional, Protocol


class CompileError(ValueError):
    """Raised when a query is malformed or outside the language."""


@dataclass(frozen=True)
class ColumnRef:
    alias: str       # table alias in the query ("r")
    provider: str    # provider name in the catalog ("reviews")
    column: str      # column name ("review")

    type_name: ClassVar[str] = "quail.column_ref"


# the text between the answer cue and a scored label: one space, as
# after a colon
LABEL_PREFIX = " "


@dataclass(frozen=True)
class Prompt:
    """Bound prompt text and token layout for one LLM call.

    Token counts are filled when a tokenizer is available; None means the
    planner must calculate them. A classification prompt keeps the text
    before each label in label_prefix. Its optional lettered prompt names
    each label with a distinct one-token letter for the letters rule.
    """
    template: str
    args: tuple    # tuple[ColumnRef, ...] in placeholder order
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


# the planner's estimate for a predicate written without a selectivity
DEFAULT_SELECTIVITY = 0.2

# the comparisons an AI.SCORE predicate may use, in SQL spelling
SCORE_COMPARISONS = ("<", "<=", ">", ">=")

# what a model call answers: "boolean" is AI.IF, yes or no;
# "score" is AI.SCORE, a float between 0 and 1; "label" is
# AI.CLASSIFY, one label from a fixed list
MODEL_CALL_KINDS = ("boolean", "score", "label")


def effective_selectivity(selectivity: Optional[float]) -> float:
    """Return the selectivity the planner uses: the given one or the default."""
    return DEFAULT_SELECTIVITY if selectivity is None else selectivity


# The suffix of the column holding each label's probability, beside an
# AI.CLASSIFY label column
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
                f"model call kind must be one of {MODEL_CALL_KINDS}, "
                f"got {self.kind!r}")
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
    if any(not isinstance(label, str) or not label.strip()
           for label in labels):
        raise CompileError("every AI.CLASSIFY label is non-empty text")
    folded = [label.casefold() for label in labels]
    if len(set(folded)) != len(folded):
        raise CompileError(
            f"AI.CLASSIFY labels must differ ignoring case, got {list(labels)}")
    if descriptions and len(descriptions) != len(labels):
        raise CompileError(
            "AI.CLASSIFY needs one description per label, empty for none")
    for description in descriptions:
        if not isinstance(description, str):
            raise CompileError("an AI.CLASSIFY description is text or empty")
        if len(description.split()) > DESCRIPTION_WORDS:
            raise CompileError(
                f"an AI.CLASSIFY description has at most {DESCRIPTION_WORDS} "
                f"words, got {len(description.split())}")


# the word limits Snowflake documents for AI_CLASSIFY
DESCRIPTION_WORDS = 25
TASK_DESCRIPTION_WORDS = 50


def validate_task_description(text: str) -> None:
    """Check an AI.CLASSIFY task description's length.

    Raises:
        CompileError: More than TASK_DESCRIPTION_WORDS words.
    """
    if len(text.split()) > TASK_DESCRIPTION_WORDS:
        raise CompileError(
            f"an AI.CLASSIFY task description has at most "
            f"{TASK_DESCRIPTION_WORDS} words, got {len(text.split())}")


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
            raise CompileError(
                f"unsupported AI.SCORE comparison {self.comparison!r}")
        if not 0.0 <= self.threshold <= 1.0:
            raise CompileError("AI.SCORE threshold must be between 0 and 1")


@dataclass(frozen=True)
class InList:
    """A column's value is one of the listed values, as SQL IN."""
    column: ColumnRef
    values: tuple    # tuple[str, ...]

    type_name: ClassVar[str] = "quail.in_list"

    def validate(self) -> None:
        # only a repeated value is an error; a value the column never
        # holds matches no row
        if len(set(self.values)) != len(self.values):
            raise CompileError(
                f"{self.column.alias}.{self.column.column} IN lists a "
                f"value twice")


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
    return any(
        is_score(predicate.expression)
        for predicates in operators.filters.values()
        for predicate in predicates
    ) or any(is_score(join.predicate) for join in operators.joins) or any(
        isinstance(expression, Alias) and expression.expression.kind == "score"
        for expression in operators.projections
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
                "an AI.SCORE call is a predicate only when compared with "
                "a threshold")
        if expression.kind == "label":
            raise CompileError(
                "an AI.CLASSIFY label is tested by a Filter on its "
                "column, not by a SemanticFilter")
        return
    raise CompileError(
        f"a predicate is a model call or a comparison, got "
        f"{type(expression).__name__}")


@dataclass(frozen=True)
class FilterPredicate:
    """One predicate over a table's documents, with its selectivity hint."""
    expression: Any                       # ModelCall | Compare
    selectivity: Optional[float] = None   # fraction of documents that
    #                                       pass; None means not given

    type_name: ClassVar[str] = "quail.filter_predicate"

    @property
    def prompt(self) -> Prompt:
        """The prompt the predicate asks."""
        return model_call(self.expression).prompt


class LogicalNode(Protocol):
    """Node in a logical query plan."""

    type_name: ClassVar[str]

    def children(self) -> tuple["LogicalNode", ...]: ...

    def expressions(self) -> tuple[Any, ...]: ...

    def output_schema(self) -> tuple[ColumnRef, ...]: ...

    def validate(self) -> None: ...

    def with_children(
        self, children: tuple["LogicalNode", ...]
    ) -> "LogicalNode": ...

    def with_expressions(self, expressions: tuple[Any, ...]) -> "LogicalNode": ...

    def explain_fields(self) -> dict[str, Any]: ...


COLUMN_COMPARISONS = ("=", "<>", "<", "<=", ">", ">=", "in", "is null",
                      "is not null")


@dataclass(frozen=True)
class ColumnPredicate:
    """A test of one source column against literals, decided without the model.

    ``value`` is the literal compared with, a tuple of literals for
    ``in``, and None for the null tests.
    """
    column: ColumnRef
    comparison: str
    value: Any = None

    type_name: ClassVar[str] = "quail.column_predicate"

    def aliases(self) -> tuple[str, ...]:
        return (self.column.alias,)

    def validate(self) -> None:
        if not isinstance(self.column, ColumnRef):
            raise CompileError("a column predicate tests a column reference")
        if self.comparison not in COLUMN_COMPARISONS:
            raise CompileError(
                f"unsupported column comparison {self.comparison!r}; "
                f"supported: {', '.join(COLUMN_COMPARISONS)}")
        if self.comparison == "in":
            if not isinstance(self.value, tuple) or not self.value:
                raise CompileError("IN needs a non-empty list of literals")
        elif self.comparison.startswith("is"):
            if self.value is not None:
                raise CompileError("a null test takes no value")
        elif self.value is None or isinstance(self.value, (tuple, list)):
            raise CompileError(
                f"{self.comparison} compares with one literal")

    def __str__(self) -> str:
        name = f"{self.column.alias}.{self.column.column}"
        if self.comparison == "in":
            return f"{name} IN ({', '.join(repr(v) for v in self.value)})"
        if self.comparison.startswith("is"):
            return f"{name} {self.comparison.upper()}"
        return f"{name} {self.comparison} {self.value!r}"


@dataclass(frozen=True)
class Scan:
    """Which column of which provider supplies the document text.

    ``column`` is tokenized for the model. ``columns`` names the source
    columns kept as values for the result rows, and includes ``column``
    only when the query returns the document text itself. The
    projection pushdown rule fills ``columns``; before it runs the
    tuple is empty. ``predicates`` are column tests, all of which a
    document must pass before any operator reads it.
    """
    provider: str
    alias: str
    column: str
    columns: tuple = ()    # tuple[str, ...]
    predicates: tuple = ()    # tuple[ColumnPredicate, ...]

    type_name: ClassVar[str] = "quail.scan"

    def children(self) -> tuple:
        return ()

    def expressions(self) -> tuple:
        return self.predicates

    def output_schema(self) -> tuple[ColumnRef, ...]:
        fields = [ColumnRef(self.alias, self.provider, self.column)]
        fields.extend(
            ColumnRef(self.alias, self.provider, column)
            for column in self.columns if column != self.column
        )
        return tuple(fields)

    def validate(self) -> None:
        if not self.provider or not self.alias or not self.column:
            raise CompileError("Scan needs a provider, alias, and column")
        if len(set(self.columns)) != len(self.columns):
            raise CompileError(
                f"Scan {self.alias!r} lists a column twice: {self.columns}")
        for predicate in self.predicates:
            if not isinstance(predicate, ColumnPredicate):
                raise CompileError(
                    f"a Scan predicate is a ColumnPredicate, got "
                    f"{type(predicate).__name__}")
            predicate.validate()
            if predicate.column.alias != self.alias:
                raise CompileError(
                    f"predicate {predicate} is on the Scan of "
                    f"{self.alias!r}")

    def with_children(self, children: tuple) -> "Scan":
        if children:
            raise CompileError("Scan has no children")
        return self

    def with_expressions(self, expressions: tuple) -> "Scan":
        return replace(self, predicates=tuple(expressions))

    def explain_fields(self) -> dict:
        return {
            "provider": self.provider,
            "alias": self.alias,
            "column": self.column,
            "columns": list(self.columns),
            "predicates": [str(predicate) for predicate in self.predicates],
        }


@dataclass(frozen=True)
class SemanticFilter:
    """Predicates over one table's documents, in written order.

    ``order`` is the predicates' execution order, as positions in
    ``predicates``, once the filter_order rule has chosen one; empty
    means as written.
    """
    input: LogicalNode
    predicates: tuple    # tuple[FilterPredicate, ...], written order
    order: tuple = ()    # tuple[int, ...]

    type_name: ClassVar[str] = "quail.semantic_filter"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return self.predicates

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return self.input.output_schema()

    def validate(self) -> None:
        if not self.predicates:
            raise CompileError("SemanticFilter needs at least one predicate")
        if self.order and sorted(self.order) != list(range(len(self.predicates))):
            raise CompileError(
                f"a filter's order lists each predicate position once, "
                f"got {list(self.order)} for {len(self.predicates)} predicates")
        for predicate in self.predicates:
            validate_predicate(predicate.expression)
            if len(model_call(predicate.expression).aliases()) != 1:
                raise CompileError(
                    "a SemanticFilter predicate reads one table")

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 1:
            raise CompileError("SemanticFilter needs one child")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple):
        if not expressions:
            raise CompileError("SemanticFilter needs at least one predicate")
        return replace(self, predicates=expressions)

    def explain_fields(self) -> dict:
        return {
            "predicates": len(self.predicates),
            "selectivities": [p.selectivity for p in self.predicates],
            "expressions": [_explain(p.expression) for p in self.predicates],
            "order": list(self.order),
        }


@dataclass(frozen=True)
class Filter:
    """Rows whose condition holds, decided without the model.

    The condition tests the label column a one-table SemanticClassify
    below adds.
    """
    input: LogicalNode
    condition: InList
    selectivity: Optional[float] = None   # fraction of rows that pass;
    #                                       None means not given

    type_name: ClassVar[str] = "quail.filter"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return (self.condition,)

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return self.input.output_schema()

    def validate(self) -> None:
        self.condition.validate()
        column = self.condition.column
        if not any(node.alias == column.alias and node.name == column.column
                   and len(node.call.aliases()) == 1
                   for node in classifications(self.input)):
            raise CompileError(
                f"a Filter tests the label column of a one-table "
                f"classification below it; nothing below computes "
                f"{column.alias}.{column.column}")

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 1:
            raise CompileError("Filter needs one child")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple):
        if len(expressions) != 1:
            raise CompileError("Filter needs one condition")
        return replace(self, condition=expressions[0])

    def explain_fields(self) -> dict:
        return {
            "condition": _explain(self.condition),
            "selectivity": self.selectivity,
        }


@dataclass(frozen=True)
class Equality:
    """One ordinary join condition: two columns of two tables are equal."""
    left: ColumnRef
    right: ColumnRef

    type_name: ClassVar[str] = "quail.equality"

    def aliases(self) -> tuple[str, str]:
        return self.left.alias, self.right.alias

    def __str__(self) -> str:
        return (f"{self.left.alias}.{self.left.column} = "
                f"{self.right.alias}.{self.right.column}")


@dataclass(frozen=True)
class Join:
    """One binary relational join; ``on`` empty means a cross join.

    The pairs it produces are the tuples a SemanticJoin above it asks
    the model about. Every Equality names one column on each side.
    """
    left: LogicalNode
    right: LogicalNode
    on: tuple = ()    # tuple[Equality, ...]

    type_name: ClassVar[str] = "quail.join"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.left, self.right)

    def expressions(self) -> tuple:
        return self.on

    def output_schema(self) -> tuple[ColumnRef, ...]:
        fields = list(self.left.output_schema())
        fields.extend(field for field in self.right.output_schema()
                      if field not in fields)
        return tuple(fields)

    def validate(self) -> None:
        left = {field.alias for field in self.left.output_schema()}
        right = {field.alias for field in self.right.output_schema()}
        for condition in self.on:
            sides = set(condition.aliases())
            if not (sides & left and sides & right) or len(sides) != 2:
                raise CompileError(
                    f"join condition {condition} must name one table "
                    f"on each side of the join ({sorted(left)} and "
                    f"{sorted(right)})")

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 2:
            raise CompileError("Join needs two children")
        return replace(self, left=children[0], right=children[1])

    def with_expressions(self, expressions: tuple):
        return replace(self, on=tuple(expressions))

    def explain_fields(self) -> dict:
        return {"on": [str(condition) for condition in self.on]
                or "cross"}


@dataclass(frozen=True)
class SemanticJoin:
    """One model predicate over the pairs its input Join produces.

    A query that joins three or more tables has one SemanticJoin per
    pair, each feeding the Join of the next. ``exec_idx`` and
    ``exec_anchor`` are the join_order rule's decision: the join's
    position among the query's joins in execution order and the table
    whose KV its stage keeps, ``anchor`` when one was written. Both
    are None before the rule runs.
    """
    input: LogicalNode                 # a Join, under any Apply that
    #                                    returns its pairs
    predicate: Any                     # ModelCall | Compare
    semantics: str = "full"            # full | exists | anti
    selectivity: Optional[float] = None    # fraction of tuples that pass
    anchor: Optional[str] = None       # table alias whose KV is kept;
    #                                    None = planner picks
    exec_idx: Optional[int] = None
    exec_anchor: Optional[str] = None

    type_name: ClassVar[str] = "quail.semantic_join"

    @property
    def prompt(self) -> Prompt:
        """The prompt asked of every pair."""
        return model_call(self.predicate).prompt

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return (self.predicate,)

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return self.input.output_schema()

    def validate(self) -> None:
        if self.semantics not in {"full", "exists", "anti"}:
            raise CompileError(
                f"unknown join semantics {self.semantics!r}")
        validate_predicate(self.predicate)
        present = {field.alias for field in self.input.output_schema()}
        aliases = model_call(self.predicate).aliases()
        missing = set(aliases) - present
        if missing:
            raise CompileError(
                f"the join predicate reads {sorted(missing)}, which its "
                f"input does not produce ({sorted(present)})")
        if (self.exec_idx is None) != (self.exec_anchor is None):
            raise CompileError(
                "a planned join has both an execution position and an anchor")
        if self.exec_anchor is not None and self.exec_anchor not in aliases:
            raise CompileError(
                f"the planned anchor {self.exec_anchor!r} is not a table "
                f"of the join ({list(aliases)})")

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 1:
            raise CompileError("SemanticJoin needs one input")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple):
        if len(expressions) != 1:
            raise CompileError("SemanticJoin needs one predicate")
        return replace(self, predicate=expressions[0])

    def explain_fields(self) -> dict:
        return {
            "semantics": self.semantics,
            "selectivity": self.selectivity,
            "anchor": self.anchor,
            "expression": _explain(self.predicate),
            "exec_idx": self.exec_idx,
            "exec_anchor": self.exec_anchor,
        }


@dataclass(frozen=True)
class SemanticClassify:
    """One AI.CLASSIFY call over its input rows, adding the label column.

    Every input row passes through with its label in the column named
    ``name``; with ``probabilities`` a second column, ``name`` plus
    PROBABILITIES_SUFFIX, holds each label's probability. A call over
    one table starts on that table above its AI.IF SemanticFilter and
    below any join; the lift_classifications rule moves it above the
    joins when a join reads the table. A call over two tables sits
    above the SemanticJoin of the two. A Filter testing the label
    column sits above this node, and the root Project returns the
    column through an Alias of the same call.
    """
    input: LogicalNode
    call: ModelCall
    name: str

    type_name: ClassVar[str] = "quail.semantic_classify"

    @property
    def probabilities(self) -> bool:
        """Whether the probabilities column is added beside the label."""
        return self.call.probabilities

    @property
    def alias(self) -> str:
        """The table alias the label column belongs to."""
        return self.call.aliases()[0]

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return (self.call,)

    def output_schema(self) -> tuple[ColumnRef, ...]:
        fields = self.input.output_schema()
        provider = next((field.provider for field in fields
                         if field.alias == self.alias), "")
        added = [ColumnRef(self.alias, provider, self.name)]
        if self.probabilities:
            added.append(ColumnRef(self.alias, provider,
                                   self.name + PROBABILITIES_SUFFIX))
        return fields + tuple(added)

    def validate(self) -> None:
        self.call.validate()
        if self.call.kind != "label":
            raise CompileError("SemanticClassify needs an AI.CLASSIFY call")
        if not self.name or "." in self.name:
            raise CompileError(
                f"a classification needs a column name without a dot, "
                f"got {self.name!r}")
        aliases = self.call.aliases()
        if len(aliases) not in (1, 2):
            raise CompileError(
                "AI.CLASSIFY reads one document, or one from each side "
                "of a join")
        present = {field.alias for field in self.input.output_schema()}
        missing = set(aliases) - present
        if missing:
            raise CompileError(
                f"the classification reads {sorted(missing)}, which its "
                f"input does not produce ({sorted(present)})")
        if any(field.alias == self.alias and field.column == self.name
               for field in self.input.output_schema()):
            raise CompileError(f"the name {self.name!r} is already used")
        if len(aliases) == 2 and not any(
                isinstance(node, SemanticJoin)
                and set(model_call(node.predicate).aliases()) == set(aliases)
                for node in _subtree(self.input)):
            raise CompileError(
                f"the classification of {aliases[0]!r} x {aliases[1]!r} "
                f"joined rows needs a join of the two")

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 1:
            raise CompileError("SemanticClassify needs one input")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple):
        if len(expressions) != 1:
            raise CompileError("SemanticClassify needs one call")
        return replace(self, call=expressions[0])

    def explain_fields(self) -> dict:
        return {
            "name": self.name,
            "expression": _explain(self.call),
            "probabilities": self.probabilities,
        }


def _subtree(node) -> tuple:
    """Return the node and every node below it, children first."""
    nodes = []
    for child in node.children():
        nodes.extend(_subtree(child))
    nodes.append(node)
    return tuple(nodes)


def classifications(node) -> tuple:
    """Return the SemanticClassify nodes at or below a node, children first."""
    return tuple(found for found in _subtree(node)
                 if isinstance(found, SemanticClassify))


def classified_above_joins(root) -> frozenset:
    """Return the aliases whose one-table classification sits above a join.

    The planner runs such a classification after the joins, over the
    documents the joins matched.
    """
    return frozenset(
        node.alias for node in classifications(root)
        if len(node.call.aliases()) == 1
        and any(isinstance(below, SemanticJoin) for below in _subtree(node.input)))


def _explain(expression) -> str:
    if isinstance(expression, Compare):
        return (f"{_explain(expression.call)} {expression.comparison} "
                f"{expression.threshold}")
    if isinstance(expression, InList):
        return (f"{expression.column.alias}.{expression.column.column} "
                f"IN {list(expression.values)}")
    if isinstance(expression, Alias):
        if expression.expression.kind == "label":
            # the SemanticClassify below computes the column
            return expression.name
        return f"{_explain(expression.expression)} AS {expression.name}"
    if expression.kind == "label":
        return (f"AI.CLASSIFY({expression.prompt.template!r}, "
                f"{list(expression.labels)})")
    function = "AI.SCORE" if expression.kind == "score" else "AI.IF"
    return f"{function}({expression.prompt.template!r})"


APPLY_KINDS = ("per_batch", "barrier")
APPLY_IDS = ("preserve", "drop", "pairs")


@dataclass(frozen=True)
class Apply:
    """A user function between two operators.

    The function is registered on the session under ``function`` and
    receives an Arrow table per alias: the alias's row indices under
    the alias name plus the listed columns. ``kind`` says how it runs:
    ``per_batch`` on each batch of survivors a streaming operator hands
    over, ``barrier`` once over every survivor. ``ids`` says what it
    returns: ``preserve`` every input id, ``drop`` a subset of them,
    ``pairs`` a relation of (left alias, right alias) rows for the join
    at ``written_pos``. A function never invents an id.
    """
    input: LogicalNode
    function: str
    kind: str
    ids: str
    columns: tuple            # tuple[ColumnRef, ...]
    aliases: tuple            # the alias, or the join's two aliases
    written_pos: Optional[int] = None

    type_name: ClassVar[str] = "quail.apply"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return self.columns

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return self.input.output_schema()

    def validate(self) -> None:
        if self.kind not in APPLY_KINDS:
            raise CompileError(
                f"apply kind must be one of {APPLY_KINDS}, got {self.kind!r}")
        if self.ids not in APPLY_IDS:
            raise CompileError(
                f"apply ids must be one of {APPLY_IDS}, got {self.ids!r}")
        if not self.function:
            raise CompileError("apply needs a registered function name")
        present = {field.alias for field in self.input.output_schema()}
        if not set(self.aliases) <= present:
            raise CompileError(
                f"apply {self.function!r} names tables {self.aliases} "
                f"outside its input ({sorted(present)})")
        for ref in self.columns:
            if ref.alias not in self.aliases:
                raise CompileError(
                    f"apply {self.function!r} reads {ref.alias}.{ref.column} "
                    f"but works on {self.aliases}")
        if (self.ids == "pairs") != (len(self.aliases) == 2):
            raise CompileError(
                "an apply returning pairs works on exactly two tables; "
                "one returning ids works on one")

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 1:
            raise CompileError("Apply needs one child")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple):
        return replace(self, columns=tuple(expressions))

    def explain_fields(self) -> dict:
        return {
            "function": self.function,
            "kind": self.kind,
            "ids": self.ids,
            "columns": [f"{ref.alias}.{ref.column}" for ref in self.columns],
        }


@dataclass(frozen=True)
class Project:
    """Column projection. Always the root operator."""
    input: LogicalNode
    columns: tuple    # tuple[ColumnRef | Alias, ...]
    limit: Optional[int] = None

    type_name: ClassVar[str] = "quail.logical_project"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return self.columns

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return tuple(
            column for column in self.columns
            if isinstance(column, ColumnRef)
        )

    def validate(self) -> None:
        if not self.columns:
            raise CompileError("Project needs at least one column")
        computed = {(node.call, node.name)
                    for node in classifications(self.input)}
        for column in self.columns:
            if isinstance(column, Alias):
                column.validate()
                if column.expression.kind == "label" \
                        and (column.expression, column.name) not in computed:
                    raise CompileError(
                        f"the label column {column.name!r} needs a "
                        f"SemanticClassify below the Project")
            elif not isinstance(column, ColumnRef):
                raise CompileError(
                    f"a projected column is a column reference or a named "
                    f"expression, got {type(column).__name__}")
        names = [
            f"{column.alias}.{column.column}"
            if isinstance(column, ColumnRef) else column.name
            for column in self.columns
        ]
        if len(names) != len(set(names)):
            raise CompileError(f"projection names must be unique, got {names}")
        if self.limit is not None and self.limit <= 0:
            raise CompileError("LIMIT must be a positive integer")

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 1:
            raise CompileError("Project needs one child")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple):
        if not expressions:
            raise CompileError("Project needs at least one column")
        return replace(self, columns=expressions)

    def explain_fields(self) -> dict:
        return {
            "columns": [
                (f"{column.alias}.{column.column}"
                 if isinstance(column, ColumnRef) else _explain(column))
                for column in self.columns
            ],
            "limit": self.limit,
        }


@dataclass(frozen=True)
class LabelFilter:
    """A Filter on a one-table classification's label, as the planner reads it.

    Attributes:
        call: The classification that computes the label.
        values: The labels the Filter keeps.
        selectivity: The Filter's selectivity hint, or None.
        position: Its written position among the table's filters,
            counted after the table's AI.IF predicates.
    """

    call: "ModelCall"
    values: tuple
    selectivity: Optional[float]
    position: int


@dataclass(frozen=True)
class LabelWork:
    """Classification calls, result columns, and label filters in a plan.

    Attributes:
        calls: (classification call, anchor alias) pairs. Calls a Filter
            tests appear first in filter order, followed by calls
            returned in SELECT.
        names: Mapping from classification call to its column name, as the
            SemanticClassify nodes name them.
        tests: Written filter positions of the Filters on each classification.
        projected: Mapping from classifications returned in SELECT to column names.
    """

    calls: tuple
    names: dict
    tests: dict
    projected: dict


def _label_work(classifies, columns, label_filters) -> LabelWork:
    projected = {column.expression: column.name
                 for column in columns
                 if isinstance(column, Alias) and column.expression.kind == "label"}
    names = {node.call: node.name for node in classifies}
    tests = {}
    calls = []
    for alias, label_filters_of_alias in label_filters.items():
        for test in label_filters_of_alias:
            if test.call not in tests:
                calls.append((test.call, alias))
            tests.setdefault(test.call, []).append(test.position)
    for call in projected:
        if call not in tests:
            calls.append((call, call.aliases()[0]))
    return LabelWork(tuple(calls), names, tests, projected)


@dataclass(frozen=True)
class LogicalPlan:
    root: LogicalNode

    def output_schema(self) -> tuple[ColumnRef, ...]:
        """Return the columns produced by the root node."""
        return self.root.output_schema()

    def walk(self) -> tuple[LogicalNode, ...]:
        """Return logical nodes with each child before its parent."""
        nodes = []

        def visit(node):
            for child in node.children():
                visit(child)
            nodes.append(node)

        visit(self.root)
        return tuple(nodes)

    def validate(self) -> None:
        """Validate every logical node."""
        for node in self.walk():
            node.validate()

    def operators(self) -> "Operators":
        """Return the plan's operators as the planner reads them."""
        scans, filters, joins, applies, classifies = [], {}, [], [], []
        tested, aliases = {}, []
        for node in self.walk():
            if isinstance(node, Scan):
                scans.append(node)
            elif isinstance(node, SemanticFilter):
                for predicate in node.predicates:
                    (alias,) = model_call(predicate.expression).aliases()
                    filters.setdefault(alias, []).append(predicate)
                    if alias not in aliases:
                        aliases.append(alias)
            elif isinstance(node, Filter):
                alias = node.condition.column.alias
                tested.setdefault(alias, []).append(node)
                if alias not in aliases:
                    aliases.append(alias)
            elif isinstance(node, SemanticJoin):
                joins.append(node)
            elif isinstance(node, Apply):
                applies.append(node)
            elif isinstance(node, SemanticClassify):
                classifies.append(node)
        calls = {(node.alias, node.name): node.call for node in classifies
                 if len(node.call.aliases()) == 1}
        label_filters = {
            alias: tuple(
                LabelFilter(
                    call=calls[(alias, node.condition.column.column)],
                    values=node.condition.values,
                    selectivity=node.selectivity,
                    position=len(filters.get(alias, ())) + index)
                for index, node in enumerate(tested[alias]))
            for alias in aliases if alias in tested}
        columns = self.root.columns if isinstance(self.root, Project) else ()
        return Operators(
            scans=tuple(scans),
            filters={alias: tuple(filters[alias]) for alias in aliases
                     if alias in filters},
            label_filters=label_filters,
            joins=tuple(joins),
            applies=tuple(applies),
            projections=tuple(column for column in columns
                              if isinstance(column, Alias)),
            labels=_label_work(classifies, columns, label_filters),
            classifies=tuple(classifies),
        )


@dataclass(frozen=True)
class Operators:
    """The operators of one plan, in the order the plan was written.

    ``walk`` lists children before parents, so scans, joins, applies,
    and classifications follow written order, each alias's AI.IF
    predicates keep their order, and each alias's label filters keep
    theirs.
    """
    scans: tuple           # tuple[Scan, ...]
    filters: dict          # alias -> tuple[FilterPredicate, ...]
    joins: tuple           # tuple[SemanticJoin, ...]
    applies: tuple         # tuple[Apply, ...]
    labels: LabelWork
    projections: tuple = ()     # named model calls in SELECT order
    classifies: tuple = ()      # tuple[SemanticClassify, ...]
    label_filters: dict = field(default_factory=dict)
    #                     alias -> tuple[LabelFilter, ...]

    def all_filters(self) -> dict:
        """Return alias -> its AI.IF predicates, then its LabelFilters."""
        aliases = [*self.filters, *(alias for alias in self.label_filters
                                    if alias not in self.filters)]
        return {alias: (*self.filters.get(alias, ()),
                        *self.label_filters.get(alias, ()))
                for alias in aliases}

    @property
    def prompts(self) -> tuple:
        """Return filter, label, join, and remaining result-column prompts in order."""
        calls = tuple(
            model_call(predicate.expression)
            for predicates in self.filters.values()
            for predicate in predicates
        ) + tuple(
            test.call
            for tests in self.label_filters.values()
            for test in tests
        ) + tuple(model_call(join.predicate) for join in self.joins)
        return tuple(call.prompt for call in calls) + tuple(
            column.expression.prompt for column in self.projections
            if column.expression not in calls
        )


def join_outer_input(join: "SemanticJoin") -> "LogicalNode":
    """The tree the join extends: everything before its new tables."""
    node = join.input
    while isinstance(node, (Join, Apply)):
        node = node.left if isinstance(node, Join) else node.input
    return node


def join_conditions(join: "SemanticJoin") -> tuple:
    """The Equality conditions under one SemanticJoin, in written order."""
    conditions = []

    def visit(node):
        if isinstance(node, Apply):
            visit(node.input)
        elif isinstance(node, Join):
            visit(node.left)
            conditions.extend(node.on)

    visit(join.input)
    return tuple(conditions)


def oriented_join_conditions(join: "SemanticJoin") -> tuple | None:
    """Return one alias order and every equality oriented to that order."""
    conditions = join_conditions(join)
    if not conditions:
        return None
    left_alias, right_alias = conditions[0].aliases()
    oriented = []
    for condition in conditions:
        left, right = condition.left, condition.right
        if left.alias != left_alias:
            left, right = right, left
        oriented.append((left, right))
    return left_alias, right_alias, tuple(oriented)


def join_applies(join: "SemanticJoin") -> tuple:
    """The Apply nodes that return pairs for one SemanticJoin."""
    applies = []
    node = join.input
    while isinstance(node, Apply):
        applies.append(node)
        node = node.input
    return tuple(reversed(applies))


@dataclass(frozen=True)
class JoinSpec:
    """The join predicate (or one EXISTS/anti term), pre-assembly."""
    aliases: tuple             # newly joined table aliases
    predicate: Any             # ModelCall | Compare
    semantics: str = "full"
    selectivity: Optional[float] = None
    anchor: Optional[str] = None
    on: tuple = ()             # tuple[Equality, ...] over the tables
    applies: tuple = ()        # (function, kind, columns) returning pairs

    @property
    def prompt(self) -> Prompt:
        """The prompt asked of every tuple."""
        return model_call(self.predicate).prompt


class LogicalPlanBuilder:
    """Build the built in logical nodes used by both front ends."""

    def __init__(self):
        self._tables = []
        self._nodes = {}
        self._root = None
        self._joins = 0

    def add_scan(
        self,
        alias: str,
        provider: str,
        column: str,
        predicates: tuple[FilterPredicate, ...] = (),
        applies: tuple = (),
        labels: tuple = (),
        label_filters: tuple = (),
        column_predicates: tuple = (),
    ) -> None:
        """Add one table with its filters, applies, and classifications.

        Args:
            alias: The table's alias in the query.
            provider: The registered provider name.
            column: The document column.
            predicates: The table's AI.IF predicates in written order.
            column_predicates: ColumnPredicate tests the scan applies
                before any operator reads a document.
            applies: (function, kind, ids, columns) per apply, in order.
            labels: (call, name) per one-table classification. A call
                that a label filter tests sits below the first Filter
                testing it; the others sit above the last Filter, in the
                order given.
            label_filters: (name, values, selectivity) per Filter on a
                classification's label column, in written order.

        Raises:
            CompileError: The alias is taken, or a label filter names a
                column no classification in labels computes.
        """
        if alias in self._nodes:
            raise CompileError(f"duplicate table alias {alias!r}")
        node = Scan(provider=provider, alias=alias, column=column,
                    predicates=tuple(column_predicates))
        if predicates:
            node = SemanticFilter(node, tuple(predicates))
        for function, kind, ids, columns in applies:
            node = Apply(node, function=function, kind=kind, ids=ids,
                         columns=tuple(columns), aliases=(alias,))
        calls = {name: call for call, name in labels}
        classified = []
        for name, values, selectivity in label_filters:
            if name not in calls:
                raise CompileError(
                    f"a filter on a label of {alias!r} tests a "
                    f"classification the query does not name")
            if calls[name] not in classified:
                node = SemanticClassify(node, calls[name], name)
                classified.append(calls[name])
            node = Filter(node, InList(ColumnRef(alias, provider, name),
                                       tuple(values)), selectivity)
        for call, name in labels:
            if call not in classified:
                node = SemanticClassify(node, call, name)
                classified.append(call)
        self._tables.append(alias)
        self._nodes[alias] = node
        if self._root is None:
            self._root = node

    def add_classify(self, call: ModelCall, name: str) -> None:
        """Classify the rows of the join the call reads.

        The node sits directly above the SemanticJoin over the call's
        two tables.
        """
        aliases = set(call.aliases())

        def insert(node):
            if isinstance(node, SemanticJoin) \
                    and set(model_call(node.predicate).aliases()) == aliases:
                return SemanticClassify(node, call, name)
            if isinstance(node, (SemanticJoin, SemanticClassify, Apply)):
                return replace(node, input=insert(node.input))
            if isinstance(node, Join):
                return replace(node, left=insert(node.left))
            raise CompileError(
                f"the classification of {call.aliases()[0]!r} x "
                f"{call.aliases()[1]!r} joined rows needs a join of the two")

        if self._root is None or len(aliases) != 2:
            raise CompileError(
                "a classification of joined rows reads one document from "
                "each side of a join")
        self._root = insert(self._root)

    def add_join(self, join: JoinSpec) -> None:
        """Join each new table onto the tree, then ask the prompt.

        The equalities go on the Join that brings the last table in;
        both front ends only produce conditions over that table.
        """
        if self._root is None:
            raise CompileError("a logical join needs an input table")
        root = self._root
        for alias in join.aliases:
            root = Join(root, self._nodes[alias])
        present = {field.alias for field in root.output_schema()}
        for condition in join.on:
            if not set(condition.aliases()) <= present:
                raise CompileError(
                    f"join condition {condition} names a table this join "
                    f"does not bring in ({list(join.aliases)}); put it on "
                    f"the JOIN that introduces the table")
        if join.on:
            root = replace(root, on=tuple(join.on))
        written_pos = self._joins
        self._joins += 1
        for function, kind, columns in join.applies:
            aliases = tuple(dict.fromkeys(ref.alias for ref in join.prompt.args))
            root = Apply(root, function=function, kind=kind, ids="pairs",
                         columns=tuple(columns), aliases=aliases,
                         written_pos=written_pos)
        self._root = SemanticJoin(
            input=root,
            predicate=join.predicate,
            semantics=join.semantics,
            selectivity=join.selectivity,
            anchor=join.anchor,
        )

    def add_cross_join(self, alias: str) -> None:
        """Add one relational cross join without an AI predicate."""
        if self._root is None:
            raise CompileError("a logical join needs an input table")
        self._root = Join(self._root, self._nodes[alias])

    def project(
        self, columns: tuple, limit: int | None = None
    ) -> LogicalPlan:
        if self._root is None:
            raise CompileError("a logical plan needs an input table")
        plan = LogicalPlan(Project(self._root, tuple(columns), limit))
        plan.validate()
        return plan
