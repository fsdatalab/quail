"""Logical query operators and their validation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, ClassVar, Optional, Protocol

from quail.logical.expressions import (
    PROBABILITIES_SUFFIX,
    AggregateCall,
    Aggregation,
    Alias,
    ColumnPredicate,
    ColumnRef,
    CompileError,
    HavingTest,
    InList,
    ModelCall,
    Prompt,
    SortKey,
    _explain,
    column_name,
    model_call,
    validate_predicate,
)


class LogicalNode(Protocol):
    """Node in a logical query plan."""

    type_name: ClassVar[str]

    def children(self) -> tuple["LogicalNode", ...]: ...

    def expressions(self) -> tuple[Any, ...]: ...

    def output_schema(self) -> tuple[ColumnRef, ...]: ...

    def validate(self) -> None: ...

    def with_children(self, children: tuple["LogicalNode", ...]) -> "LogicalNode": ...

    def with_expressions(self, expressions: tuple[Any, ...]) -> "LogicalNode": ...

    def explain_fields(self) -> dict[str, Any]: ...


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
    columns: tuple = ()  # tuple[str, ...]
    predicates: tuple = ()  # tuple[ColumnPredicate, ...]

    type_name: ClassVar[str] = "quail.scan"

    def children(self) -> tuple:
        return ()

    def expressions(self) -> tuple:
        return self.predicates

    def output_schema(self) -> tuple[ColumnRef, ...]:
        fields = [ColumnRef(self.alias, self.provider, self.column)]
        fields.extend(
            ColumnRef(self.alias, self.provider, column)
            for column in self.columns
            if column != self.column
        )
        return tuple(fields)

    def validate(self) -> None:
        if not self.provider or not self.alias or not self.column:
            raise CompileError("Scan needs a provider, alias, and column")
        if len(set(self.columns)) != len(self.columns):
            raise CompileError(
                f"Scan {self.alias!r} lists a column twice: {self.columns}"
            )
        for predicate in self.predicates:
            if not isinstance(predicate, ColumnPredicate):
                raise CompileError(
                    f"a Scan predicate is a ColumnPredicate, got "
                    f"{type(predicate).__name__}"
                )
            predicate.validate()
            if predicate.column.alias != self.alias:
                raise CompileError(
                    f"predicate {predicate} is on the Scan of {self.alias!r}"
                )

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
    means as written. ``stop_key`` names columns of the filtered table
    once the per_key_stop rule has found that the query asks only
    whether any document of each key value passes; the executor then
    skips a key's remaining documents once one survives.
    """

    input: LogicalNode
    predicates: tuple  # tuple[FilterPredicate, ...], written order
    order: tuple = ()  # tuple[int, ...]
    stop_key: tuple = ()  # tuple[str, ...]

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
                f"got {list(self.order)} for {len(self.predicates)} predicates"
            )
        for predicate in self.predicates:
            validate_predicate(predicate.expression)
            if len(model_call(predicate.expression).aliases()) != 1:
                raise CompileError("a SemanticFilter predicate reads one table")

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
            "stop_key": list(self.stop_key),
        }


@dataclass(frozen=True)
class Filter:
    """Rows whose condition holds, decided without the model.

    The condition tests the label column a one-table SemanticClassify
    below adds.
    """

    input: LogicalNode
    condition: InList
    selectivity: Optional[float] = None  # fraction of rows that pass;
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
        if not any(
            node.alias == column.alias
            and node.name == column.column
            and len(node.call.aliases()) == 1
            for node in classifications(self.input)
        ):
            raise CompileError(
                f"a Filter tests the label column of a one-table "
                f"classification below it; nothing below computes "
                f"{column.alias}.{column.column}"
            )

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
class Join:
    """One binary relational join; ``on`` empty means a cross join.

    The pairs it produces are the tuples a SemanticJoin above it asks
    the model about. Every Equality names one column on each side.
    """

    left: LogicalNode
    right: LogicalNode
    on: tuple = ()  # tuple[Equality, ...]

    type_name: ClassVar[str] = "quail.join"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.left, self.right)

    def expressions(self) -> tuple:
        return self.on

    def output_schema(self) -> tuple[ColumnRef, ...]:
        fields = list(self.left.output_schema())
        fields.extend(
            field for field in self.right.output_schema() if field not in fields
        )
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
                    f"{sorted(right)})"
                )

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 2:
            raise CompileError("Join needs two children")
        return replace(self, left=children[0], right=children[1])

    def with_expressions(self, expressions: tuple):
        return replace(self, on=tuple(expressions))

    def explain_fields(self) -> dict:
        return {"on": [str(condition) for condition in self.on] or "cross"}


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

    input: LogicalNode  # a Join, under any Apply that
    #                                    returns its pairs
    predicate: Any  # ModelCall | Compare
    semantics: str = "full"  # full | exists | anti
    selectivity: Optional[float] = None  # fraction of tuples that pass
    anchor: Optional[str] = None  # table alias whose KV is kept;
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
            raise CompileError(f"unknown join semantics {self.semantics!r}")
        validate_predicate(self.predicate)
        present = {field.alias for field in self.input.output_schema()}
        aliases = model_call(self.predicate).aliases()
        missing = set(aliases) - present
        if missing:
            raise CompileError(
                f"the join predicate reads {sorted(missing)}, which its "
                f"input does not produce ({sorted(present)})"
            )
        if (self.exec_idx is None) != (self.exec_anchor is None):
            raise CompileError(
                "a planned join has both an execution position and an anchor"
            )
        if self.exec_anchor is not None and self.exec_anchor not in aliases:
            raise CompileError(
                f"the planned anchor {self.exec_anchor!r} is not a table "
                f"of the join ({list(aliases)})"
            )

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
        provider = next(
            (field.provider for field in fields if field.alias == self.alias), ""
        )
        added = [ColumnRef(self.alias, provider, self.name)]
        if self.probabilities:
            added.append(
                ColumnRef(self.alias, provider, self.name + PROBABILITIES_SUFFIX)
            )
        return fields + tuple(added)

    def validate(self) -> None:
        self.call.validate()
        if self.call.kind != "label":
            raise CompileError("SemanticClassify needs an AI.CLASSIFY call")
        if not self.name or "." in self.name:
            raise CompileError(
                f"a classification needs a column name without a dot, got {self.name!r}"
            )
        aliases = self.call.aliases()
        if len(aliases) not in (1, 2):
            raise CompileError(
                "AI.CLASSIFY reads one document, or one from each side of a join"
            )
        present = {field.alias for field in self.input.output_schema()}
        missing = set(aliases) - present
        if missing:
            raise CompileError(
                f"the classification reads {sorted(missing)}, which its "
                f"input does not produce ({sorted(present)})"
            )
        if any(
            field.alias == self.alias and field.column == self.name
            for field in self.input.output_schema()
        ):
            raise CompileError(f"the name {self.name!r} is already used")
        if len(aliases) == 2 and not any(
            isinstance(node, SemanticJoin)
            and set(model_call(node.predicate).aliases()) == set(aliases)
            for node in _subtree(self.input)
        ):
            raise CompileError(
                f"the classification of {aliases[0]!r} x {aliases[1]!r} "
                f"joined rows needs a join of the two"
            )

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
    return tuple(
        found for found in _subtree(node) if isinstance(found, SemanticClassify)
    )


def classified_above_joins(root) -> frozenset:
    """Return the aliases whose one-table classification sits above a join.

    The planner runs such a classification after the joins, over the
    documents the joins matched.
    """
    return frozenset(
        node.alias
        for node in classifications(root)
        if len(node.call.aliases()) == 1
        and any(isinstance(below, SemanticJoin) for below in _subtree(node.input))
    )


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
    columns: tuple  # tuple[ColumnRef, ...]
    aliases: tuple  # the alias, or the join's two aliases
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
                f"apply kind must be one of {APPLY_KINDS}, got {self.kind!r}"
            )
        if self.ids not in APPLY_IDS:
            raise CompileError(
                f"apply ids must be one of {APPLY_IDS}, got {self.ids!r}"
            )
        if not self.function:
            raise CompileError("apply needs a registered function name")
        present = {field.alias for field in self.input.output_schema()}
        if not set(self.aliases) <= present:
            raise CompileError(
                f"apply {self.function!r} names tables {self.aliases} "
                f"outside its input ({sorted(present)})"
            )
        for ref in self.columns:
            if ref.alias not in self.aliases:
                raise CompileError(
                    f"apply {self.function!r} reads {ref.alias}.{ref.column} "
                    f"but works on {self.aliases}"
                )
        if (self.ids == "pairs") != (len(self.aliases) == 2):
            raise CompileError(
                "an apply returning pairs works on exactly two tables; "
                "one returning ids works on one"
            )

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
    """Select source columns and named model outputs."""

    input: LogicalNode
    columns: tuple[ColumnRef | Alias, ...]
    type_name: ClassVar[str] = "quail.logical_project"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return self.columns

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return tuple(column for column in self.columns if isinstance(column, ColumnRef))

    def validate(self) -> None:
        computed = {(node.call, node.name) for node in classifications(self.input)}
        for column in self.columns:
            if isinstance(column, Alias):
                column.validate()
                if (
                    column.expression.kind == "label"
                    and (column.expression, column.name) not in computed
                ):
                    raise CompileError(
                        f"the label column {column.name!r} needs a "
                        f"SemanticClassify below the Project"
                    )
            elif not isinstance(column, ColumnRef):
                raise CompileError(
                    f"a projected column is a column reference or a named "
                    f"expression, got {type(column).__name__}"
                )
        names = [
            f"{column.alias}.{column.column}"
            if isinstance(column, ColumnRef)
            else column.name
            for column in self.columns
        ]
        if len(names) != len(set(names)):
            raise CompileError(f"projection names must be unique, got {names}")

    def result_columns(self) -> tuple[str, ...]:
        """Return the selected column names."""
        return tuple(column_name(column) for column in self.columns)

    def with_children(self, children: tuple[LogicalNode, ...]) -> "Project":
        if len(children) != 1:
            raise CompileError("Project needs one child")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple) -> "Project":
        return replace(self, columns=expressions)

    def explain_fields(self) -> dict:
        return {
            "columns": [
                column_name(column)
                if isinstance(column, ColumnRef)
                else _explain(column)
                for column in self.columns
            ]
        }


@dataclass(frozen=True)
class Aggregate:
    """Group projected rows and compute aggregate outputs."""

    input: Project
    keys: tuple[str, ...]
    aggregates: tuple[AggregateCall, ...]
    output: tuple[str, ...]
    having: tuple[HavingTest, ...] = ()
    type_name: ClassVar[str] = "quail.logical_aggregate"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return (*self.keys, *self.aggregates, *self.having)

    def output_schema(self) -> tuple[ColumnRef, ...]:
        fields = {column_name(ref): ref for ref in self.input.output_schema()}
        return tuple(fields.get(name, ColumnRef("", "", name)) for name in self.output)

    def validate(self) -> None:
        if not isinstance(self.input, Project):
            raise CompileError("Aggregate needs projected input columns")
        self.specification().validate(self.input.result_columns())

    def specification(self) -> Aggregation:
        """Return the grouping expressions and requested outputs."""
        return Aggregation(self.keys, self.aggregates, self.output, self.having)

    def result_columns(self) -> tuple[str, ...]:
        """Return the group keys and aggregate names in output order."""
        return self.output

    def with_children(self, children: tuple[LogicalNode, ...]) -> "Aggregate":
        if len(children) != 1:
            raise CompileError("Aggregate needs one child")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple) -> "Aggregate":
        if len(expressions) != len(self.expressions()):
            raise CompileError("Aggregate expression count must stay the same")
        keys_end = len(self.keys)
        aggregates_end = keys_end + len(self.aggregates)
        return replace(
            self,
            keys=expressions[:keys_end],
            aggregates=expressions[keys_end:aggregates_end],
            having=expressions[aggregates_end:],
        )

    def explain_fields(self) -> dict:
        return {"aggregation": str(self.specification())}


@dataclass(frozen=True)
class Result:
    """Order result rows, remove duplicates, and apply an offset and limit."""

    input: LogicalNode
    limit: int | None = None
    order: tuple[SortKey, ...] = ()
    offset: int = 0
    distinct: bool = False
    type_name: ClassVar[str] = "quail.logical_result"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple[SortKey, ...]:
        return self.order

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return self.input.output_schema()

    def result_columns(self) -> tuple[str, ...]:
        """Return column names after projection or aggregation."""
        return self.input.result_columns()

    def validate(self) -> None:
        if not isinstance(self.input, (Project, Aggregate)):
            raise CompileError("Result needs a projection or aggregation")
        if self.limit is not None and self.limit <= 0:
            raise CompileError("LIMIT must be a positive integer")
        if self.offset < 0:
            raise CompileError("OFFSET must be a nonnegative integer")
        projected = set(self.result_columns())
        projection = (
            self.input.input if isinstance(self.input, Aggregate) else self.input
        )
        for key in self.order:
            if not isinstance(key, SortKey):
                raise CompileError(
                    f"an ORDER BY term is a SortKey, got {type(key).__name__}"
                )
            key.validate()
            if (
                isinstance(key.expression, Alias)
                and key.expression not in projection.columns
            ):
                raise CompileError(
                    f"ORDER BY {key.name!r} names an expression the "
                    f"projection does not return"
                )
            if isinstance(self.input, Aggregate) and key.name not in projected:
                raise CompileError(
                    f"ORDER BY {key.name!r} with GROUP BY names a key or "
                    f"an aggregate in the SELECT list"
                )
            if self.distinct and key.name not in projected:
                raise CompileError(
                    f"ORDER BY {key.name!r} with DISTINCT needs the "
                    f"column in the SELECT list"
                )

    def with_children(self, children: tuple[LogicalNode, ...]) -> "Result":
        if len(children) != 1:
            raise CompileError("Result needs one child")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple) -> "Result":
        return replace(self, order=expressions)

    def explain_fields(self) -> dict:
        return {
            "order": [str(key) for key in self.order],
            "limit": self.limit,
            "offset": self.offset,
            "distinct": self.distinct,
        }
