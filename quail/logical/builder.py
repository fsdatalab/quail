"""Construct logical plans from bound frontend expressions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

from quail.logical.expressions import (
    Aggregation,
    ColumnRef,
    CompileError,
    FilterPredicate,
    InList,
    ModelCall,
    Prompt,
    model_call,
)
from quail.logical.nodes import (
    Aggregate,
    Apply,
    Filter,
    Join,
    Project,
    Result,
    Scan,
    SemanticClassify,
    SemanticExtract,
    SemanticFilter,
    SemanticJoin,
)
from quail.logical.plan import (
    LogicalPlan,
)


@dataclass(frozen=True)
class JoinSpec:
    """The join predicate (or one EXISTS/anti term), pre-assembly."""

    aliases: tuple  # newly joined table aliases
    predicate: Any  # ModelCall | Compare
    semantics: str = "full"
    selectivity: Optional[float] = None
    anchor: Optional[str] = None
    on: tuple = ()  # tuple[Equality, ...] over the tables
    applies: tuple = ()  # (function, kind, columns) returning pairs

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
        regular_predicates: tuple = (),
        extracts: tuple = (),
    ) -> None:
        """Add one table with its filters, applies, classifications, and extractions.

        Args:
            alias: The table's alias in the query.
            provider: The registered provider name.
            column: The document column.
            predicates: The table's AI.IF predicates in written order.
            regular_predicates: RegularPredicate tests the scan applies
                before any operator reads a document.
            applies: (function, kind, ids, columns) per apply, in order.
            labels: (call, name) per one-table classification. A call
                that a label filter tests sits below the first Filter
                testing it; the others sit above the last Filter, in the
                order given.
            label_filters: (name, values, selectivity) per Filter on a
                classification's label column, in written order.
            extracts: (call, name) per extraction, in order. They sit
                above the classifications and label filters.

        Raises:
            CompileError: The alias is taken, or a label filter names a
                column no classification in labels computes.
        """
        if alias in self._nodes:
            raise CompileError(f"duplicate table alias {alias!r}")
        node = Scan(
            provider=provider,
            alias=alias,
            column=column,
            predicates=tuple(regular_predicates),
        )
        if predicates:
            node = SemanticFilter(node, tuple(predicates))
        for function, kind, ids, columns in applies:
            node = Apply(
                node,
                function=function,
                kind=kind,
                ids=ids,
                columns=tuple(columns),
                aliases=(alias,),
            )
        calls = {name: call for call, name in labels}
        classified = []
        for name, values, selectivity in label_filters:
            if name not in calls:
                raise CompileError(
                    f"a filter on a label of {alias!r} tests a "
                    f"classification the query does not name"
                )
            if calls[name] not in classified:
                node = SemanticClassify(node, calls[name], name)
                classified.append(calls[name])
            node = Filter(
                node,
                InList(ColumnRef(alias, provider, name), tuple(values)),
                selectivity,
            )
        for call, name in labels:
            if call not in classified:
                node = SemanticClassify(node, call, name)
                classified.append(call)
        for call, name in extracts:
            node = SemanticExtract(node, call, name)
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
            if (
                isinstance(node, SemanticJoin)
                and set(model_call(node.predicate).aliases()) == aliases
            ):
                return SemanticClassify(node, call, name)
            if isinstance(node, (SemanticJoin, SemanticClassify, Apply)):
                return replace(node, input=insert(node.input))
            if isinstance(node, Join):
                return replace(node, left=insert(node.left))
            raise CompileError(
                f"the classification of {call.aliases()[0]!r} x "
                f"{call.aliases()[1]!r} joined rows needs a join of the two"
            )

        if self._root is None or len(aliases) != 2:
            raise CompileError(
                "a classification of joined rows reads one document from "
                "each side of a join"
            )
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
                    f"the JOIN that introduces the table"
                )
        if join.on:
            root = replace(root, on=tuple(join.on))
        written_pos = self._joins
        self._joins += 1
        for function, kind, columns in join.applies:
            aliases = tuple(dict.fromkeys(ref.alias for ref in join.prompt.args))
            root = Apply(
                root,
                function=function,
                kind=kind,
                ids="pairs",
                columns=tuple(columns),
                aliases=aliases,
                written_pos=written_pos,
            )
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
        self,
        columns: tuple,
        limit: int | None = None,
        order: tuple = (),
        offset: int = 0,
        distinct: bool = False,
        aggregation: Aggregation | None = None,
    ) -> LogicalPlan:
        if self._root is None:
            raise CompileError("a logical plan needs an input table")
        if not columns and aggregation is None:
            raise CompileError("Project needs at least one column")
        node = Project(self._root, tuple(columns))
        if aggregation is not None:
            node = Aggregate(
                node,
                aggregation.keys,
                aggregation.aggregates,
                aggregation.output,
                aggregation.having,
            )
        if limit is not None or order or offset or distinct:
            node = Result(node, limit, tuple(order), offset, distinct)
        plan = LogicalPlan(node)
        plan.validate()
        return plan
