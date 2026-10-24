"""Construct logical plans from bound frontend expressions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

from quail.logical.expressions import (
    Aggregation,
    ColumnRef,
    CompileError,
    FilterPredicate,
    ModelCall,
    Prompt,
    RegularPredicate,
    aliased_call,
    column_name,
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
    SemanticSaeFeatures,
    Unnest,
    Window,
)
from quail.logical.plan import (
    LogicalPlan,
)


@dataclass(frozen=True)
class JoinSpec:
    """The join predicate (or one EXISTS/anti term), pre-assembly."""

    aliases: tuple  # newly joined table aliases
    predicate: Any  # ModelCall | RegularPredicate over an AI.SCORE call
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
        extract_filters: tuple = (),
        sae_features: tuple = (),
        schema=None,
    ) -> None:
        """Add one table with its filters, applies, classifications, and extractions.

        Args:
            alias: The table's alias in the query.
            provider: The registered provider name.
            column: The document column.
            schema: The provider's Arrow schema, for typing the result.
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
            extract_filters: (RegularPredicate, selectivity) per Filter
                on an extraction's answer or span column, in written
                order. Each sits right above the extraction it tests.
            sae_features: (call, name) per AI.SAE_FEATURES call, in
                order. They sit above the extractions.

        Raises:
            CompileError: The alias is taken, a label filter names a
                column no classification in labels computes, or an
                extraction filter names a column no extraction computes.
        """
        if alias in self._nodes:
            raise CompileError(f"duplicate table alias {alias!r}")
        node = Scan(
            provider=provider,
            alias=alias,
            column=column,
            predicates=tuple(regular_predicates),
            schema=schema,
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
            unknown = sorted(set(values) - set(calls[name].labels))
            if unknown:
                raise CompileError(
                    f"the filter on {name!r} lists {unknown}, which are not "
                    f"among its labels {list(calls[name].labels)}")
            if calls[name] not in classified:
                node = SemanticClassify(node, calls[name], name)
                classified.append(calls[name])
            node = Filter(
                node,
                RegularPredicate(ColumnRef(alias, provider, name), "in", tuple(values)),
                selectivity,
            )
        for call, name in labels:
            if call not in classified:
                node = SemanticClassify(node, call, name)
                classified.append(call)
        tests = list(extract_filters)
        for call, name in extracts:
            extraction = SemanticExtract(node, call, name)
            node = extraction
            for predicate, selectivity in extract_filters:
                if predicate.column.column in (name, extraction.span_name):
                    node = Filter(node, predicate, selectivity)
                    tests.remove((predicate, selectivity))
        if tests:
            raise CompileError(
                f"the filter {tests[0][0]} tests an extraction the query "
                f"does not return")
        for call, name in sae_features:
            node = SemanticSaeFeatures(node, call, name)
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
        written_position = self._joins
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
                written_position=written_position,
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
        structs: tuple = (),
        windows: tuple = (),
        output: tuple = (),
        unnests: tuple = (),
        unnest_tests: tuple = (),
        unnest_aliases: tuple = (),
    ) -> LogicalPlan:
        """Build the plan's result nodes over the projected columns.

        Args:
            columns: The projected value expressions and Aliases.
            limit: LIMIT, or None.
            order: The SortKeys of ORDER BY.
            offset: OFFSET.
            distinct: Whether the result is DISTINCT.
            aggregation: GROUP BY and its aggregates, or None.
            structs: The AI.EXTRACT struct columns the projection packs.
            windows: The WindowCalls over the projected or grouped rows.
            output: The result columns in SELECT order when windows
                or unnested lists are present.
            unnests: The list columns of each Unnest, lowest first; the
                lists of one Unnest are zipped.
            unnest_tests: RegularPredicates on unnested field columns.
                Each becomes a Filter above the Unnest that adds its
                column.
            unnest_aliases: The Aliases that name unnested field
                columns, each on the Unnest that adds its column.
        """
        if self._root is None:
            raise CompileError("a logical plan needs an input table")
        if not columns and aggregation is None:
            raise CompileError("Project needs at least one column")
        if structs and aggregation is not None:
            raise CompileError(
                "GROUP BY does not read AI.EXTRACT struct columns; extract "
                "each field as its own column")
        if structs and distinct:
            raise CompileError(
                "DISTINCT does not compare AI.EXTRACT struct columns; extract "
                "each field as its own column")
        unnested = {name for lists in unnests for name in lists}
        _check_feature_lists(columns, order, distinct, aggregation, unnested)
        node = _unnested(Project(self._root, tuple(columns), tuple(structs)),
                         unnests, unnest_tests, unnest_aliases)
        if aggregation is not None:
            node = Aggregate(
                node,
                aggregation.keys,
                aggregation.aggregates,
                aggregation.output,
                aggregation.having,
            )
        if windows:
            node = Window(node, tuple(windows), tuple(output))
        # over an Unnest, the result puts its columns in SELECT order
        selected = (tuple(output) if unnests and not windows
                    and aggregation is None else ())
        if limit is not None or order or offset or distinct or selected:
            node = Result(node, limit, tuple(order), offset, distinct, selected)
        plan = LogicalPlan(node)
        plan.validate()
        return plan


def _unnested(node, unnests, tests, aliases):
    """Return the node under its Unnests, each test above its field's Unnest.

    Each Alias goes on the Unnest that adds the column it names.

    Raises:
        CompileError: An Unnest names no list column below it, or a test
            or an Alias reads no unnested field.
    """
    tests, aliases = list(tests), list(aliases)
    for lists in unnests:
        node = Unnest(node, tuple(lists))
        node.validate()
        fields = {name for names in node.fields().values() for name in names}
        named = [alias for alias in aliases
                 if column_name(alias.expression) in fields]
        if named:
            node = replace(node, aliases=tuple(named))
            node.validate()
            aliases = [alias for alias in aliases if alias not in named]
        for test in [test for test in tests if column_name(test.column) in fields]:
            node = Filter(node, test)
            tests.remove(test)
    if tests:
        raise CompileError(
            f"{tests[0]} tests no field of an unnested list")
    if aliases:
        raise CompileError(
            f"{aliases[0].name!r} names no field of an unnested list")
    return node


def _check_feature_lists(columns, order, distinct, aggregation,
                         unnested) -> None:
    """Refuse an AI.SAE_FEATURES list where the query compares values.

    An unnested list is compared by its scalar fields, so it is exempt.

    Raises:
        CompileError: ORDER BY, GROUP BY, or DISTINCT reads a feature
            list column.
    """
    names = {column.name for column in columns
             if aliased_call(column, "features") is not None
             and column.name not in unnested}
    if not names:
        return
    if aggregation is not None or distinct:
        raise CompileError(
            "GROUP BY and DISTINCT do not compare AI.SAE_FEATURES list "
            "columns")
    for key in order:
        expression = key.expression
        if (getattr(expression, "name", None) in names
                or aliased_call(expression, "features") is not None
                or (isinstance(expression, ModelCall)
                    and expression.kind == "features")):
            raise CompileError(
                "ORDER BY does not compare AI.SAE_FEATURES list columns")
