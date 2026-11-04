"""Construct logical plans from bound frontend expressions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

from quail.common.expressions import (
    SPAN_SUFFIX,
    Aggregation,
    CompileError,
    FilterPredicate,
    ModelCall,
    Prompt,
    aliased_call,
    column_name,
    column_references,
    condition_columns,
    model_call,
)
from quail.logical.nodes import (
    Aggregate,
    Filter,
    Join,
    Project,
    Result,
    Scan,
    SemanticClassify,
    SemanticExtract,
    SemanticFilter,
    SemanticJoin,
    SemanticMap,
    SemanticSaeFeatures,
    Unnest,
    Window,
)
from quail.logical.nodes.relational import readable_columns
from quail.logical.placement import result_columns_read, result_row_condition
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

    def add_scan(
        self,
        alias: str,
        provider: str,
        column: str,
        predicates: tuple[FilterPredicate, ...] = (),
        labels: tuple = (),
        regular_predicates: tuple = (),
        extracts: tuple = (),
        computed_filters: tuple = (),
        sae_features: tuple = (),
        maps: tuple = (),
        schema=None,
    ) -> None:
        """Add one table with its filters, classifications, and extractions.

        Args:
            alias: The table's alias in the query.
            provider: The registered provider name.
            column: The document column.
            schema: The provider's Arrow schema, for typing the result.
            predicates: The table's AI.IF predicates in written order.
            regular_predicates: The conditions the scan applies before
                any operator reads a document.
            labels: (call, name) per one-table classification. A call
                that a label filter tests sits below the first Filter
                testing it; the others sit above the last Filter, in the
                order given.
            extracts: (call, name) per extraction, in order. They sit
                above the classifications and label filters.
            computed_filters: (condition, selectivity) per Filter on the
                columns of column operators, in written order: a
                classification's label, an extraction's answer and span,
                or a feature call's or a map's answer. A label filter
                sits above its classification, in written order; any
                other filter sits right above the last operator it reads.
            sae_features: (call, name) per AI.SAE_FEATURES call, in
                order. They sit above the extractions.
            maps: (call, name) per AI.MAP call, in order. They sit
                above the feature calls.

        Raises:
            CompileError: The alias is taken, a label filter lists a
                value that is not a label, or a filter names a column no
                classification or extraction computes.
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
        calls = {name: call for call, name in labels}
        label_filters = [(condition, selectivity)
                         for condition, selectivity in computed_filters
                         if set(condition_columns(condition)) & set(calls)]
        classified = []
        for condition, selectivity in label_filters:
            for name in condition_columns(condition):
                if name in calls and calls[name] not in classified:
                    node = SemanticClassify(node, calls[name], name)
                    classified.append(calls[name])
            _check_labels(condition, calls)
            node = Filter(node, condition, selectivity)
        for call, name in labels:
            if call not in classified:
                node = SemanticClassify(node, call, name)
                classified.append(call)
        tests = [test for test in computed_filters if test not in label_filters]
        answers = {column for calls_of_kind in (extracts, sae_features, maps)
                   for _, name in calls_of_kind
                   for column in (name, name + SPAN_SUFFIX)}
        done = set()
        for node_type, calls_of_kind in ((SemanticExtract, extracts),
                                         (SemanticSaeFeatures, sae_features),
                                         (SemanticMap, maps)):
            for call, name in calls_of_kind:
                column = node_type(node, call, name)
                node = column
                done.update(column.added_columns)
                # a test runs after the last column operator it reads
                for predicate, selectivity in list(tests):
                    if set(condition_columns(predicate)) & answers <= done:
                        node = Filter(node, predicate, selectivity)
                        tests.remove((predicate, selectivity))
        if tests:
            raise CompileError(
                f"the filter {tests[0][0]} tests a column the query does not "
                f"return")
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
            if isinstance(node, (SemanticJoin, SemanticClassify)):
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
        row_tests: tuple = (),
        unnest_outer: tuple = (),
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
            row_tests: The conditions on result rows, such as one over
                two tables or over an unnested field. Each becomes a
                Filter above the Project, or above the Unnest that adds
                the last field it reads; the Project also returns the
                source columns they read, and a later Project drops them.
            unnest_outer: For each Unnest, whether a row with no
                elements returns one row of nulls; empty means none do.
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
        expanded = {column_name(column): column for column in columns
                    if any(ref.alias in unnested
                           for ref in column_references(column))}
        source = tuple(column for column in columns
                       if column_name(column) not in expanded)
        written = Project(self._root, source, tuple(structs))
        returned = written.result_columns()
        row_tests = tuple(result_row_condition(test, written.packing())
                          for test in row_tests)
        if (unnests or row_tests) and aggregation is None:
            source += tuple(key.expression for key in order
                            if column_references(key.expression)
                            and not any(ref.alias in unnested for ref in
                                        column_references(key.expression))
                            and key.expression not in source)
        source += _row_test_columns(row_tests, unnested, source, returned)
        node = _unnested(Project(self._root, source, tuple(structs)),
                         unnests, row_tests,
                         unnest_outer or (False,) * len(unnests))
        if unnests or row_tests:
            if aggregation is not None or windows:
                selected = tuple(column_name(column) for column in columns
                                 if column_name(column) not in unnested)
            else:
                selected = tuple(output) if unnests else returned
            node = Project(node, tuple(expanded.get(name, name) for name in selected))
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
        if limit is not None or order or offset or distinct:
            node = Result(node, limit, tuple(order), offset, distinct)
        plan = LogicalPlan(node)
        plan.validate()
        return plan


def _check_labels(condition, calls) -> None:
    """Refuse a label filter that lists a value its classification cannot return.

    Raises:
        CompileError: An = or IN test of a label lists a value that is
            not among the labels.
    """
    for test in condition.tests():
        references = column_references(test.expression)
        if test.comparison not in ("=", "in") or len(references) != 1:
            continue
        name = references[0].column
        if name not in calls:
            continue
        values = test.value if isinstance(test.value, tuple) else (test.value,)
        unknown = sorted(set(values) - set(calls[name].labels))
        if unknown:
            raise CompileError(
                f"the filter on {name!r} lists {unknown}, which are not "
                f"among its labels {list(calls[name].labels)}")


def _row_test_columns(tests, unnested, source, returned) -> tuple:
    """Return the source columns result row tests read that the Project lacks."""
    present = set(returned) | {column_name(column) for column in source}
    extra = []
    for test in tests:
        for reference in (reference for found in test.tests()
                          for reference in column_references(found.expression)):
            if (reference.provider and reference.alias not in unnested
                    and column_name(reference) not in present):
                extra.append(reference)
                present.add(column_name(reference))
    return tuple(extra)


def _unnested(node, unnests, tests, outer):
    """Return the node under its Unnests and result row Filters.

    Each test sits above the Project, or above the Unnest that adds the
    last column it reads.

    Raises:
        CompileError: An Unnest names no list column below it, or a test
            reads a column no result row holds.
    """
    tests = list(tests)

    def place(node):
        available = readable_columns(node)
        for test in [test for test in tests
                     if set(result_columns_read(test)) <= available]:
            node = Filter(node, test)
            tests.remove(test)
        return node

    node = place(node)
    for lists, keep_empty in zip(unnests, outer, strict=True):
        node = Unnest(node, tuple(lists), outer=keep_empty)
        node.validate()
        node = place(node)
    if tests:
        missing = sorted(set(result_columns_read(tests[0]))
                         - readable_columns(node))
        raise CompileError(
            f"{tests[0]} reads {missing}, which no result row holds")
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
