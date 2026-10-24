"""Logical plan traversal and operator inspection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pyarrow as pa
from pyarrow import compute as pc

from quail.logical.expressions import (
    PROBABILITIES_SUFFIX,
    SPAN_SUFFIX,
    Alias,
    ColumnRef,
    CompileError,
    ModelCall,
    RegularPredicate,
    aliased_call,
    column_name,
    column_references,
    evaluate_expression,
    model_call,
)
from quail.logical.nodes import (
    Aggregate,
    Apply,
    Filter,
    Join,
    LogicalNode,
    Project,
    Result,
    Scan,
    SemanticClassify,
    SemanticExtract,
    SemanticFilter,
    SemanticJoin,
    Window,
)


@dataclass(frozen=True)
class ColumnTest:
    """A Filter on a column a classification or an extraction computes.

    Attributes:
        name: The label column, or the extraction whose answer or
            span the Filter tests, as the SemanticClassify or
            SemanticExtract names it.
        call: The model call that computes the column.
        condition: The RegularPredicate the Filter applies.
        selectivity: The Filter's selectivity hint, or None.
        position: Its written position among the table's filters,
            counted after the table's AI.IF predicates, with the label
            filters before the answer filters.
    """

    name: str
    call: "ModelCall"
    condition: RegularPredicate
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
    projected = {
        column.expression: column.name
        for column in columns
        if aliased_call(column, "label") is not None
    }
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

    @property
    def projection(self) -> Project:
        """Return the projection that supplies result columns."""
        node = self.root
        while isinstance(node, (Result, Aggregate, Window)):
            node = node.input
        if not isinstance(node, Project):
            raise CompileError("a query plan needs a projection")
        return node

    @property
    def result(self) -> Result:
        """Return result modifiers, using defaults when they are absent."""
        return self.root if isinstance(self.root, Result) else Result(self.root)

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

    def schema(self) -> pa.Schema:
        """Return the Arrow schema of the result, without running the plan.

        The types follow the result the executor builds: a source column
        has its provider's type, a score is a float, a label is text, an
        extraction is a struct of answer fields beside a struct of their
        character spans, and aggregates and windows have the types of
        the Arrow kernels that compute them.

        Raises:
            CompileError: A Scan carries no provider schema.
        """
        scans = {node.alias: node for node in self.walk() if isinstance(node, Scan)}
        for scan in scans.values():
            if scan.schema is None:
                raise CompileError(
                    f"the Scan of {scan.alias!r} carries no schema; the result "
                    f"cannot be typed")
        fields = _projected_fields(self.projection, scans)
        for node in self.walk():
            if isinstance(node, Aggregate):
                fields = _aggregated_fields(node, fields)
            elif isinstance(node, Window):
                fields = _windowed_fields(node, fields)
        return pa.schema(fields)

    def operators(self) -> "Operators":
        """Return the plan's operators as the planner reads them."""
        scans, filters, joins, applies, classifies = [], {}, [], [], []
        extracts, tested, aliases = [], {}, []
        nodes = self.walk()
        for node in nodes:
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
            elif isinstance(node, SemanticExtract):
                extracts.append(node)
        calls = {
            (node.alias, node.name): node.call
            for node in classifies
            if len(node.call.aliases()) == 1
        }
        label_filters, extract_filters = {}, {}
        for alias in aliases:
            labels = [node for node in tested.get(alias, ())
                      if (alias, node.condition.column.column) in calls]
            if labels:
                label_filters[alias] = tuple(
                    ColumnTest(
                        name=node.condition.column.column,
                        call=calls[(alias, node.condition.column.column)],
                        condition=node.condition,
                        selectivity=node.selectivity,
                        position=len(filters.get(alias, ())) + index,
                    )
                    for index, node in enumerate(labels))
            answers = [node for node in tested.get(alias, ()) if node not in labels]
            if answers:
                extract_filters[alias] = tuple(
                    ColumnTest(
                        name=_extraction(node.condition.column.column, extracts).name,
                        call=_extraction(node.condition.column.column, extracts).call,
                        condition=node.condition,
                        selectivity=node.selectivity,
                        position=len(filters.get(alias, ())) + len(labels) + index,
                    )
                    for index, node in enumerate(answers))
        columns = next(
            (
                node.columns
                for node in reversed(nodes)
                if isinstance(node, Project)
            ),
            (),
        )
        return Operators(
            scans=tuple(scans),
            filters={
                alias: tuple(filters[alias]) for alias in aliases if alias in filters
            },
            label_filters=label_filters,
            extract_filters=extract_filters,
            joins=tuple(joins),
            applies=tuple(applies),
            projections=tuple(
                column for column in columns if aliased_call(column) is not None
            ),
            labels=_label_work(classifies, columns, label_filters),
            classifies=tuple(classifies),
            extracts=tuple(extracts),
        )


@dataclass(frozen=True)
class Operators:
    """The operators of one plan, in the order the plan was written.

    ``walk`` lists children before parents, so scans, joins, applies,
    classifications, and extractions follow written order, each
    alias's AI.IF predicates keep their order, and each alias's label
    filters keep theirs.
    """

    scans: tuple  # tuple[Scan, ...]
    filters: dict  # alias -> tuple[FilterPredicate, ...]
    joins: tuple  # tuple[SemanticJoin, ...]
    applies: tuple  # tuple[Apply, ...]
    labels: LabelWork
    projections: tuple = ()  # named model calls in SELECT order
    classifies: tuple = ()  # tuple[SemanticClassify, ...]
    extracts: tuple = ()  # tuple[SemanticExtract, ...]
    label_filters: dict = field(default_factory=dict)
    #                     alias -> tuple[ColumnTest, ...] on labels
    extract_filters: dict = field(default_factory=dict)
    #                     alias -> tuple[ColumnTest, ...] on answers

    def all_filters(self) -> dict:
        """Return alias -> its AI.IF predicates, then its ColumnTests."""
        aliases = list(dict.fromkeys(
            (*self.filters, *self.label_filters, *self.extract_filters)))
        return {
            alias: (*self.filters.get(alias, ()), *self.label_filters.get(alias, ()),
                    *self.extract_filters.get(alias, ()))
            for alias in aliases
        }

    @property
    def prompts(self) -> tuple:
        """Return filter, label, join, and remaining result-column prompts in order."""
        calls = (
            tuple(
                model_call(predicate.expression)
                for predicates in self.filters.values()
                for predicate in predicates
            )
            + tuple(
                test.call for tests in self.label_filters.values() for test in tests
            )
            + tuple(model_call(join.predicate) for join in self.joins)
        )
        return tuple(call.prompt for call in calls) + tuple(
            column.expression.prompt
            for column in self.projections
            if column.expression not in calls
        )


def _extraction(column: str, extracts):
    """Return the SemanticExtract computing an answer or span column."""
    return next(node for node in extracts if column in (node.name, node.span_name))


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


SPAN_TYPE = pa.struct([("start", pa.int32()), ("end", pa.int32())])
PROBABILITIES_TYPE = pa.map_(pa.string(), pa.float64())

# the Arrow kernel each aggregate runs, which fixes its output type
_AGGREGATE_KERNELS = {"sum": pc.sum, "avg": pc.mean, "min": pc.min, "max": pc.max}


def _projected_fields(project: Project, scans: dict) -> dict:
    """Return name -> Arrow type of each column a Project returns, packed."""
    types = {}
    for column in project.columns:
        name = column_name(column)
        call = aliased_call(column)
        if call is None:
            (ref, *_) = column_references(column)
            table = scans[ref.alias].schema.empty_table()
            types[name] = evaluate_expression(column, table).type
        elif call.kind == "score":
            types[name] = pa.float64()
        elif call.kind == "label":
            types[name] = pa.string()
            if call.probabilities:
                types[name + PROBABILITIES_SUFFIX] = PROBABILITIES_TYPE
        else:
            types[name] = pa.string()
            types[name + SPAN_SUFFIX] = SPAN_TYPE
    packing = dict(project.packing())
    owner = {member: struct for struct, members in packing.items()
             for _, member in members}
    packed = {}
    for name, value_type in types.items():
        struct = owner.get(name)
        if struct is None:
            packed[name] = value_type
        elif struct not in packed:
            packed[struct] = pa.struct([
                (field, types[member]) for field, member in packing[struct]])
    return packed


def _aggregated_fields(node: Aggregate, fields: dict) -> dict:
    """Return name -> Arrow type of an Aggregate's output columns."""
    types = dict(fields)
    for call in node.aggregates:
        if call.function in ("count", "count_distinct"):
            types[call.name] = pa.int64()
        else:
            empty = pa.array([], fields[call.argument])
            types[call.name] = _AGGREGATE_KERNELS[call.function](empty).type
    return {name: types[name] for name in node.output}


def _windowed_fields(node: Window, fields: dict) -> dict:
    """Return name -> Arrow type of a Window's output columns."""
    types = dict(fields)
    for call in node.calls:
        if call.function in ("row_number", "count"):
            types[call.name] = pa.int64()
        elif call.function in ("lag", "lead"):
            types[call.name] = fields[call.argument]
        else:
            empty = pa.array([], fields[call.argument])
            types[call.name] = pc.cumulative_sum(pc.fill_null(empty, 0)).type
    return {name: types[name] for name in node.output}


def cte_schema(plan: LogicalPlan) -> pa.Schema:
    """Return the schema of a plan's result, named as a CTE's columns."""
    return pa.schema([
        pa.field(name, field.type)
        for name, field in zip(cte_column_names(plan), plan.schema())])


def cte_column_names(plan: LogicalPlan) -> tuple[str, ...]:
    """Return the column names a CTE gives its result, as SQL names them.

    A projected column reference is named by its column alone, so
    ``SELECT c.id`` gives ``id``; an AS name, a struct column, an
    aggregate, and a window keep their names.

    Raises:
        CompileError: A projected expression has no AS name, or two
            columns would share one name.
    """
    expressions = {column_name(column): column
                   for column in plan.projection.columns}
    names = []
    for name in plan.root.result_columns():
        column = expressions.get(name)
        if isinstance(column, ColumnRef):
            name = column.column
        elif column is not None and not isinstance(column, Alias):
            raise CompileError(
                f"{name} in a CTE needs an AS name; a later query reads "
                f"the column by that name")
        names.append(name)
    if len(set(names)) != len(names):
        raise CompileError(
            f"the CTE returns a column name twice: {names}; give one an AS name")
    return tuple(names)


@dataclass(frozen=True)
class Cte:
    """A common table expression: a named query result defined with WITH.

    A subquery in FROM is a Cte too, named by its alias.

    Attributes:
        name: The name later queries read the result under.
        plan: The query that computes the result.
    """

    name: str
    plan: LogicalPlan


@dataclass(frozen=True)
class WithPlan:
    """A query with CTEs; each CTE runs before the queries that read it.

    Attributes:
        ctes: The CTEs in dependency order. Each reads registered tables
            and earlier CTEs only, and each is read by the query or by
            a later CTE.
        query: The final SELECT.
    """

    ctes: tuple[Cte, ...]
    query: LogicalPlan
