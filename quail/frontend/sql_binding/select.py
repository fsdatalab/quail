"""Bind projections, grouping, and result ordering."""

from dataclasses import dataclass, field

from sqlglot import exp

from quail.frontend.sql_binding.expressions import (
    _FLIPPED,
    _PLAIN_COMPARISONS,
    ExpressionBinder,
    _conjuncts,
    _is_call,
)
from quail.logical import (
    AggregateCall,
    Aggregation,
    Alias,
    ColumnRef,
    CompileError,
    HavingTest,
    SortKey,
)


def _parse_limit(tree) -> int | None:
    """Extract a plain LIMIT N from the parse tree."""
    limit_node = tree.args.get("limit")
    if limit_node is None:
        return None
    expr = limit_node.expression
    if not isinstance(expr, exp.Literal) or expr.is_string:
        raise CompileError("LIMIT must be a positive integer")
    value = int(expr.this)
    if value <= 0:
        raise CompileError("LIMIT must be a positive integer")
    return value


def _parse_offset(tree) -> int:
    """Extract a plain OFFSET N from the parse tree; 0 when absent."""
    offset_node = tree.args.get("offset")
    if offset_node is None:
        return 0
    expr = offset_node.expression
    if not isinstance(expr, exp.Literal) or expr.is_string:
        raise CompileError("OFFSET must be a nonnegative integer")
    value = int(expr.this)
    if value < 0:
        raise CompileError("OFFSET must be a nonnegative integer")
    return value


def _parse_distinct(tree) -> bool:
    """Return whether the SELECT is DISTINCT; DISTINCT ON is refused."""
    distinct = tree.args.get("distinct")
    if distinct is None:
        return False
    if distinct.args.get("on") is not None:
        raise CompileError("DISTINCT ON is not supported; use DISTINCT")
    return True


def bind_order(
    tree: exp.Select,
    b: ExpressionBinder,
    columns: tuple[ColumnRef | Alias, ...],
    aggregation: Aggregation | None = None,
) -> tuple[SortKey, ...]:
    """Bind each ORDER BY term to a projected name or a source column."""
    order = tree.args.get("order")
    if order is None:
        return ()
    named = {column.name: column for column in columns if isinstance(column, Alias)}
    if aggregation is not None:
        named.update(
            {aggregate.name: aggregate for aggregate in aggregation.aggregates}
        )
    keys = []
    for term in order.expressions:
        target = term.this
        if not isinstance(target, exp.Column):
            raise CompileError(
                f"ORDER BY {target.sql()} is not a column; give an "
                f"expression an AS name in SELECT and order by that name"
            )
        if not target.table and target.name in named:
            expression = named[target.name]
        else:
            expression = b.resolve_column(target)
        keys.append(
            SortKey(
                expression,
                descending=bool(term.args.get("desc")),
                nulls_first=bool(term.args.get("nulls_first")),
            )
        )
    return tuple(keys)


_AGGREGATE_CLASSES = {
    exp.Count: "count",
    exp.Sum: "sum",
    exp.Avg: "avg",
    exp.Min: "min",
    exp.Max: "max",
}


def _column_name(column) -> str:
    return (
        f"{column.alias}.{column.column}"
        if isinstance(column, ColumnRef)
        else column.name
    )


@dataclass(frozen=True)
class BoundSelect:
    """Columns needed before aggregation and the requested aggregate outputs."""

    columns: tuple[ColumnRef | Alias, ...]
    aggregation: Aggregation | None


@dataclass
class SelectBinder:
    """Resolve SELECT expressions and own their intermediate columns."""

    binder: ExpressionBinder
    columns: list[ColumnRef | Alias] = field(default_factory=list)
    plain: list[str] = field(default_factory=list)
    output: list[str] = field(default_factory=list)
    aggregates: list[AggregateCall] = field(default_factory=list)

    def register_argument(self, node: exp.Expression, hidden: str) -> str:
        """Bind an aggregate's or key's expression to a projected column name.

        A column or AI expression not yet projected is appended to
        ``columns``, an AI expression under the ``hidden`` name.
        """
        b, columns = self.binder, self.columns
        names = {column.name: column for column in columns if isinstance(column, Alias)}
        if isinstance(node, exp.Column):
            if not node.table and node.name in names:
                return node.name
            ref = b.resolve_column(node)
            if ref not in columns:
                columns.append(ref)
            return _column_name(ref)
        if not (_is_call(node, "AI_SCORE") or isinstance(node, exp.AIClassify)):
            raise CompileError(f"{node.sql()} is not a column or an AI expression")
        call = b.bind_ai_value(node)
        for name, column in names.items():
            if column.expression == call:
                return name
        columns.append(Alias(call, hidden))
        return hidden

    def bind_aggregate(
        self, node: exp.Expression, name: str, hidden: str
    ) -> AggregateCall:
        """Bind one COUNT, SUM, AVG, MIN, or MAX call."""
        function = _AGGREGATE_CLASSES[type(node)]
        argument = node.this
        if isinstance(argument, exp.Star):
            if function != "count":
                raise CompileError(f"{node.sql()} needs a column")
            return AggregateCall("count", None, name)
        if isinstance(argument, exp.Distinct):
            if function != "count" or len(argument.expressions) != 1:
                raise CompileError(
                    "DISTINCT inside an aggregate is COUNT(DISTINCT column)"
                )
            function, argument = "count_distinct", argument.expressions[0]
        return AggregateCall(function, self.register_argument(argument, hidden), name)

    def bind_projection(self, tree: exp.Select) -> None:
        """Bind SELECT outputs and collect aggregate inputs."""
        b, columns = self.binder, self.columns
        for index, e in enumerate(tree.expressions):
            node = e.this if isinstance(e, exp.Alias) else e
            if type(node) in _AGGREGATE_CLASSES:
                if not isinstance(e, exp.Alias) or not e.alias:
                    raise CompileError(f"{node.sql()} needs an AS name in SELECT")
                aggregate = self.bind_aggregate(node, e.alias, f"__agg_{index}")
                self.aggregates.append(aggregate)
                self.output.append(aggregate.name)
                continue
            if any(type(call) in _AGGREGATE_CLASSES for call in node.walk()):
                raise CompileError(
                    "an aggregate in SELECT is a direct COUNT, SUM, AVG, MIN, "
                    "or MAX call with an AS name"
                )
            selected = _compile_projection(b, [e])
            for column in selected:
                if column not in columns:
                    columns.append(column)
            self.plain.extend(_column_name(column) for column in selected)
            self.output.extend(_column_name(column) for column in selected)

    def bind_group(self, tree: exp.Select) -> list[str]:
        """Resolve GROUP BY expressions and SELECT positions."""
        group = tree.args.get("group")
        keys = []
        for index, node in enumerate(group.expressions if group else ()):
            if isinstance(node, exp.Literal) and not node.is_string:
                position = int(node.this)
                if not 1 <= position <= len(self.output):
                    raise CompileError(f"GROUP BY {position} is out of range")
                keys.append(self.output[position - 1])
                continue
            keys.append(self.register_argument(node, f"__key_{index}"))
        return keys

    def bind_having(self, tree: exp.Select) -> list[HavingTest]:
        """Bind HAVING conditions and register any additional aggregates."""
        having = tree.args.get("having")
        tests = []
        for index, term in enumerate(_conjuncts(having.this) if having else []):
            comparison = _PLAIN_COMPARISONS.get(type(term))
            if comparison is None:
                raise CompileError(
                    f"HAVING compares an aggregate with a number, got {term.sql()}"
                )
            left, right = term.this, term.expression
            if not isinstance(right, (exp.Literal, exp.Neg)):
                left, right, comparison = right, left, _FLIPPED[comparison]
            if type(left) in _AGGREGATE_CLASSES:
                candidate = self.bind_aggregate(
                    left, f"__having_{index}", f"__agg_having_{index}"
                )
                same = next(
                    (
                        a
                        for a in self.aggregates
                        if (a.function, a.argument)
                        == (candidate.function, candidate.argument)
                    ),
                    None,
                )
                if same is None:
                    self.aggregates.append(candidate)
                    same = candidate
            elif isinstance(left, exp.Column) and not left.table:
                same = next((a for a in self.aggregates if a.name == left.name), None)
                if same is None:
                    raise CompileError(
                        f"HAVING {left.name} is not an aggregate of this SELECT"
                    )
            else:
                raise CompileError(f"HAVING tests an aggregate, got {term.sql()}")
            tests.append(HavingTest(same, comparison, _number(right)))
        return tests

    def bind(self, tree: exp.Select) -> BoundSelect:
        """Bind SELECT and its grouping clauses."""
        self.bind_projection(tree)
        keys = self.bind_group(tree)
        tests = self.bind_having(tree)
        if not self.aggregates and not keys and not tests:
            if len(self.output) != len(set(self.output)):
                raise CompileError(
                    f"projection names must be unique, got {self.output}")
            return BoundSelect(tuple(self.columns), None)
        for name in self.plain:
            if name not in keys:
                raise CompileError(
                    f"{name} is in SELECT but not in GROUP BY and not aggregated"
                )
        return BoundSelect(
            tuple(self.columns),
            Aggregation(
                tuple(keys), tuple(self.aggregates), tuple(self.output), tuple(tests)
            ),
        )


def _number(node):
    """Return the number a literal names, or raise."""
    if isinstance(node, exp.Neg):
        return -_number(node.this)
    if isinstance(node, exp.Literal) and not node.is_string:
        text = str(node.this)
        return float(text) if "." in text or "e" in text.lower() else int(text)
    raise CompileError(f"HAVING compares with a number, got {node.sql()}")


def _compile_projection(b: ExpressionBinder, expressions) -> list:
    columns = []
    for e in expressions:
        if isinstance(e, exp.Star):
            for alias, provider in b.tables:
                for c in b.catalog.get(provider).columns:
                    columns.append(ColumnRef(alias=alias, provider=provider, column=c))
            continue
        alias = None
        if isinstance(e, exp.Alias):
            alias = e.alias
            e = e.this
        if _is_call(e, "AI_SCORE"):
            if not alias:
                raise CompileError("AI.SCORE needs an AS name in SELECT")
            call = b.bind_ai_value(e)
            if len(call.aliases()) not in {1, 2}:
                raise CompileError(
                    "projected AI.SCORE must reference one or two relations"
                )
            columns.append(Alias(call, alias))
            continue
        if isinstance(e, exp.AIClassify):
            if not alias:
                raise CompileError("AI.CLASSIFY needs an AS name in SELECT")
            call = b.bind_ai_value(e)
            columns.append(Alias(call, alias))
            continue
        if any(
            _is_call(call, "AI_SCORE") or isinstance(call, exp.AIClassify)
            for call in e.walk()
        ):
            raise CompileError(
                "AI.SCORE and AI.CLASSIFY in SELECT must be direct "
                "expressions with an AS name"
            )
        if not isinstance(e, exp.Column):
            raise CompileError(
                f"the SELECT list is column selection only, got "
                f"{e.sql()}: nothing computed, per the projection "
                f"contract"
            )
        columns.append(b.resolve_column(e))
    return columns
