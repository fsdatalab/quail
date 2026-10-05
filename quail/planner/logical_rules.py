"""Built in logical optimizer rules."""

from __future__ import annotations

from dataclasses import replace

from quail.cost import budgets
from quail.cost.work import Work
from quail.logical import (
    Aggregate,
    Alias,
    Apply,
    ColumnPredicate,
    ColumnRef,
    Compare,
    Equality,
    Filter,
    FilterPredicate,
    InList,
    Join,
    LogicalNode,
    LogicalPlan,
    ModelCall,
    Operators,
    Project,
    Result,
    Scan,
    SemanticClassify,
    SemanticFilter,
    SemanticJoin,
    SortKey,
    has_score,
    model_call,
)
from quail.planner import joins as joinsearch
from quail.planner import pricing
from quail.planner.filters import default_order_rule, order_filters_indexed
from quail.planner.statistics import (
    PlanStatistics,
    cached_statistics,
    filter_orders,
    filter_works,
    live_after_filters,
    undecided,
)


def _column_refs(expression) -> tuple[ColumnRef, ...]:
    """Return the column references one logical expression reads."""
    if isinstance(expression, ColumnRef):
        return (expression,)
    if isinstance(expression, FilterPredicate):
        return _column_refs(expression.expression)
    if isinstance(expression, (ModelCall, Compare, Alias)):
        return tuple(model_call(expression).prompt.args)
    if isinstance(expression, (InList, ColumnPredicate)):
        return (expression.column,)
    if isinstance(expression, Equality):
        return (expression.left, expression.right)
    if isinstance(expression, SortKey):
        return _column_refs(expression.expression)
    return ()


def push_down_projection(root: LogicalNode) -> LogicalNode:
    """Rewrite every Scan to keep only the column values the plan reads.

    One pass from the root toward the scans. Each node on the way adds
    the columns its expressions read to the set carried down, in first
    use order; a Scan keeps the columns collected for its alias. Every
    node that reads an alias's column is an ancestor of that alias's
    Scan, so the path holds the complete set.

    The document column is always tokenized. It is kept as a value too
    only when the root projection returns it.
    """
    plan = LogicalPlan(root)
    projection = next((node for node in reversed(plan.walk())
                       if isinstance(node, Project)), None)
    result = plan.result
    returned = {
        (ref.alias, ref.column)
        for ref in (projection.output_schema() if projection is not None else ())
    } | {
        (ref.alias, ref.column)
        for key in result.order
        for ref in _column_refs(key)
    }
    score_inputs = {
        (ref.alias, ref.column)
        for expression in (projection.columns if projection is not None else ())
        if isinstance(expression, Alias)
        for ref in _column_refs(expression)
    }

    def descend(node, needed):
        needed = {alias: dict(columns) for alias, columns in needed.items()}
        expressions = node.expressions()
        for expression in expressions:
            for ref in _column_refs(expression):
                needed.setdefault(ref.alias, {})[ref.column] = None
        if isinstance(node, SemanticClassify):
            # the label columns a classification adds are not read below it
            for field in node.output_schema()[len(node.input.output_schema()):]:
                needed.get(field.alias, {}).pop(field.column, None)
        if isinstance(node, Scan):
            columns = tuple(
                column for column in needed.get(node.alias, ())
                if column != node.column
                or (node.alias, column) in returned
                or (node.alias, column) in score_inputs
            )
            return node if columns == node.columns else replace(
                node, columns=columns)
        children = tuple(descend(child, needed) for child in node.children())
        if children == node.children():
            return node
        return node.with_children(children)

    return descend(root, {})


class ProjectionPushdown:
    """Push the projected column set down to each Scan."""

    name = "projection_pushdown"

    def rewrite(self, root, context):
        rewritten = push_down_projection(root)
        return None if rewritten is root else rewritten


def _movable_filter_alias(node) -> str | None:
    """Return the table a cheap one-table filter reads, or None for other nodes.

    A cheap filter is an Apply returning ids, or a Filter on a label that
    a SemanticClassify below it computed. An AI.IF filter runs the model
    and is not moved.
    """
    if isinstance(node, Apply) and node.ids != "pairs":
        return node.aliases[0]
    if isinstance(node, Filter):
        return node.condition.column.alias
    return None


def _lets_through(node, alias: str) -> bool:
    """Return whether a filter on one table may move below node.

    True for a Join, a full SemanticJoin, a classification of joined
    rows, and any node on another table. False for the table's own
    chain, a gate (an exists or anti join), an Apply returning a join's
    pairs, and any other node.
    """
    if isinstance(node, Join):
        return True
    if isinstance(node, SemanticJoin):
        return node.semantics == "full"
    if isinstance(node, SemanticClassify):
        return len(node.call.aliases()) == 2 or node.alias != alias
    if isinstance(node, Apply):
        return node.ids != "pairs" and node.aliases[0] != alias
    if isinstance(node, Filter):
        return node.condition.column.alias != alias
    if isinstance(node, SemanticFilter):
        return all(model_call(p.expression).aliases()[0] != alias
                   for p in node.predicates)
    return False


def _place_filter(filter_node, node, alias: str):
    """Put a one-table filter below the joins above its table.

    Returns:
        The subtree under node with the filter as low as it goes, or
        None when a gate, an Apply returning pairs, or the table's own
        chain stops it before any Join.
    """
    if isinstance(node, Join):
        left = {field.alias for field in node.left.output_schema()}
        side = "left" if alias in left else "right"
        child = getattr(node, side)
        placed = _place_filter(filter_node, child, alias)
        if placed is None:
            placed = filter_node.with_children((child,))
        return replace(node, **{side: placed})
    if _lets_through(node, alias):
        (child,) = node.children()
        placed = _place_filter(filter_node, child, alias)
        return None if placed is None else node.with_children((placed,))
    return None


def push_down_filters(root: LogicalNode) -> LogicalNode | None:
    """Move every cheap one-table filter below the joins above its table.

    An Apply returning ids and a Filter on a label column read one
    table, so they commute with the joins, the classifications of
    joined rows, and the other tables' operators above that table.
    Each moves to the top of its own table's chain and keeps its order
    with the table's other filters. A filter stays where it is when a
    gate (an exists or anti join) or an Apply returning a join's pairs
    sits between it and the first Join below.

    Returns:
        The rewritten root, or None when no filter moved.
    """
    moved = False

    def visit(node):
        nonlocal moved
        children = tuple(visit(child) for child in node.children())
        if children != node.children():
            node = node.with_children(children)
        alias = _movable_filter_alias(node)
        if alias is None:
            return node
        placed = _place_filter(node, node.input, alias)
        if placed is None:
            return node
        moved = True
        return placed

    rewritten = visit(root)
    return rewritten if moved else None


class FilterPushdown:
    """Push cheap one-table filters below the joins above their table.

    Like Catalyst's PushDownPredicates, the move is unconditional: an
    Apply returning ids and a Filter on a label column run no model, so
    they thin a table before the joins read it. Both front ends already
    place such filters on their table; the rule rewrites plans built
    another way.
    """

    name = "filter_pushdown"

    def rewrite(self, root, context):
        return push_down_filters(root)


def _label_chain(node) -> tuple:
    """Split a table's chain at its classifications.

    Returns:
        The node below the table's lowest SemanticClassify, and the
        SemanticClassify and label filter nodes above it, lowest first.
    """
    lifted = []
    while isinstance(node, (SemanticClassify, Filter)):
        lifted.append(node)
        node = node.input
    return node, tuple(reversed(lifted))


def _chain_alias(node) -> str:
    """Return the table alias a classification or label filter node reads."""
    if isinstance(node, SemanticClassify):
        return node.alias
    return node.condition.column.alias


def lift_classifications(root: LogicalNode) -> LogicalNode | None:
    """Move every joined table's classifications above the joins.

    Each one-table SemanticClassify of a table that a SemanticJoin
    reads, with the Filters on its label column, leaves the table's
    chain and moves above the topmost join, under the root Project, in
    scan order. A classification of a table that no join reads stays.

    Returns:
        The rewritten root, or None when no classification moved.
    """
    if isinstance(root, (Result, Aggregate)):
        rewritten = lift_classifications(root.input)
        return None if rewritten is None else root.with_children((rewritten,))
    joined = {alias for node in LogicalPlan(root).walk()
              if isinstance(node, SemanticJoin)
              for alias in model_call(node.predicate).aliases()}
    chains = {}

    def cut(node):
        base, chain = _label_chain(node)
        if chain and _chain_alias(chain[0]) in joined:
            chains.setdefault(_chain_alias(chain[0]), []).extend(chain)
            return base
        return node

    def visit(node):
        if isinstance(node, Join):
            return replace(node, left=visit(node.left), right=cut(node.right))
        if isinstance(node, (SemanticJoin, Apply)) or (
                isinstance(node, SemanticClassify)
                and len(node.call.aliases()) == 2):
            return node.with_children((visit(node.input),))
        return cut(node)

    top, lifted = _label_chain(root.input)
    for node in lifted:
        chains.setdefault(_chain_alias(node), []).append(node)
    spine = visit(top)
    if not chains:
        return None
    for scan in LogicalPlan(root).walk():
        if isinstance(scan, Scan):
            for node in chains.get(scan.alias, ()):
                spine = node.with_children((spine,))
    lifted = root.with_children((spine,))
    return None if lifted == root else lifted


class ClassifyPlacement:
    """Classify a joined table before its joins or after them, by cost.

    Before the joins, a classification labels every document its
    AI.IF filters kept, and a filter on its label thins the join's
    input. After the joins, it labels only the documents the joins
    matched. The rule keeps the plan as written unless the plan with
    every joined table's classifications above the joins
    (lift_classifications) costs less.

    The cost is the estimated seconds of the candidate's physical plan
    with the label_scoring rule applied (quail.planner.pricing). The
    rule does nothing when prices() is false for the context.
    """

    name = "classify_placement"

    def __init__(self, cost=None):
        """Make the rule with a cost function.

        Args:
            cost: Callable mapping a LogicalPlan and the context to the
                plan's estimated seconds, or None when the plan is
                refused. The default prices build_physical_plan's plan.
        """
        self.cost = pricing.estimated_seconds if cost is None else cost

    def rewrite(self, root, context):
        if context is not None and not prices(root, context):
            return None
        lifted = lift_classifications(root)
        if lifted is None:
            return None
        before = self.cost(LogicalPlan(root), context)
        after = self.cost(LogicalPlan(lifted), context)
        if before is None or after is None or not after < before:
            return None
        return lifted


def prices(root, context) -> bool:
    """Return whether the Quail cost model applies to a plan and context.

    It applies when the context has a model, the backend is quail, the
    model's weights fit one GPU, and the plan has no AI.SCORE, which the
    reranker planner plans.
    """
    return (context.model is not None and context.backend == "quail"
            and budgets.minimum_weight_gpus(context.model, context.device) == 1
            and not has_score(LogicalPlan(root)))


def _statistics(root, context) -> PlanStatistics:
    """Return the plan's statistics, shared through the context's memo."""
    return cached_statistics(
        LogicalPlan(root), context.memo, model=context.model,
        device=context.device, doc_tokens=context.document_tokens,
        pair_fractions=context.pair_fractions,
        scan_fractions=context.scan_fractions)


def _order_rule(context) -> str:
    return context.order or default_order_rule({}, ())[0]


class FilterOrder:
    """Order each table's AI.IF predicates by expected cost.

    Each SemanticFilter's predicates get the order that minimizes the
    chain's ideal expected time (order_filters_indexed): the first
    predicate scans every document's prefix, each later one asks over
    the KV the chain keeps, and a predicate's selectivity thins what
    follows. The rule records the order on the node when it differs
    from the written order. With order="as_written", it clears every
    recorded order. Both front ends put a table's AI.IF predicates in
    one SemanticFilter, so the order covers the whole chain.
    """

    name = "filter_order"

    def rewrite(self, root, context):
        if not prices(root, context):
            return None
        rule = _order_rule(context)
        statistics = _statistics(root, context)
        model, device = context.model, context.device

        def visit(node):
            children = tuple(visit(child) for child in node.children())
            if children != node.children():
                node = node.with_children(children)
            if not isinstance(node, SemanticFilter):
                return node
            if rule == "as_written":
                return node if not node.order else replace(node, order=())
            (alias,) = model_call(node.predicates[0].expression).aliases()
            ordered = order_filters_indexed(
                list(node.predicates), rule,
                prefix_tokens=(statistics.pre
                               + statistics.stats[alias].mean_doc_tokens),
                model=model, device=device, chunk_tokens=statistics.chunk)
            order = () if ordered == list(range(len(ordered))) else tuple(ordered)
            return node if order == node.order else replace(node, order=order)

        rewritten = visit(root)
        return None if rewritten == root else rewritten


class JoinOrder:
    """Choose the joins' stage order and each stage's anchor by cost.

    The left-deep search (quail.planner.joins.search_joins) prices
    every connected stage order with every anchor choice together,
    because a stage's cost depends on which table's KV is computed once
    and which documents' KV earlier stages left resident. It ranks each
    candidate by the whole query's estimated seconds, including the
    filter chains in their decided order. A written anchor is kept.
    With order="as_written", only the anchors are chosen. When no
    connected left-deep order exists, the written order is priced. The
    rule records each join's execution position and anchor on its
    SemanticJoin.
    """

    name = "join_order"

    def rewrite(self, root, context):
        if not prices(root, context):
            return None
        plan = LogicalPlan(root)
        joins = plan.operators().joins
        if not joins:
            return None
        found = self._search(plan, context)
        decided = {position: anchor for position, anchor in found["seq"]}
        exec_idx = {position: index
                    for index, (position, _) in enumerate(found["seq"])}
        position = 0

        def visit(node):
            nonlocal position
            children = tuple(visit(child) for child in node.children())
            if children != node.children():
                node = node.with_children(children)
            if isinstance(node, SemanticJoin):
                stage = (exec_idx[position], decided[position])
                position += 1
                if (node.exec_idx, node.exec_anchor) != stage:
                    node = replace(node, exec_idx=stage[0], exec_anchor=stage[1])
            return node

        rewritten = visit(root)
        if rewritten == root:
            return None
        return rewritten

    def _search(self, plan, context) -> dict:
        """Run the search once per plan and filter order; return its result."""
        key = ("join_order", undecided(plan.root),
               tuple(sorted((alias, tuple(order)) for alias, order
                            in filter_orders(plan).items())))
        if key in context.memo:
            return context.memo[key]
        statistics = _statistics(plan.root, context)
        model, device = context.model, context.device
        base_work = sum(filter_works(plan, statistics, model).values(), Work())
        live = live_after_filters(plan, statistics)
        filtered = set(plan.operators().all_filters())
        fixed = _order_rule(context) == "as_written"

        found = joinsearch.search_joins(
            statistics.specs, live, statistics.lengths, filtered,
            statistics.pre, statistics.chunk, model, device,
            base_work=base_work, fixed_order=fixed)
        if found is None:
            found = joinsearch.search_joins(
                statistics.specs, live, statistics.lengths, filtered,
                statistics.pre, statistics.chunk, model, device,
                base_work=base_work, fixed_order=True)
        context.memo[key] = found
        return found


class DistinctElimination:
    """Clear a DISTINCT that cannot remove rows.

    A filter returns each document at most once and a join pairs each
    row pair once, so the result rows are unique when the projection
    returns the id column of every scanned table. The rule reads the id
    columns from the catalog and leaves a plan alone without one, or
    when an apply() supplies the rows. A GROUP BY whose output returns
    every key is unique as well.
    """

    name = "distinct_elimination"

    def rewrite(self, root, context):
        if not isinstance(root, Result) or not root.distinct:
            return None
        if isinstance(root.input, Aggregate):
            # a group's keys name it once in the aggregate's output
            return (replace(root, distinct=False)
                    if set(root.input.keys) <= set(root.input.output)
                    else None)
        if not isinstance(root.input, Project) or context.catalog is None:
            return None
        plan = LogicalPlan(root)
        if any(isinstance(node, Apply) for node in plan.walk()):
            return None
        returned = {(ref.alias, ref.column)
                    for ref in root.input.output_schema()}
        for scan in plan.operators().scans:
            id_column = context.catalog.get(scan.provider).id_col
            if (scan.alias, id_column) not in returned:
                return None
        return replace(root, distinct=False)


def _filtered_alias(operators: Operators) -> str | None:
    """Return the one alias whose AI.IF predicates are the only model calls."""
    if (operators.joins or operators.classifies or operators.applies
            or operators.projections or operators.label_filters
            or len(operators.filters) != 1):
        return None
    (alias,) = operators.filters
    return alias


def _stop_key(columns, alias) -> tuple[str, ...] | None:
    """Return the alias's columns the projection returns, or None."""
    key = []
    for column in columns:
        if not isinstance(column, ColumnRef) or column.alias != alias:
            return None
        if column.column not in key:
            key.append(column.column)
    return tuple(key)


class PerKeyStop:
    """Stop a filter per key once one of the key's documents passes.

    A DISTINCT over columns of one table asks only whether any document
    of each key value passes. When that table's AI.IF predicates are
    the plan's only model calls, the rule puts the key on its
    SemanticFilter nodes. The executor then reads a key's documents one
    at a time and skips the rest once one survives, so the answer
    tables list only the documents it read. The result rows do not
    change: each key value appears once either way.
    """

    name = "per_key_stop"

    def rewrite(self, root, context):
        if not isinstance(root, Result) or not root.distinct:
            return None
        if not isinstance(root.input, Project):
            return None
        alias = _filtered_alias(LogicalPlan(root).operators())
        if alias is None:
            return None
        key = _stop_key(root.input.columns, alias)
        if not key:
            return None

        def mark(node):
            if isinstance(node, SemanticFilter):
                return node if node.stop_key == key else replace(
                    node, stop_key=key)
            children = tuple(mark(child) for child in node.children())
            return node if children == node.children() else (
                node.with_children(children))

        marked = mark(root)
        return None if marked == root else marked


def built_in_logical_rules() -> tuple:
    """Return the logical rules registered with the built in registry.

    In order: distinct_elimination, per_key_stop, projection_pushdown,
    filter_pushdown, classify_placement, filter_order, and join_order.
    join_order follows filter_order because it ranks candidates by the
    whole query's seconds, the filter chains in their decided order
    included.
    """
    return (DistinctElimination(), PerKeyStop(), ProjectionPushdown(),
            FilterPushdown(), ClassifyPlacement(), FilterOrder(), JoinOrder())
