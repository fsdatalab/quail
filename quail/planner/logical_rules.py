"""Built in logical optimizer rules."""

from __future__ import annotations

from dataclasses import replace

from quail.cost import budgets
from quail.cost.work import Work
from quail.logical import (
    Alias,
    Apply,
    ColumnRef,
    Compare,
    Equality,
    FilterPredicate,
    Join,
    LabelIn,
    LogicalNode,
    LogicalPlan,
    ModelCall,
    Project,
    Scan,
    SemanticClassify,
    SemanticFilter,
    SemanticJoin,
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
    if isinstance(expression, (ModelCall, Compare, LabelIn, Alias)):
        return tuple(model_call(expression).prompt.args)
    if isinstance(expression, Equality):
        return (expression.left, expression.right)
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
    returned = {
        (ref.alias, ref.column)
        for ref in (root.output_schema() if isinstance(root, Project)
                    else ())
    }
    score_inputs = {
        (ref.alias, ref.column)
        for expression in (root.columns if isinstance(root, Project) else ())
        if isinstance(expression, Alias)
        for ref in _column_refs(expression)
    }

    def descend(node, needed):
        needed = {alias: dict(columns) for alias, columns in needed.items()}
        for expression in node.expressions():
            for ref in _column_refs(expression):
                needed.setdefault(ref.alias, {})[ref.column] = None
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
    """The table a cheap one-table filter reads, or None for other nodes.

    A cheap filter is an Apply returning ids, or a SemanticFilter
    testing labels a SemanticClassify below it computed. An AI.IF
    filter asks the model and is not moved.
    """
    if isinstance(node, Apply) and node.ids != "pairs":
        return node.aliases[0]
    if isinstance(node, SemanticFilter) and all(
            isinstance(p.expression, LabelIn) for p in node.predicates):
        return model_call(node.predicates[0].expression).aliases()[0]
    return None


def _lets_through(node, alias: str) -> bool:
    """Whether a filter on one table may sit below node instead of above.

    True for a Join, a full SemanticJoin, a classification of joined
    rows, and any node working on another table. False at the table's
    own chain, at a gate (exists or anti join) and at an Apply
    returning the pairs a join asks about, whose rows the filter would
    change.
    """
    if isinstance(node, Join):
        return True
    if isinstance(node, SemanticJoin):
        return node.semantics == "full"
    if isinstance(node, SemanticClassify):
        return len(node.call.aliases()) == 2 or node.alias != alias
    if isinstance(node, Apply):
        return node.ids != "pairs" and node.aliases[0] != alias
    if isinstance(node, SemanticFilter):
        return _movable_filter_alias(node) != alias and all(
            model_call(p.expression).aliases()[0] != alias
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

    An Apply returning ids and a SemanticFilter testing labels read one
    table, so they commute with the joins, the classifications of
    joined rows, and the other tables' operators above that table.
    Each moves down to the top of its own table's chain, below the
    nodes it passed, keeping its order with the other filters of the
    table. A filter stays where it is when a gate (an exists or anti
    join) or an Apply returning a join's pairs sits between it and
    the first Join below.

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
    Apply returning ids and a filter on a label cost nothing the model
    runs, so they thin a table before the joins read it. Both front
    ends already place such filters on their table, so the rule
    rewrites plans built another way.
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
    while isinstance(node, SemanticClassify) or (
            isinstance(node, SemanticFilter)
            and all(isinstance(p.expression, LabelIn) for p in node.predicates)):
        lifted.append(node)
        node = node.input
    return node, tuple(reversed(lifted))


def _chain_alias(node) -> str:
    """The table alias a classification or label filter node works on."""
    if isinstance(node, SemanticClassify):
        return node.alias
    return model_call(node.predicates[0].expression).aliases()[0]


def lift_classifications(root: LogicalNode) -> LogicalNode | None:
    """Move every joined table's classifications above the joins.

    Each one-table SemanticClassify of a table a SemanticJoin reads,
    with the SemanticFilters testing its label, leaves the table's
    chain and sits above the topmost join under the root Project, in
    scan order. A classification of a table no join reads stays.

    Returns:
        The rewritten root, or None when no classification moved.
    """
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
    AI.IF filters kept and a filter on its label thins the join's
    input; after the joins, it labels only the documents the joins
    matched. The rule keeps the plan as written, with each
    classification on its table, unless the plan with every joined
    table's classifications above the joins (lift_classifications)
    costs less.

    The cost is a whole physical plan's estimated seconds
    (quail.planner.pricing): build_physical_plan's plan for the
    candidate, with the label_scoring rule applied so each
    classification's scoring rule is counted. It reads the model and
    device specs, the document token counts, and the pair fractions
    from the logical planning context, so the rule does nothing on a
    context without statistics or for another backend.
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
    """Whether the Quail cost model applies to a plan and context.

    It needs the statistics (a model and device), prices the Quail
    backend's filter, join, and classification plans only (an
    AI.SCORE plan is the reranker planner's), and has nothing to say
    about a model whose weights do not fit one GPU, which the physical
    planner refuses.
    """
    return (context.model is not None and context.backend == "quail"
            and budgets.minimum_weight_gpus(context.model, context.device) == 1
            and not has_score(LogicalPlan(root)))


def _statistics(root, context) -> PlanStatistics:
    """The plan's statistics, shared through the context's memo."""
    return cached_statistics(
        LogicalPlan(root), context.memo, model=context.model,
        device=context.device, doc_tokens=context.document_tokens,
        pair_fractions=context.pair_fractions)


def _order_rule(context) -> str:
    return context.order or default_order_rule({}, ())[0]


class FilterOrder:
    """Order each table's AI.IF predicates by expected cost.

    Each SemanticFilter's predicates get the order that minimizes the
    chain's ideal expected time (order_filters_indexed): the first
    predicate scans every document's prefix, each later one asks over
    the KV the chain keeps, and a predicate's selectivity thins what
    follows. The rule records the order on the node when it differs
    from the written order; with order="as_written" it leaves every
    node as written. Both front ends put a table's AI.IF predicates in
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
            asks = [index for index, predicate in enumerate(node.predicates)
                    if not isinstance(predicate.expression, LabelIn)]
            if not asks or rule == "as_written":
                return node if not node.order else replace(node, order=())
            (alias,) = model_call(node.predicates[asks[0]].expression).aliases()
            ordered = [asks[index] for index in order_filters_indexed(
                [node.predicates[index] for index in asks], rule,
                prefix_tokens=(statistics.pre
                               + statistics.stats[alias].mean_doc_tokens),
                model=model, device=device, chunk_tokens=statistics.chunk)]
            ordered.extend(index for index in range(len(node.predicates))
                           if index not in asks)
            order = () if ordered == list(range(len(ordered))) else tuple(ordered)
            return node if order == node.order else replace(node, order=order)

        rewritten = visit(root)
        return None if rewritten == root else rewritten


class JoinOrder:
    """Choose the joins' stage order and each stage's anchor by cost.

    The left-deep search (quail.planner.joins.search_joins) prices
    every connected stage order with every anchor choice together,
    since a stage's cost depends on which table's KV is computed once
    and on which documents' KV earlier stages left resident, and
    ranks each candidate by the whole query's predicted seconds, the
    filter chains in their decided order included. A written anchor
    is honored; with order="as_written" only the anchors are chosen.
    When no connected left-deep order exists the written order is
    priced. The rule records each join's execution position and
    anchor on its SemanticJoin.
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
        filtered = set(plan.operators().filters)
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


def built_in_logical_rules() -> tuple:
    """Return the logical rules registered with the built in registry.

    In order: projection_pushdown, filter_pushdown, classify_placement,
    filter_order, and join_order. join_order follows filter_order
    because it ranks candidates by the whole query's seconds, the
    filter chains in their decided order included.
    """
    return (ProjectionPushdown(), FilterPushdown(), ClassifyPlacement(),
            FilterOrder(), JoinOrder())
