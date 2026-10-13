"""Built in logical optimizer rules."""

from __future__ import annotations

from dataclasses import replace

from quail.logical import (
    Aggregate,
    Alias,
    Apply,
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
    RegularPredicate,
    Result,
    Scan,
    SemanticClassify,
    SemanticFilter,
    SemanticJoin,
    SortKey,
    is_score,
    model_call,
)


def _column_refs(expression) -> tuple[ColumnRef, ...]:
    """Return the column references one logical expression reads."""
    if isinstance(expression, ColumnRef):
        return (expression,)
    if isinstance(expression, FilterPredicate):
        return _column_refs(expression.expression)
    if isinstance(expression, (ModelCall, Compare, Alias)):
        return tuple(model_call(expression).prompt.args)
    if isinstance(expression, (InList, RegularPredicate)):
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
            tested = {predicate.column.column for predicate in node.predicates}
            columns = tuple(
                column for column in needed.get(node.alias, ())
                if column != node.column
                or (node.alias, column) in returned
                or (node.alias, column) in score_inputs
                or column in tested
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
    cost_based = False

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
    cost_based = False

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


class DistinctElimination:
    """Clear a DISTINCT that cannot remove rows.

    A filter returns each document at most once and a join pairs each
    row pair once, so the result rows are unique when the projection
    returns the id column of every scanned table. The id column is
    taken as a key: nothing checks it, since that would read the
    whole column, and a table that repeats an id gets the repeated
    rows back. The rule reads the id columns from the catalog and
    leaves a plan alone without one, or when an apply() supplies the
    rows. A GROUP BY whose output returns every key is unique as well.
    """

    name = "distinct_elimination"
    cost_based = False

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
    if any(is_score(p.expression) for p in operators.filters[alias]):
        return None
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


class DistinctPushdown:
    """Push a DISTINCT into a filter: stop a key once one document passes.

    A DISTINCT over columns of one table asks only whether any document
    of each key value passes. When that table's AI.IF predicates are
    the plan's only model calls, the rule puts the key on its
    SemanticFilter nodes. The executor then reads a key's documents one
    at a time and skips the rest once one survives, so the answer
    tables list only the documents it read. The result rows do not
    change: each key value appears once either way.
    """

    name = "distinct_pushdown"
    cost_based = False

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

    In order: distinct_elimination, distinct_pushdown, projection_pushdown,
    and filter_pushdown. Every one is heuristic: it rewrites the plan
    without pricing it. The cost-based choices (where classifications
    run, filter order, join order and anchors) are made in the physical
    phase (quail.planner.ordering and the Quail backend's candidates).
    """
    return (DistinctElimination(), DistinctPushdown(), ProjectionPushdown(),
            FilterPushdown())
