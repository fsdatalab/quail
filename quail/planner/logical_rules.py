"""Built in logical optimizer rules."""

from __future__ import annotations

from dataclasses import replace

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
    model_call,
)
from quail.planner import pricing


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
    (quail.planner.pricing): the Quail planner's plan for the
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
                refused. The default prices the Quail planner's plan.
        """
        self.cost = pricing.estimated_seconds if cost is None else cost

    def rewrite(self, root, context):
        if context is not None and (
                context.model is None or context.backend != "quail"):
            return None
        lifted = lift_classifications(root)
        if lifted is None:
            return None
        before = self.cost(LogicalPlan(root), context)
        after = self.cost(LogicalPlan(lifted), context)
        if before is None or after is None or not after < before:
            return None
        return lifted


def built_in_logical_rules() -> tuple:
    """Return the logical rules registered with the built in registry.

    In order: projection_pushdown, filter_pushdown, and
    classify_placement.
    """
    return (ProjectionPushdown(), FilterPushdown(), ClassifyPlacement())
