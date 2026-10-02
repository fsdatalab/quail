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

    The cost is a whole physical plan's estimated seconds, which needs
    each table's token counts, the model and device specs, and the
    join search. The logical optimizer context carries none of these,
    and the session tokenizes after the logical rules run, so the
    Quail planner applies this rule itself with its cost function.
    """

    name = "classify_placement"

    def __init__(self, cost):
        """Make the rule with the planner's cost function.

        Args:
            cost: Callable mapping a LogicalPlan to its physical plan's
                estimated seconds, or None when the plan is refused.
        """
        self.cost = cost

    def rewrite(self, root, context):
        lifted = lift_classifications(root)
        if lifted is None:
            return None
        before = self.cost(LogicalPlan(root))
        after = self.cost(LogicalPlan(lifted))
        if before is None or after is None or not after < before:
            return None
        return lifted


def built_in_logical_rules() -> tuple:
    """Return the logical rules registered with the built in registry."""
    return (ProjectionPushdown(),)
