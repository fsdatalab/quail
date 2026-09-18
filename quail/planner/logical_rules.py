"""Built in logical optimizer rules."""

from __future__ import annotations

from dataclasses import replace

from quail.logical import (
    Alias,
    ColumnRef,
    Compare,
    Equality,
    FilterPredicate,
    LogicalNode,
    ModelCall,
    Project,
    Scan,
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


def built_in_logical_rules() -> tuple:
    """Return the logical rules registered with the built in registry."""
    return (ProjectionPushdown(),)
