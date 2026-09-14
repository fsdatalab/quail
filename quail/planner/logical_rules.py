"""Built in logical optimizer rules."""

from __future__ import annotations

from dataclasses import replace

from quail.logical import (
    ColumnRef,
    Equality,
    FilterPredicate,
    LogicalNode,
    Project,
    Prompt,
    Scan,
)


def _column_refs(expression) -> tuple[ColumnRef, ...]:
    """Return the column references one logical expression reads."""
    if isinstance(expression, ColumnRef):
        return (expression,)
    if isinstance(expression, FilterPredicate):
        return tuple(expression.prompt.args)
    if isinstance(expression, Prompt):
        return tuple(expression.args)
    if isinstance(expression, Equality):
        return (expression.left, expression.right)
    return ()


def required_columns(root: LogicalNode) -> dict[str, tuple[str, ...]]:
    """Return the source columns each alias reads, in first use order.

    A column is read when the root projection returns it, when a
    filter or join prompt names it, or when a join condition compares
    it.
    """
    needed: dict[str, dict[str, None]] = {}

    def visit(node):
        for expression in node.expressions():
            for ref in _column_refs(expression):
                needed.setdefault(ref.alias, {})[ref.column] = None
        for child in node.children():
            visit(child)

    visit(root)
    return {alias: tuple(columns) for alias, columns in needed.items()}


def push_down_projection(root: LogicalNode) -> LogicalNode:
    """Rewrite every Scan to keep only the column values the plan reads.

    The document column is always tokenized. It is kept as a value too
    only when the root projection returns it.
    """
    needed = required_columns(root)
    returned = {
        (ref.alias, ref.column)
        for ref in (root.output_schema() if isinstance(root, Project)
                    else ())
    }

    def rewrite(node):
        if isinstance(node, Scan):
            columns = tuple(
                column for column in needed.get(node.alias, ())
                if column != node.column
                or (node.alias, column) in returned
            )
            return node if columns == node.columns else replace(
                node, columns=columns)
        children = tuple(rewrite(child) for child in node.children())
        if children == node.children():
            return node
        return node.with_children(children)

    return rewrite(root)


class ProjectionPushdown:
    """Push the projected column set down to each Scan.

    The rule fires at the root Project, where the whole tree is
    visible, and rewrites the Scans beneath it. Other nodes pass
    through unchanged.
    """

    name = "projection_pushdown"

    def rewrite(self, node, context):
        if not isinstance(node, Project):
            return None
        return push_down_projection(node)


def built_in_logical_rules() -> tuple:
    """Return the logical rules registered with the built in registry."""
    return (ProjectionPushdown(),)
