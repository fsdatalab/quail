"""Rules that change query trees."""

from __future__ import annotations

from dataclasses import replace

from quail.logical import (
    Aggregate,
    Project,
    Result,
    SemanticFilter,
)
from quail.planner.optimizer.rules import LogicalRule
from quail.planner.rules.tree_transforms import (
    _filtered_alias,
    _stop_key,
    lift_maps,
    push_down_filters,
    push_down_projection,
)


class ShouldEliminateDistinct(LogicalRule):
    """Clear DISTINCT when an aggregate returns every grouping key."""

    name = "should_eliminate_distinct"

    def rewrites(self, root, context):
        if not isinstance(root, Result) or not root.distinct:
            return [root]
        if isinstance(root.input, Aggregate):
            # a group's keys name it once in the aggregate's output
            return [replace(root, distinct=False)
                    if set(root.input.keys) <= set(root.input.output)
                    else root]
        return [root]


class ShouldPushDownDistinct(LogicalRule):
    """Push a DISTINCT into a filter: stop a key once one document passes.

    A DISTINCT over columns of one table asks only whether any document
    of each key value passes. When that table's AI.IF predicates are
    the plan's only model calls, the rule puts the key on its
    SemanticFilter nodes. The executor then reads a key's documents one
    at a time and skips the rest once one survives, so the answer
    tables list only the documents it read. The result rows do not
    change: each key value appears once either way.
    """

    name = "should_push_down_distinct"

    def rewrites(self, root, context):
        if not isinstance(root, Result) or not root.distinct:
            return [root]
        if not isinstance(root.input, Project):
            return [root]
        alias = _filtered_alias(root.input.input)
        if alias is None:
            return [root]
        key = _stop_key(root.input.columns, alias)
        if not key:
            return [root]

        def mark(node):
            if isinstance(node, SemanticFilter):
                return node if node.stop_key == key else replace(
                    node, stop_key=key)
            children = tuple(mark(child) for child in node.children())
            return node if children == node.children() else (
                node.with_children(children))

        marked = mark(root)
        return [marked]


class ShouldPushDownProjection(LogicalRule):
    """Push the projected column set down to each Scan."""

    name = "should_push_down_projection"

    def rewrites(self, root, context):
        rewritten = push_down_projection(root)
        return [rewritten]


class ShouldPushDownFilters(LogicalRule):
    """Push cheap one-table filters below the joins above their table.

    A label condition tests each row independently and can thin its
    table before a full join. A callback depends on its input relation
    and keeps its position.
    """

    name = "should_push_down_filters"

    def rewrites(self, root, context):
        rewritten = push_down_filters(root)
        return [root if rewritten is None else rewritten]


class ShouldLiftMaps(LogicalRule):
    """Offer classification and extraction placement above joins."""

    name = "should_lift_maps"

    def rewrites(self, root, context):
        rewritten = lift_maps(root)
        return [root] if rewritten is None else [root, rewritten]
