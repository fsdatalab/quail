"""Rules that change query trees."""

from __future__ import annotations

from dataclasses import replace

from quail.logical import (
    Aggregate,
    Apply,
    LogicalPlan,
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

    name = "should_eliminate_distinct"

    def rewrites(self, root, context):
        if not isinstance(root, Result) or not root.distinct:
            return [root]
        if isinstance(root.input, Aggregate):
            # a group's keys name it once in the aggregate's output
            return [replace(root, distinct=False)
                    if set(root.input.keys) <= set(root.input.output)
                    else root]
        if not isinstance(root.input, Project) or context.catalog is None:
            return [root]
        plan = LogicalPlan(root)
        if any(isinstance(node, Apply) for node in plan.walk()):
            return [root]
        returned = {(ref.alias, ref.column)
                    for ref in root.input.output_schema()}
        for scan in plan.operators().scans:
            id_column = context.catalog.get(scan.provider).id_col
            if (scan.alias, id_column) not in returned:
                return [root]
        return [replace(root, distinct=False)]


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
        alias = _filtered_alias(LogicalPlan(root).operators())
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
    """Push one-table label and answer filters below the joins above their table.

    Like Catalyst's PushDownPredicates, the move is unconditional: a
    Filter on a label column runs no model, so it thins a table before
    the joins read it. Both front ends already place such filters on
    their table; the rule rewrites plans built another way. An Apply
    stays where it was written, since its function sees its input rows.
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
