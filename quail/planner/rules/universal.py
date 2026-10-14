"""Rules applied whenever their conditions hold, without comparing costs."""

from __future__ import annotations

from dataclasses import replace

from quail.cost.retention import coefficients
from quail.logical import (
    Aggregate,
    Apply,
    LogicalPlan,
    Project,
    Result,
    SemanticFilter,
)
from quail.physical import (
    Aggregate as PhysicalAggregate,
)
from quail.physical import (
    AiFilter,
    AiInference,
    AiJoin,
    Limit,
    PhysicalGraph,
    Sort,
)
from quail.physical.pipelines import build_pipelines
from quail.planner.graph import retention
from quail.planner.plan import plan_kv_retention
from quail.planner.rules.tree_transforms import (
    _filtered_alias,
    _stop_key,
    push_down_filters,
    push_down_projection,
)
from quail.planner.search.statistics import cached_statistics, live_after_filters


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
    target = "logical"
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
    target = "logical"
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


class ProjectionPushdown:
    """Push the projected column set down to each Scan."""

    name = "projection_pushdown"
    target = "logical"
    cost_based = False

    def rewrite(self, root, context):
        rewritten = push_down_projection(root)
        return None if rewritten is root else rewritten


class FilterPushdown:
    """Push cheap one-table filters below the joins above their table.

    Like Catalyst's PushDownPredicates, the move is unconditional: an
    Apply returning ids and a Filter on a label column run no model, so
    they thin a table before the joins read it. Both front ends already
    place such filters on their table; the rule rewrites plans built
    another way.
    """

    name = "filter_pushdown"
    target = "logical"
    cost_based = False

    def rewrite(self, root, context):
        return push_down_filters(root)


class LimitPushdown:
    """Let a LIMIT end the filter run once enough documents survived.

    The rule puts the limit in the ``filter_limit`` setting, which the
    executor reads to stop admitting documents once that many survived
    the sequence's last stage. The pushdown keeps the first survivors to
    finish, so it is sound only when those are the rows the query
    returns: the graph ends in a Limit and has no Sort or PhysicalAggregate (an
    ORDER BY, DISTINCT, OFFSET, or GROUP BY needs every survivor), no
    AiJoin (survivors are settled per join round), and no AiInference (a
    score comparison after the sequence can drop survivors). Otherwise the
    rule removes the setting.
    """

    name = "limit_pushdown"
    target = "physical"
    cost_based = False

    def rewrite(self, graph: PhysicalGraph, context) -> PhysicalGraph | None:
        if context is None:
            return None
        limit = next((node for node in graph.nodes
                      if isinstance(node, Limit)), None)
        blocked = any(isinstance(node, (Sort, PhysicalAggregate, AiJoin, AiInference))
                      for node in graph.nodes)
        if limit is None or blocked:
            context.settings.pop("filter_limit", None)
        else:
            context.settings["filter_limit"] = limit.count
        return None


class KvRetention:
    """Keep the KV that later operators read resident, within the page cap.

    The schedule (quail.planner.graph.retention.schedule) records, at each
    execution boundary, which aliases a later stage anchors on and how
    likely each document is to survive to that use. The executor's
    retention pool keeps their KV by that priority, up to the page cap.

    - A filter sequence keeps its survivors' KV (keep_kv) when a later join
      anchors on its table, unless the sequence streams into that join in
      one pipeline on one GPU.
    - A join keeps its anchor's KV (keep_kv) when a later group
      anchors on the same table.
    - A sequence writes KV pages only when a later stage, the next operator
      of its pipeline, or the retention pool reads them.

    The rule needs the logical plan on the context and fires only on a
    plan without a ``retention`` setting, so a later pass keeps the
    first schedule. It puts the schedule, the retention coefficients,
    and the page cap in the ``retention`` setting.
    """

    name = "kv_retention"
    target = "physical"
    cost_based = False

    def rewrite(self, graph: PhysicalGraph, context) -> PhysicalGraph | None:
        if context is None or context.logical_plan is None \
                or "retention" in context.settings:
            return None
        logical = context.logical_plan
        statistics = cached_statistics(
            logical, context.memo, model=context.model, device=context.device,
            doc_tokens=context.document_tokens,
            pair_fractions=context.pair_fractions,
            scan_fractions=context.scan_fractions, context=context)
        joins = [node for node in graph.nodes if isinstance(node, AiJoin)]
        grouped = []
        for node in joins:
            if (not grouped or grouped[-1][-1].anchor != node.anchor
                    or grouped[-1][-1].semantics != "full"
                    or node.semantics != "full"):
                grouped.append([])
            grouped[-1].append(node)
        groups = [[(statistics.specs[node.written_pos], node.anchor)
                   for node in group] for group in grouped]
        schedule = retention.schedule(
            [predicate for group in groups for predicate in group],
            live_after_filters(logical, statistics),
            [group[0].node_id for group in grouped])
        before, after = schedule["before"], schedule["after"]
        schedule["before"] = {
            node.node_id: before[group[0].node_id]
            for group in grouped for node in group}
        schedule["after"] = {
            node.node_id: (after[group[0].node_id] if node is group[-1]
                           else before[group[0].node_id])
            for group in grouped for node in group}
        paths = build_pipelines(graph, context.gpu_count)
        streamed = {
            node.alias for node in graph.nodes if isinstance(node, AiFilter)
            and any(isinstance(member, AiJoin)
                    for member in paths[node.node_id])}
        schedule["initial"] = {
            alias: use for alias, use in schedule["initial"].items()
            if alias not in streamed}
        schedule.update(**coefficients(context.model, context.device),
                        cap_pages=statistics.cap_pages)
        context.settings["retention"] = schedule
        return plan_kv_retention(
            graph, context.gpu_count, tuple(context.settings.get("pre_ids", ())))
