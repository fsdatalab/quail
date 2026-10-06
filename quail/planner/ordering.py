"""Choose each table's filter order and the joins' stage order and anchors.

Both choices are physical: they change how a plan runs, not what it
returns. build_physical_plan makes them while it lowers a logical plan.
"""

from __future__ import annotations

from quail.cost.work import Work
from quail.logical import LogicalPlan, SemanticFilter, model_call
from quail.planner import join_order as joinsearch
from quail.planner.filter_order import order_filters_indexed
from quail.planner.statistics import (
    PlanStatistics,
    filter_works,
    live_after_filters,
    prepare_filter_costs,
)


def choose_filter_orders(plan: LogicalPlan, statistics: PlanStatistics, *,
                         model, device, rule: str, context=None) -> dict:
    """Return alias -> written positions of its model predicates in run order.

    Each SemanticFilter's predicates are ordered by order_filters_indexed:
    every predicate is tried first, and later predicates are ranked by
    cost per expected rejection. With rule "as_written", each filter
    keeps its written order. A written position counts the alias's
    predicates over every SemanticFilter, lowest node first, as
    Operators lists them.

    Args:
        plan: The logical plan.
        statistics: The plan's statistics.
        model: Model spec.
        device: Device spec.
        rule: "by_cost" or "as_written".
        context: The planning context, needed to price AI.SCORE
            predicates.
    """
    counted = {}
    orders = {}
    for node in plan.walk():
        if not isinstance(node, SemanticFilter):
            continue
        positions = []
        for predicate in node.predicates:
            (alias,) = model_call(predicate.expression).aliases()
            positions.append((alias, counted.get(alias, 0)))
            counted[alias] = counted.get(alias, 0) + 1
        (alias,) = model_call(node.predicates[0].expression).aliases()
        costs = prepare_filter_costs(
            node.predicates,
            prefix_tokens=(statistics.pre
                           + statistics.stats[alias].mean_doc_tokens),
            model=model, device=device, chunk_tokens=statistics.chunk,
            context=context)
        ordered = order_filters_indexed(
            costs, rule, count=statistics.stats[alias].n_docs)
        for index in ordered:
            owner, position = positions[index]
            orders.setdefault(owner, []).append(position)
    return orders


def choose_join_sequence(plan: LogicalPlan, statistics: PlanStatistics,
                         orders: dict, *, model, device, rule: str) -> list:
    """Return [(written position, anchor)] for the joins in run order.

    The left-deep search (join_order.search_joins) prices every
    connected stage order with every anchor choice together, ranking
    each by the whole query's estimated seconds with the filter chains
    in their chosen order. A written anchor is kept. With rule
    "as_written", only the anchors are chosen. When no connected
    left-deep order exists, the written order is priced.

    Args:
        plan: The logical plan.
        statistics: The plan's statistics.
        orders: alias -> written positions of its model predicates in
            run order, from choose_filter_orders.
        model: Model spec.
        device: Device spec.
        rule: "by_cost" or "as_written".
    """
    if not plan.operators().joins:
        return []
    base_work = sum(
        filter_works(plan, statistics, model, orders).values(), Work())
    live = live_after_filters(plan, statistics)
    filtered = set(plan.operators().all_filters())
    arguments = (statistics.specs, live, statistics.lengths, filtered,
                 statistics.pre, statistics.chunk, model, device)
    found = joinsearch.search_joins(
        *arguments, base_work=base_work, fixed_order=rule == "as_written")
    if found is None:
        found = joinsearch.search_joins(
            *arguments, base_work=base_work, fixed_order=True)
    return found["seq"]
