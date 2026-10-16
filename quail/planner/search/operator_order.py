"""Choose filter order, join order, and anchors using estimated cost."""

from dataclasses import dataclass

from quail.cost.work import Work
from quail.logical import (
    LogicalPlan,
    SemanticFilter,
    model_call,
)
from quail.logical.plan import Operators
from quail.planner.operators.common import question_tokens
from quail.planner.operators.filter import filter_alias_work, prepare_filter_costs
from quail.planner.plan import Refusal
from quail.planner.search import join_order as joinsearch
from quail.planner.search.filter_order import default_order_rule, order_filters_indexed
from quail.planner.search.statistics import (
    PlanStatistics,
    cached_statistics,
    live_after_filters,
)
from quail.planner.search.table_steps import resident_after, table_steps
from quail.planner.validation import join_input_tokens
from quail.specs import ModelSpec


@dataclass(frozen=True)
class OperatorOrder:
    """Operator order, input statistics, and work estimates for graph construction."""

    operators: Operators
    statistics: PlanStatistics
    filter_orders: dict
    step_orders: dict  # alias -> its TableSteps in run order
    join_sequence: list
    join_steps: list
    live: dict[str, float]
    filter_work: dict[str, Work]
    order_rule: str
    order_source: str


def choose_operator_order(plan, operators, model, device, doc_tokens, order,
                pair_fractions, context, scan_fractions):
    """Choose operator order and estimate work without constructing graph nodes."""
    scans, filters, joins = operators.scans, operators.filters, operators.joins
    statistics = cached_statistics(
        plan, {} if context is None else context.memo, model=model,
        device=device, doc_tokens=doc_tokens, pair_fractions=pair_fractions,
        scan_fractions=scan_fractions, context=context)
    stats, chunk, pre = statistics.stats, statistics.chunk, statistics.pre
    specs = statistics.specs
    asks = statistics.asks
    ask_filters = {alias: [filters[alias][position] for position in positions]
                   for alias, positions in asks.items() if positions}
    rule, source = (order, f"user: order={order!r}") if order else \
        default_order_rule(operators.all_filters(), joins)
    orders = choose_filter_orders(
        plan, statistics, model=model, device=device, rule=rule,
        context=context)
    steps = choose_step_orders(operators, statistics, rule=rule, context=context)

    live0 = live_after_filters(plan, statistics)
    works = filter_works(plan, statistics, model, orders)

    seq = [(specs[position], anchor) for position, anchor in
           choose_join_sequence(plan, statistics, orders, model=model,
                                device=device, rule=rule)]
    _, join_steps = joinsearch.walk(
        seq, live0, statistics.lengths, set(operators.all_filters()), pre,
        model, device)
    anchors = {spec.written_pos: anchor for spec, anchor in seq}
    for s in scans:
        fq = max((question_tokens(context.lowered(p.prompt), model.canvas_tokens)
                  for p in ask_filters.get(s.alias, ())), default=None)
        if fq is None:
            continue
        need = pre + stats[s.alias].max_doc_tokens + fq
        if need > chunk:
            return Refusal(
                reasons=(f"a document in {s.alias!r} needs {need} tokens "
                         f"with its prompt, but the forward pass budget "
                         f"is {chunk} tokens",),
                constraint="suffix_over_chunk",
                needed=need, available=chunk, unit="tokens")
    for spec in specs:
        if spec.cost is not None:
            continue
        anchor = anchors[spec.written_pos]
        need = join_input_tokens(
            spec, anchor, {alias: stats[alias].max_doc_tokens
                           for alias in spec.aliases}, pre)
        if need > chunk:
            return Refusal(
                reasons=(f"one join pair anchored on {anchor!r} needs "
                         f"{need} tokens, but the forward pass budget "
                         f"is {chunk} tokens",),
                constraint="suffix_over_chunk",
                needed=need, available=chunk, unit="tokens")

    return OperatorOrder(operators, statistics, orders, steps, seq, join_steps,
                         live0, works, rule, source)


def filter_works(plan: LogicalPlan, statistics: PlanStatistics,
                 model: ModelSpec, orders: dict) -> dict:
    """Return the expected Work of each table's filter sequence in run order.

    Args:
        plan: The logical plan.
        statistics: The plan's statistics.
        model: Model spec.
        orders: alias -> written positions of its model predicates in
            run order.
    """
    return {
        alias: filter_alias_work(
            predicates, statistics.stats[alias],
            [i for i in orders.get(alias, []) if i in statistics.asks[alias]],
            statistics.pre, statistics.lowered, model.canvas_tokens,
            model.sliding_window)
        for alias, predicates in plan.operators().filters.items()
    }


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


def choose_step_orders(operators: Operators, statistics: PlanStatistics, *,
                       rule: str, context) -> dict:
    """Return alias -> its asks, classifications, and extractions in run order.

    The steps are ordered together by order_filters_indexed, each priced
    as resident or afresh by the step before it (table_steps). With
    rule "as_written", the asks keep their written order and the
    classifications and extractions follow them. Without a planning
    context, only the asks are ordered.

    Args:
        operators: The plan's operators.
        statistics: The plan's statistics.
        rule: "by_cost" or "as_written".
        context: The planning context, which prices the steps.
    """
    if context is None:
        return {}
    orders = {}
    for scan in operators.scans:
        steps = table_steps(operators, scan.alias, context=context,
                            statistics=statistics)
        if not steps:
            continue
        ordered = order_filters_indexed(
            steps, rule, count=statistics.stats[scan.alias].n_docs,
            resident=resident_after)
        orders[scan.alias] = [steps[index] for index in ordered]
    return orders


def choose_join_sequence(plan: LogicalPlan, statistics: PlanStatistics,
                         orders: dict, *, model, device, rule: str) -> list:
    """Return [(written position, anchor)] for the joins in run order.

    The left-deep search (join_order.search_joins) prices every
    connected stage order with every anchor choice together, ranking
    each by the whole query's estimated seconds with the filter sequences
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
    return list(found.seq)
