"""Choose filter order, join order, and anchors using estimated cost."""

from dataclasses import dataclass

from quail.cost.work import Work, ask, scan
from quail.logical import (
    LogicalPlan,
)
from quail.logical.plan import Operators
from quail.planner.operators.common import question_tokens
from quail.planner.plan import Refusal
from quail.planner.search import join_order as joinsearch
from quail.planner.search.filter_order import default_order_rule, order_filters_indexed
from quail.planner.search.statistics import (
    PlanStatistics,
    cached_statistics,
    expected_rows_after_filters,
)
from quail.planner.search.table_steps import (
    ASK,
    PlanPricing,
    resident_after,
    table_steps,
)
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
    expected_rows: dict[str, float]
    filter_work: dict[str, Work]
    order_rule: str
    order_source: str


def choose_operator_order(plan, operators, model, device, document_tokens, order,
                pair_fractions, context, scan_fractions):
    """Choose operator order and estimate work without constructing graph nodes."""
    scans, filters, joins = operators.scans, operators.filters, operators.joins
    statistics = cached_statistics(
        plan, {} if context is None else context.memo, model=model,
        device=device, document_tokens=document_tokens, pair_fractions=pair_fractions,
        scan_fractions=scan_fractions, context=context)
    stats, chunk, pre = statistics.stats, statistics.chunk, statistics.pre
    specs = statistics.specs
    asks = statistics.asks
    ask_filters = {alias: [filters[alias][position] for position in positions]
                   for alias, positions in asks.items() if positions}
    rule, source = (order, f"user: order={order!r}") if order else \
        default_order_rule(operators.all_filters(), joins)
    steps = choose_step_orders(operators, statistics, rule=rule, context=context)
    orders = {alias: [step.key for step in table if step.kind == ASK]
              for alias, table in steps.items()}

    live0 = expected_rows_after_filters(plan, statistics)
    works = filter_works(plan, statistics, model, steps)

    seq = [(specs[position], anchor) for position, anchor in
           choose_join_sequence(plan, statistics, steps, model=model,
                                device=device, rule=rule)]
    _, join_steps = joinsearch.walk(
        seq, live0, statistics.lengths, set(operators.all_filters()), pre,
        model, device)
    anchors = {spec.written_position: anchor for spec, anchor in seq}
    for s in scans:
        fq = max((question_tokens(context.lowered(p.prompt), model.canvas_tokens)
                  for p in ask_filters.get(s.alias, ())), default=None)
        if fq is None:
            continue
        need = pre + stats[s.alias].max_document_tokens + fq
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
        anchor = anchors[spec.written_position]
        need = join_input_tokens(
            spec, anchor, {alias: stats[alias].max_document_tokens
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
                 model: ModelSpec, step_orders: dict) -> dict:
    """Return the expected Work of each table's AI.IF asks in run order.

    An ask reads the documents afresh when it starts the table's
    sequence or follows a step that leaves no KV for it, and the
    resident KV otherwise; it reads the documents the steps before it
    keep. The classifications and extractions among the steps carry
    their own estimates; a score predicate is priced by its AiScore.

    Args:
        plan: The logical plan.
        statistics: The plan's statistics.
        model: Model spec.
        step_orders: alias -> its TableSteps in run order.
    """
    works = {}
    for alias, predicates in plan.operators().filters.items():
        stats = statistics.stats[alias]
        prefix = statistics.pre + stats.mean_document_tokens
        count, total, previous = float(stats.num_documents), Work(), None
        for step in step_orders.get(alias, ()):
            if step.kind == ASK and step.key in statistics.asks[alias]:
                question = question_tokens(
                    statistics.lowered[predicates[step.key].prompt],
                    model.canvas_tokens)
                operation = (ask if previous is not None
                             and resident_after(previous, step) else scan)
                total = total + operation(
                    prefix, question, window=model.sliding_window) * count
            count *= step.selectivity
            previous = step
        works[alias] = total
    return works


def choose_step_orders(operators: Operators, statistics: PlanStatistics, *,
                       rule: str, context) -> dict:
    """Return alias -> its predicates, classifications, and extractions in run order.

    The steps are ordered together by order_filters_indexed: every
    step is tried first, and later steps are ranked by cost per
    expected rejection, each priced as resident or afresh by the step
    before it (table_steps). With rule "as_written", the predicates
    keep their written order and the classifications and extractions
    follow them. A predicate's written position counts the alias's
    predicates over every SemanticFilter, lowest node first, as
    Operators lists them.

    Args:
        operators: The plan's operators.
        statistics: The plan's statistics.
        rule: "by_cost" or "as_written".
        context: The planning context, which prices the steps.
    """
    orders = {}
    for table in operators.scans:
        positions = range(len(operators.filters.get(table.alias, ())))
        steps = table_steps(operators, table.alias, positions,
                            PlanPricing(context, statistics, table.alias))
        if not steps:
            continue
        ordered = order_filters_indexed(
            steps, rule, count=statistics.stats[table.alias].num_documents,
            resident=resident_after)
        orders[table.alias] = [steps[index] for index in ordered]
    return orders


def choose_join_sequence(plan: LogicalPlan, statistics: PlanStatistics,
                         step_orders: dict, *, model, device, rule: str) -> list:
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
        step_orders: alias -> its TableSteps in run order, from
            choose_step_orders.
        model: Model spec.
        device: Device spec.
        rule: "by_cost" or "as_written".
    """
    if not plan.operators().joins:
        return []
    base_work = sum(
        filter_works(plan, statistics, model, step_orders).values(), Work())
    expected_rows = expected_rows_after_filters(plan, statistics)
    filtered = set(plan.operators().all_filters())
    arguments = (statistics.specs, expected_rows, statistics.lengths, filtered,
                 statistics.pre, statistics.chunk, model, device)
    found = joinsearch.search_joins(
        *arguments, base_work=base_work, fixed_order=rule == "as_written")
    if found is None:
        found = joinsearch.search_joins(
            *arguments, base_work=base_work, fixed_order=True)
    return list(found.seq)


def resident_asks(step_orders: dict) -> frozenset:
    """Return the aliases whose first ask reads KV a step before it left."""
    resident = set()
    for alias, steps in step_orders.items():
        previous = None
        for step in steps:
            if step.kind == ASK:
                if previous is not None and resident_after(previous, step):
                    resident.add(alias)
                break
            previous = step
    return frozenset(resident)
