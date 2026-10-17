"""Choose filter order, join order, and anchors using estimated cost."""

from dataclasses import dataclass

from quail.cost.budgets import parallel_copies
from quail.cost.work import Work, ask, scan
from quail.logical import (
    LogicalPlan,
    SemanticClassify,
    SemanticExtract,
    SemanticFilter,
    SemanticJoin,
    is_score,
    model_call,
)
from quail.logical.plan import Operators
from quail.planner.operators.common import question_tokens
from quail.planner.plan import Refusal
from quail.planner.search import join_order as joinsearch
from quail.planner.search.filter_order import default_order_rule
from quail.planner.search.statistics import (
    PlanStatistics,
    cached_statistics,
    live_after_filters,
)
from quail.planner.search.table_steps import (
    ASK,
    PlanPricing,
    order_table_steps,
    ready_table_steps,
    resident_after,
    table_steps,
)
from quail.planner.statistics.tables import thin
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
    steps = choose_step_orders(operators, statistics, rule=rule, context=context)
    orders = {alias: [step.key for step in table if step.kind == ASK]
              for alias, table in steps.items()}

    live0 = live_after_filters(plan, statistics)

    seq = [(specs[position], anchor) for position, anchor in
           choose_join_sequence(plan, statistics, steps, model=model,
                                device=device, rule=rule, workers=context.gpu_count)]
    _, join_steps = joinsearch.walk(
        seq, live0, statistics.lengths, (), pre, model, device,
        filters_after=filters_after_joins(steps),
        resident_after=resident_steps(plan))
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
                         live0, rule, source)


def filter_works(plan: LogicalPlan, statistics: PlanStatistics,
                 model: ModelSpec, step_orders: dict, rule="as_written") -> dict:
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
        rule: The ordering rule within each interval between callbacks and joins.
    """
    filters = plan.operators().filters
    works = {alias: Work() for alias in filters}
    live = {alias: float(stats.n_docs) for alias, stats in statistics.stats.items()}
    pending = {alias: list(steps) for alias, steps in step_orders.items()}
    completed = set()
    for spec in (None, *statistics.specs):
        if spec is not None:
            thin(live, spec)
            completed.add(spec.written_pos)
        for alias, steps in ready_table_steps(pending, completed):
            prefix = statistics.pre + statistics.stats[alias].mean_doc_tokens
            previous = None
            for step in order_table_steps(steps, rule, live[alias]):
                if step.kind == ASK and step.key in statistics.asks[alias]:
                    question = question_tokens(
                        statistics.lowered[filters[alias][step.key].prompt],
                        model.canvas_tokens)
                    operation = (ask if previous is not None
                                 and resident_after(previous, step) else scan)
                    works[alias] += operation(
                        prefix, question, window=model.sliding_window) * live[alias]
                live[alias] *= step.selectivity
                previous = step
    return works


def choose_step_orders(operators: Operators, statistics: PlanStatistics, *,
                       rule: str, context) -> dict:
    """Return alias -> its predicates, classifications, and extractions in run order.

    The steps are ordered together by order_filters_indexed: every
    step is tried first, and later steps are ranked by cost per
    expected rejection, each priced as resident or afresh by the step
    before it (table_steps). With rule "as_written", the predicates
    keep their written order within each interval between callbacks.
    A predicate's written position counts the alias's
    predicates over every SemanticFilter, lowest node first, as
    Operators lists them. A callback keeps its position relative to
    model calls and tests of their answers.

    Args:
        operators: The plan's operators.
        statistics: The plan's statistics.
        rule: "by_cost" or "as_written".
        workers: Model copies sharing balanced document work.
        context: The planning context, which prices the steps.
    """
    orders = {}
    for table in operators.scans:
        positions = range(len(operators.filters.get(table.alias, ())))
        steps = table_steps(operators, table.alias, positions,
                            PlanPricing(context, statistics, table.alias))
        if not steps:
            continue
        live = float(statistics.stats[table.alias].n_docs)
        orders[table.alias] = order_table_steps(steps, rule, live)
    return orders


def filters_after_joins(step_orders):
    """Return filter selectivities with the joins that precede their evaluation."""
    return tuple((step.joins, alias, step.selectivity)
                 for alias, steps in step_orders.items() for step in steps
                 if step.joins and step.selectivity != 1.0)


def choose_join_sequence(plan: LogicalPlan, statistics: PlanStatistics,
                         step_orders: dict, *, model, device, rule: str,
                         workers: int = 1) -> list:
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
        workers: Model copies sharing balanced document work.
    """
    if not plan.operators().joins:
        return []
    base_work = sum(
        (work * (1 / parallel_copies(workers, statistics.stats[alias].n_docs))
         for alias, work in filter_works(
             plan, statistics, model, step_orders, rule).items()), Work())
    live = live_after_filters(plan, statistics)
    arguments = (statistics.specs, live, statistics.lengths, (),
                 statistics.pre, statistics.chunk, model, device)
    operators = plan.operators()
    positions = {id(node): index for index, node in enumerate(operators.joins)}
    prerequisites = {}
    for apply in operators.applies:
        if apply.ids == "pairs":
            continue
        before = {positions[id(node)] for node in LogicalPlan(apply.input).walk()
                  if isinstance(node, SemanticJoin)}
        for index in range(len(operators.joins)):
            if index not in before:
                prerequisites.setdefault(index, set()).update(before)
    found = joinsearch.search_joins(
        *arguments, base_work=base_work, fixed_order=rule == "as_written",
        prerequisites=prerequisites, filters_after=filters_after_joins(step_orders),
        resident_after=resident_steps(plan), workers=workers)
    if found is None:
        found = joinsearch.search_joins(
            *arguments, base_work=base_work, fixed_order=True,
            prerequisites=prerequisites, filters_after=filters_after_joins(step_orders),
            resident_after=resident_steps(plan), workers=workers)
    return list(found.seq)


def resident_steps(plan):
    """Return the joins preceding model steps that compute document KV."""
    operators = plan.operators()
    positions = {id(node): index for index, node in enumerate(operators.joins)}
    resident = []
    for node in operators.nodes:
        if isinstance(node, SemanticFilter):
            calls = [model_call(p.expression) for p in node.predicates
                     if not is_score(p.expression)]
        elif isinstance(node, (SemanticClassify, SemanticExtract)):
            calls = [node.call]
        else:
            continue
        before = frozenset(positions[id(child)]
                           for child in LogicalPlan(node.input).walk()
                           if isinstance(child, SemanticJoin))
        resident.extend((before, call.aliases()[0]) for call in calls
                        if len(call.aliases()) == 1)
    return tuple(resident)
