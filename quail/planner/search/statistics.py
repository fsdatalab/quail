"""The numbers the rules and the physical planner read.

One PlanStatistics holds what the cost model knows about a logical
plan before any decision: each table's document lengths, the model's
token budgets, the preamble, and the joins as the search prices them.
The functions after it read the decisions the tree rewrites recorded
on the plan's nodes: each table's filter order and the join sequence.
"""

from dataclasses import dataclass

from quail.cost import budgets
from quail.cost.retention import retention_pages
from quail.logical import (
    Filter,
    LogicalPlan,
    SemanticFilter,
    SemanticJoin,
    effective_selectivity,
    is_score,
    model_call,
)
from quail.planner.operators.common import preamble_tokens
from quail.planner.operators.join import join_specs
from quail.planner.plan import CorpusStats
from quail.planner.statistics.memory import pool_split
from quail.planner.statistics.tables import scale_alias, summarize_alias
from quail.specs import DeviceSpec, ModelSpec


@dataclass(frozen=True)
class PlanStatistics:
    """What the cost model knows about one plan before any decision.

    Attributes:
        stats: alias -> CorpusStats of its documents.
        lengths: alias -> AliasStats, the length sums the join search
            and the stage pricing read.
        chunk: The forward pass token budget.
        arena_pages: The KV arena's page split.
        admission: The admission budget in tokens.
        cap_pages: Pages the retention pool may hold.
        pre: The engine preamble's token count.
        specs: The joins as PricedJoin records, in written order.
        asks: alias -> written positions of its AI.IF predicates.
        lowered: Every prompt of the plan, lowered to token ids.
    """

    stats: dict
    lengths: dict
    chunk: int
    arena_pages: tuple
    admission: int
    cap_pages: int
    pre: int
    specs: tuple
    asks: dict
    lowered: dict


def ask_positions(filters) -> dict:
    """Return alias -> written positions of its AI.IF predicates."""
    return {alias: [i for i, p in enumerate(predicates)
                   if not is_score(p.expression)]
            for alias, predicates in filters.items()}


def plan_statistics(plan: LogicalPlan, *, model: ModelSpec,
                    device: DeviceSpec, document_tokens: dict,
                    pair_fractions=None, scan_fractions=None,
                    context=None) -> PlanStatistics:
    """Gather the statistics of one logical plan.

    Args:
        plan: The logical plan.
        model: Model spec.
        device: Device spec.
        document_tokens: alias -> per-document token counts, of every
            document in the table.
        pair_fractions: join written position -> the fraction of the
            cross product its equality conditions keep.
        scan_fractions: alias -> the fraction of its documents the
            regular predicates are expected to keep; the alias's counts and
            sums are scaled by it.
        context: Prompt preparation and model settings for score operators.

    Raises:
        ValueError: A scanned alias has no document token counts.
    """
    operators = plan.operators()
    asks = ask_positions(operators.filters)
    ask_filters = {alias: [operators.filters[alias][position]
                           for position in positions]
                   for alias, positions in asks.items() if positions}
    if context is None:
        raise ValueError("plan statistics need a planning context")
    lowered = {prompt: context.lowered(prompt) for prompt in (
        *operators.prompts, *(node.call.prompt for node in operators.extracts))}
    lengths = {a: summarize_alias(t, model.sliding_window,
                                  memo=context.memo if context is not None else None)
               for a, t in document_tokens.items()}
    chunk = budgets.chunk_budget(model, device)
    arena_split = pool_split(model, device, chunk, lengths.values())
    for alias, fraction in (scan_fractions or {}).items():
        if alias in lengths:
            lengths[alias] = scale_alias(lengths[alias], fraction)
    stats = {
        a: CorpusStats(num_documents=s.count, total_tokens=s.total,
                       max_document_tokens=s.maximum)
        for a, s in lengths.items()
    }
    for s in operators.scans:
        if s.alias not in stats:
            raise ValueError(f"no document_tokens for alias {s.alias!r}")
    admission = arena_split[0] * budgets.PAGE_TOKENS
    return PlanStatistics(
        stats=stats, lengths=lengths, chunk=chunk,
        arena_pages=tuple(arena_split), admission=admission,
        cap_pages=retention_pages(admission, chunk, budgets.PAGE_TOKENS),
        pre=preamble_tokens(
            lowered[prompt] for prompt in (
                *(p.prompt for fs in ask_filters.values() for p in fs),
                *(j.prompt for j in operators.joins))),
        specs=tuple(join_specs(operators.joins, pair_fractions,
                               model.canvas_tokens, context=context,
                               chunk=chunk)),
        lowered=lowered,
        asks=asks)


def cached_statistics(plan: LogicalPlan, memo: dict, *, model: ModelSpec,
                      device: DeviceSpec, document_tokens: dict,
                      pair_fractions=None, scan_fractions=None,
                      context=None) -> PlanStatistics:
    """Return plan_statistics, computed once per memo and plan root.

    Every tree rewrite and physical planner of one query that prices
    the same plan share one summary of each table's lengths.
    """
    key = ("statistics", plan.root)
    if key not in memo:
        memo[key] = plan_statistics(
            plan, model=model, device=device, document_tokens=document_tokens,
            pair_fractions=pair_fractions, scan_fractions=scan_fractions,
            context=context)
    return memo[key]


def filter_stop_keys(plan: LogicalPlan) -> dict:
    """Return each alias's DISTINCT stop key from its filters."""
    keys = {}
    for node in plan.walk():
        if isinstance(node, SemanticFilter) and node.stop_key:
            (alias,) = model_call(node.predicates[0].expression).aliases()
            keys[alias] = tuple(node.stop_key)
    return keys


def expected_rows_after_filters(plan: LogicalPlan, statistics: PlanStatistics) -> dict:
    """Return the expected live documents per alias after its filters run.

    A filter above a join thins nothing before the joins: its table's
    documents all reach the first join.
    """
    expected_rows = {a: float(st.num_documents) for a, st in statistics.stats.items()}
    for node in plan.walk():
        if not isinstance(node, (Filter, SemanticFilter)):
            continue
        if any(isinstance(below, SemanticJoin)
               for below in LogicalPlan(node.input).walk()):
            continue
        if isinstance(node, Filter):
            alias = node.condition.column.alias
            expected_rows[alias] *= effective_selectivity(node.selectivity)
        else:
            for predicate in node.predicates:
                (alias,) = model_call(predicate.expression).aliases()
                expected_rows[alias] *= effective_selectivity(predicate.selectivity)
    return expected_rows




















