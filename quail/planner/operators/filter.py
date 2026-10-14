"""AI.IF over one table: the cost and work of its filters."""

from functools import partial
from operator import mul

from quail.cost.filters import filter_cost, filter_sequence_work
from quail.cost.work import Work
from quail.logical import (
    LogicalPlan,
    effective_selectivity,
    is_score,
)
from quail.planner.operators.common import question_tokens
from quail.planner.operators.score import score_statistics
from quail.planner.search.filter_order import PredicateCost
from quail.planner.statistics.query import PlanStatistics
from quail.specs import ModelSpec


def prepare_filter_costs(predicates, *, prefix_tokens, model, device,
                         chunk_tokens, context=None,
                         lower=None) -> tuple[PredicateCost, ...]:
    """Prepare costs and reusable result identities for predicate ordering.

    Args:
        predicates: The model predicates of one table.
        prefix_tokens: The preamble and mean document tokens.
        model: Model spec.
        device: Device spec.
        chunk_tokens: The forward pass token budget.
        context: The planning context; prices scores and lowers prompts.
        lower: Lowers a prompt when there is no context.
    """
    lower = context.lowered if lower is None else lower
    prepared = []
    for predicate in predicates:
        if is_score(predicate.expression):
            _, cost = score_statistics(
                predicate.prompt, context, chunk_tokens, mean_tokens=prefix_tokens)
            first = later = cost.seconds
            reuse_key = predicate.prompt
        else:
            q = question_tokens(lower(predicate.prompt), model.canvas_tokens)
            first = partial(mul, filter_cost(
                q, prefix_tokens, model, device, chunk_tokens, first=True))
            later = partial(mul, filter_cost(
                q, prefix_tokens, model, device, chunk_tokens, first=False))
            reuse_key = None
        prepared.append(PredicateCost(
            effective_selectivity(predicate.selectivity), first, later, reuse_key))
    return tuple(prepared)


def filter_alias_work(preds, stats, order, pre: int, lowered,
                      canvas: int = 0, window: int = 0) -> Work:
    """Prepare a filter sequence's counts and price its recorded order."""
    predicates = [preds[index] for index in order]
    return filter_sequence_work(
        [question_tokens(lowered[predicate.prompt], canvas)
         for predicate in predicates],
        [effective_selectivity(predicate.selectivity) for predicate in predicates],
        pre + stats.mean_doc_tokens, float(stats.n_docs), window)


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
