"""AI.IF over one table: the cost and work of its filters."""

from dataclasses import dataclass
from functools import partial
from operator import mul
from typing import Callable, Hashable

from quail.cost.filters import filter_cost, filter_sequence_work
from quail.cost.work import Work
from quail.logical import (
    effective_selectivity,
    is_score,
)
from quail.physical import AiFilter, PortRef
from quail.physical.base import input_ports
from quail.planner.operators.common import question_tokens
from quail.planner.operators.score import score_statistics


@dataclass(frozen=True)
class PredicateCost:
    """One predicate's costs and reusable result identity."""

    selectivity: float
    first: Callable[[float], float]
    later: Callable[[float], float]
    reuse_key: Hashable | None = None


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


def emit_filter(builder, alias, positions=None, live=None, followed=False):
    """Append a run of the alias's AI filters in their chosen order.

    Args:
        builder: The graph builder.
        alias: The table.
        positions: The written positions to emit, in run order; None
            emits every ask in the chosen order.
        live: The documents the run reads; None reads the table.
        followed: Whether a classification or an extraction continues
            the run's pipeline, so the filters write the documents' KV.
    """
    stats = builder.operator_order.statistics.stats
    orders = builder.operator_order.filter_orders
    if positions is None:
        positions = orders[alias]
    surv = stats[alias].n_docs if live is None else live
    for i in positions:
        p = builder.operators.filters[alias][i]
        fid = (f"ai_filter:{alias}" if i == orders[alias][0]
               else f"ai_filter:{alias}:{i}")
        builder.nodes.append(AiFilter(
            node_id=fid, inputs=input_ports((builder.ids_src[alias],)), alias=alias,
            written_pos=i,
            question_tokens=question_tokens(
                builder.context.lowered(p.prompt), builder.model.canvas_tokens),
            preamble_tokens=builder.context.lowered(p.prompt).preamble_tokens,
            selectivity=p.selectivity, expected_docs=round(surv, 1),
            arena_writes=(len(orders[alias]) > 1 or followed
                          or alias in builder.pipelined),
            stop_key=builder.stop_keys.get(alias, ())))
        surv *= effective_selectivity(p.selectivity)
        builder.ids_src[alias] = PortRef(fid, f"ids:{alias}")
    builder.emit_applies(alias)

