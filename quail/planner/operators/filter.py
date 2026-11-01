"""AI.IF over one table: the cost and work of its filters."""

from dataclasses import dataclass
from functools import partial
from operator import mul
from typing import Callable, Hashable

from quail.cost.filters import filter_cost
from quail.cost.packing import choose_stages
from quail.logical import (
    effective_selectivity,
    is_score,
)
from quail.physical import AiFilter, PortRef
from quail.physical.base import input_ports
from quail.planner.operators.common import question_tokens
from quail.planner.operators.score import score_statistics
from quail.planner.statistics.tables import summarize_alias


@dataclass(frozen=True)
class PredicateCost:
    """One predicate's costs and reusable result identity."""

    selectivity: float
    first: Callable[[float], float]
    later: Callable[[float], float]
    reuse_key: Hashable | None = None


def prepare_filter_costs(predicates, *, mean_tokens, model, device,
                         chunk_tokens, context=None,
                         lower=None) -> tuple[PredicateCost, ...]:
    """Prepare costs and reusable result identities for predicate ordering.

    Args:
        predicates: The model predicates of one table.
        mean_tokens: The preamble and mean document tokens.
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
                predicate.prompt, context, chunk_tokens, mean_tokens=mean_tokens)
            first = later = cost.seconds
            reuse_key = predicate.prompt
        else:
            q = question_tokens(lower(predicate.prompt), model.canvas_tokens)
            first = partial(mul, filter_cost(
                q, mean_tokens, model, device, chunk_tokens, first=True))
            later = partial(mul, filter_cost(
                q, mean_tokens, model, device, chunk_tokens, first=False))
            reuse_key = None
        prepared.append(PredicateCost(
            effective_selectivity(predicate.selectivity), first, later, reuse_key))
    return tuple(prepared)


def set_stages(builder, alias, positions, expected_rows) -> tuple[int, ...]:
    """Each filter's stage under set attention (the choose_packing step).

    Args:
        builder: The graph builder.
        alias: The table.
        positions: The written positions of its filters, in run order.
        expected_rows: The documents the first filter reads.
    """
    predicates = builder.operators.filters[alias]
    lowered = [builder.context.lowered(predicates[i].prompt) for i in positions]
    summary = summarize_alias(
        builder.document_tokens[alias], builder.model.sliding_window,
        memo=builder.context.memo)
    head = lowered[0].preamble_tokens
    return choose_stages(
        [question_tokens(prompt, builder.model.canvas_tokens) for prompt in lowered],
        [effective_selectivity(predicates[i].selectivity) for i in positions],
        expected_rows, head + summary.mean, head + summary.maximum,
        builder.model, builder.device,
        builder.operator_order.statistics.chunk, set_size=builder.model.set_size)


def emit_filter(builder, alias, positions, expected_rows, port, followed=False):
    """Append a run of the alias's AI filters in their chosen order.

    Args:
        builder: The graph builder.
        alias: The table.
        positions: The written positions to emit, in run order.
        expected_rows: The documents the run reads.
        port: The port the documents stand at before the run.
        followed: Whether a classification or an extraction continues
            the run's pipeline, so the filters write the documents' KV.

    Returns:
        Where the documents stand after the run.
    """
    orders = builder.operator_order.filter_orders
    stages = (set_stages(builder, alias, positions, expected_rows)
              if builder.model.attention_mask == "set" else (0,) * len(positions))
    surv = expected_rows
    for i, stage in zip(positions, stages):
        p = builder.operators.filters[alias][i]
        ask_id = (f"ai_filter:{alias}" if i == orders[alias][0]
               else f"ai_filter:{alias}:{i}")
        lowered = builder.context.lowered(p.prompt)
        builder.nodes.append(AiFilter(
            node_id=ask_id, inputs=input_ports((port,)), alias=alias,
            written_position=i,
            question_tokens=question_tokens(lowered, builder.model.canvas_tokens),
            preamble_tokens=lowered.preamble_tokens,
            document_head=tuple(lowered.preamble_token_ids),
            selectivity=p.selectivity, expected_rows=round(surv, 1),
            arena_writes=(builder.model.attention_mask != "set"
                          and (len(orders[alias]) > 1 or followed
                               or alias in builder.pipelined)),
            stop_key=builder.stop_keys.get(alias, ()), stage=stage))
        surv *= effective_selectivity(p.selectivity)
        port = PortRef(ask_id, f"rows:{alias}")
    return port

