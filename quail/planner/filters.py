"""Order one table's AI.IF predicates by expected cost."""

from quail.cost.sol import unrounded_seconds
from quail.cost.work import ask, scan
from quail.logical import DEFAULT_SELECTIVITY, effective_selectivity
from quail.planner.statistics import question_tokens
from quail.specs import DeviceSpec, ModelSpec


def default_order_rule(filters, joins) -> tuple[str, str]:
    """Return (rule, source).

    Always 'by_cost'; a predicate without a selectivity is priced with
    DEFAULT_SELECTIVITY. Pass order="as_written" to keep written order.
    """
    missing = any(p.selectivity is None for fs in filters.values()
                  for p in fs) or any(j.selectivity is None for j in joins)
    if missing:
        return "by_cost", (
            "default: by cost, with selectivity "
            f"{DEFAULT_SELECTIVITY:g} for predicates without one")
    return "by_cost", "default: by cost"


def filter_cost(predicate, prefix_tokens: float, model: ModelSpec,
                device: DeviceSpec, chunk_tokens: int, *, first: bool) -> float:
    """Return ideal time for one filter evaluation."""
    operation = scan if first else ask
    work = operation(prefix_tokens,
                     question_tokens(predicate.prompt, model.canvas_tokens),
                     window=model.sliding_window)
    return unrounded_seconds(work, model, device, chunk_tokens)


def order_filters_indexed(predicates, rule: str, *, prefix_tokens: float,
                          model: ModelSpec, device: DeviceSpec,
                          chunk_tokens: int):
    """Return written positions of predicates in execution order.

    'by_cost' minimizes ideal expected time. It sorts the asks once,
    then prices each predicate as the first scan. Written order breaks
    ties.
    """
    idx = list(range(len(predicates)))
    if rule == "as_written":
        return idx

    def selectivity(i):
        return effective_selectivity(predicates[i].selectivity)

    ask_costs = [
        filter_cost(p, prefix_tokens, model, device, chunk_tokens,
                    first=False)
        for p in predicates
    ]
    scan_costs = [
        filter_cost(p, prefix_tokens, model, device, chunk_tokens,
                    first=True)
        for p in predicates
    ]

    def score(i):
        killed = 1.0 - selectivity(i)
        if killed <= 0:
            return float("inf")
        return ask_costs[i] / killed

    ask_order = sorted(idx, key=score)
    prefix_live = [1.0]
    prefix_cost = [0.0]
    for i in ask_order:
        prefix_cost.append(
            prefix_cost[-1] + prefix_live[-1] * ask_costs[i])
        prefix_live.append(prefix_live[-1] * selectivity(i))

    total_ask_cost = prefix_cost[-1]
    candidates = []
    for position, first in enumerate(ask_order):
        expected = (
            scan_costs[first]
            + selectivity(first) * prefix_cost[position]
            + total_ask_cost - prefix_cost[position + 1]
        )
        candidates.append((expected, first))

    first = min(candidates)[1]
    return [first, *(i for i in ask_order if i != first)]


def order_filters(predicates, rule: str, *, prefix_tokens: float,
                  model: ModelSpec, device: DeviceSpec,
                  chunk_tokens: int):
    return [predicates[i] for i in order_filters_indexed(
        predicates, rule, prefix_tokens=prefix_tokens,
        model=model, device=device, chunk_tokens=chunk_tokens)]
