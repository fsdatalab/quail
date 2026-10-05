"""Order one table's model predicates by expected cost."""

from quail.cost.sol import unrounded_seconds
from quail.cost.work import ask, scan
from quail.logical import DEFAULT_SELECTIVITY, effective_selectivity, is_score
from quail.planner.score import score_spec
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
                          chunk_tokens: int, context=None, count: float = 1):
    """Return written positions of predicates in execution order.

    Try each first predicate, then rank remaining predicates by their
    cost per expected rejection. AI.IF charges the first predicate for
    its document and later predicates for their questions over KV.
    AI.SCORE supplies its prompt and prefix reuse cost at each survivor
    count. Repeated comparisons of a computed score cost no model work.
    Written order breaks ties; as_written bypasses the search.
    """
    indices = list(range(len(predicates)))
    if rule == "as_written" or len(indices) < 2:
        return indices
    token_parts = {}
    costs = {}

    def cost(i, live, first):
        key = (i, live, first)
        if key not in costs:
            p = predicates[i]
            if is_score(p.expression):
                spec, _ = score_spec(
                    p.prompt, name="", expected_inputs=live,
                    mean_tokens=prefix_tokens, context=context,
                    chunk_tokens=chunk_tokens,
                    token_parts_by_prompt=token_parts)
                costs[key] = spec.estimated_seconds
            else:
                costs[key] = live * filter_cost(
                    p, prefix_tokens, model, device, chunk_tokens, first=first)
        return costs[key]

    candidates = []
    for first in indices:
        ordered = [first]
        total = cost(first, count, True)
        live = count * effective_selectivity(predicates[first].selectivity)
        scored = ({predicates[first].prompt}
                  if is_score(predicates[first].expression) else set())
        remaining = [i for i in indices if i != first]
        while remaining:
            choices = []
            for i in remaining:
                p = predicates[i]
                seconds = (0.0 if p.prompt in scored else
                           cost(i, live if live > 0 else 1.0, False))
                rejected = 1.0 - effective_selectivity(p.selectivity)
                rank = (seconds / rejected if rejected > 0 else
                        0.0 if seconds == 0 else float("inf"))
                choices.append((rank, i, seconds if live > 0 else 0.0))
            _, selected, seconds = min(choices)
            total += seconds
            ordered.append(selected)
            live *= effective_selectivity(predicates[selected].selectivity)
            if is_score(predicates[selected].expression):
                scored.add(predicates[selected].prompt)
            remaining.remove(selected)
        candidates.append((total, first, ordered))
    return min(candidates)[2]


def order_filters(predicates, rule: str, *, prefix_tokens: float,
                  model: ModelSpec, device: DeviceSpec,
                  chunk_tokens: int):
    return [predicates[i] for i in order_filters_indexed(
        predicates, rule, prefix_tokens=prefix_tokens,
        model=model, device=device, chunk_tokens=chunk_tokens)]
