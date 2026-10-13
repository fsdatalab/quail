"""Choose predicate order from prepared costs, selectivities, and reuse keys."""

from dataclasses import dataclass
from typing import Callable, Hashable

from quail.logical import DEFAULT_SELECTIVITY


@dataclass(frozen=True)
class PredicateCost:
    """One predicate's costs and reusable result identity."""

    selectivity: float
    first: Callable[[float], float]
    later: Callable[[float], float]
    reuse_key: Hashable | None = None


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


def order_filters_indexed(predicates, rule: str, *, count: float = 1):
    """Choose an order using costs at each expected survivor count.

    Try each first predicate, then rank remaining predicates by their
    cost per expected rejection. A computed reuse key costs no more model
    work. Written order breaks ties; as_written bypasses the search.
    """
    indices = list(range(len(predicates)))
    if rule == "as_written" or len(indices) < 2:
        return indices
    costs = {}

    def cost(i, live, first):
        key = (i, live, first)
        if key not in costs:
            p = predicates[i]
            costs[key] = (p.first if first else p.later)(live)
        return costs[key]

    candidates = []
    for first in indices:
        ordered = [first]
        total = cost(first, count, True)
        live = count * predicates[first].selectivity
        scored = ({predicates[first].reuse_key}
                  if predicates[first].reuse_key is not None else set())
        remaining = [i for i in indices if i != first]
        while remaining:
            choices = []
            for i in remaining:
                p = predicates[i]
                seconds = (0.0 if p.reuse_key in scored else
                           cost(i, live if live > 0 else 1.0, False))
                rejected = 1.0 - p.selectivity
                rank = (seconds / rejected if rejected > 0 else
                        0.0 if seconds == 0 else float("inf"))
                choices.append((rank, i, seconds if live > 0 else 0.0))
            _, selected, seconds = min(choices)
            total += seconds
            ordered.append(selected)
            live *= predicates[selected].selectivity
            if predicates[selected].reuse_key is not None:
                scored.add(predicates[selected].reuse_key)
            remaining.remove(selected)
        candidates.append((total, first, ordered))
    return min(candidates)[2]
