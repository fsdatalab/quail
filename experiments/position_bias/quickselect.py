"""Pairwise top-k by randomized quickselect, as LOTUS runs it.

Each round picks a random pivot and compares every other candidate
with it in one batch. Candidates that beat the pivot form the better
side. If the better side holds at least k records, the search
continues there; otherwise it keeps the better side and the pivot and
looks for the rest of the k records among the worse side.

A comparator takes a batch of ordered (first, second) id pairs and
returns, for each, the probability that the first record wins. A
decision rule turns those probabilities into wins.
"""

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

Compare = Callable[[Sequence[tuple[str, str]]], list[float]]

RULES = ("raw", "calibrated", "both_orders")


@dataclass
class Selection:
    """One quickselect run's answer and what it cost.

    Attributes:
        top: The k selected ids, in no particular order.
        num_rounds: Pivot rounds.
        num_comparisons: Ordered pairs sent to the comparator.
    """

    top: list[str]
    num_rounds: int = 0
    num_comparisons: int = 0
    rounds: list[dict] = field(default_factory=list)


def candidate_wins(compare: Compare, pivot: str, candidates: Sequence[str],
                   rule: str, bias: float = 0.0) -> tuple[list[bool], int]:
    """Decide which candidates beat the pivot.

    Args:
        compare: The comparator.
        pivot: The pivot id.
        candidates: Ids compared with the pivot.
        rule: "raw" puts the candidate first and reads p > 0.5.
            "calibrated" does the same against 0.5 + bias / 2, which
            cancels a constant pull toward the first slot.
            "both_orders" asks both orders and averages
            p(candidate, pivot) with 1 - p(pivot, candidate).
        bias: The comparator's mean pull toward the first slot, for
            "calibrated".

    Returns:
        One win flag per candidate, and the number of ordered pairs sent.
    """
    forward = [(candidate, pivot) for candidate in candidates]
    probabilities = compare(forward)
    if rule == "raw":
        return [p > 0.5 for p in probabilities], len(forward)
    if rule == "calibrated":
        return [p > 0.5 + bias / 2 for p in probabilities], len(forward)
    if rule == "both_orders":
        backward = compare([(pivot, candidate) for candidate in candidates])
        return ([(p + 1 - q) / 2 > 0.5 for p, q in zip(probabilities, backward)],
                2 * len(forward))
    raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")


def quickselect(ids: Sequence[str], k: int, compare: Compare, *, rule: str,
                bias: float = 0.0, seed: int = 0) -> Selection:
    """Return k ids that the comparator ranks highest.

    Args:
        ids: The candidate ids.
        k: How many to return.
        compare: The comparator.
        rule: The decision rule; see `candidate_wins`.
        bias: The first-slot bias the "calibrated" rule cancels.
        seed: Seeds the pivot choices.
    """
    rng = random.Random(seed)
    selection = Selection(top=[])
    remaining = list(ids)
    needed = k
    while needed > 0:
        if len(remaining) <= needed:
            selection.top.extend(remaining)
            break
        pivot = rng.choice(remaining)
        candidates = [key for key in remaining if key != pivot]
        wins, sent = candidate_wins(compare, pivot, candidates, rule, bias)
        selection.num_rounds += 1
        selection.num_comparisons += sent
        better = [key for key, won in zip(candidates, wins) if won]
        worse = [key for key, won in zip(candidates, wins) if not won]
        selection.rounds.append({"pivot": pivot, "num_candidates": len(candidates),
                                 "num_better": len(better)})
        if len(better) >= needed:
            remaining = better
        else:
            selection.top.extend(better + [pivot])
            needed -= len(better) + 1
            remaining = worse
    return selection


def precision(top: Sequence[str], labels: dict[str, bool]) -> float:
    """The share of the returned ids that are relevant."""
    return sum(labels[key] for key in top) / len(top) if top else 0.0
