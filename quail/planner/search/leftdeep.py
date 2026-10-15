"""Exact left deep join search, one join per step.

The state is the joined alias set, the joins applied so far, and a
caller supplied property: the planner's KV state, or the estimator's
cached prefixes. Several work records can reach one state; a record is
kept unless another is no larger in every `Work` category, so the
frontier holds every candidate that could win under any monotone time
formula. The caller prices one step, applying one join with one anchor
after a given set of joins; the search enumerates the orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Iterable, Sequence, TypeVar

from quail.cost.work import Work

StateProperty = TypeVar("StateProperty", bound=Hashable)
Step = TypeVar("Step")


@dataclass(frozen=True)
class Candidate(Generic[StateProperty, Step]):
    """One nondominated work record and the choices that produced it."""

    work: Work
    state_property: StateProperty
    relation_order: tuple[str, ...]
    steps: tuple[Step, ...] = ()


@dataclass(frozen=True)
class SearchResult(Generic[StateProperty, Step]):
    """All final nondominated records and search accounting."""

    candidates: tuple[Candidate[StateProperty, Step], ...]
    state_count: int
    generated_count: int


PriceStep = Callable[
    [int, frozenset[str], frozenset[int], StateProperty],
    Iterable[tuple[Work, StateProperty, Step]],
]


def insert_nondominated(frontier: list, candidate) -> bool:
    """Keep one record unless another is no larger in every category."""
    if any(existing.work.dominates(candidate.work) for existing in frontier):
        return False
    frontier[:] = [
        existing
        for existing in frontier
        if not candidate.work.dominates(existing.work)
    ]
    frontier.append(candidate)
    return True


def search_left_deep(
    aliases: Sequence[str],
    edges: Sequence[frozenset[str]],
    initial_property: StateProperty,
    base_work: Work,
    price_step: PriceStep[StateProperty, Step],
    *,
    seeds: Iterable[str] = (),
) -> SearchResult[StateProperty, Step]:
    """Run the subset DP over joined aliases, applied joins, and a property.

    Args:
        aliases: Every alias the joins read.
        edges: Per join, the aliases it reads.
        initial_property: The property of every starting state.
        base_work: Work every candidate starts with.
        price_step: Called with a join's index, the joined aliases, the
            joins applied before it, and the state property; yields one
            (work, next property, step record) per way to apply it, or
            nothing when it cannot be applied there.
        seeds: Aliases already joined by completed stages, so the search
            continues a partial plan; empty to start from each alias.

    Returns:
        The nondominated records that applied every join, with the
        search's state and record counts.
    """
    seeds = frozenset(seeds)
    all_aliases = frozenset(aliases) | seeds
    states: dict[tuple[frozenset[str], frozenset[int], StateProperty],
                 list[Candidate]] = {}
    starts = [seeds] if seeds else [frozenset((alias,)) for alias in aliases]
    for relations in starts:
        states[(relations, frozenset(), initial_property)] = [
            Candidate(base_work, initial_property, tuple(sorted(relations)))]
    generated = len(starts)
    for applied_count in range(len(edges)):
        current = [(key, tuple(frontier)) for key, frontier in states.items()
                   if len(key[1]) == applied_count]
        for (relations, applied, state_property), frontier in current:
            available = [
                index for index, ends in enumerate(edges)
                if index not in applied
                and (ends <= relations
                     or (ends & relations and len(ends - relations) == 1))]
            for index in available:
                added = tuple(sorted(edges[index] - relations))
                next_relations = relations | edges[index]
                next_applied = applied | {index}
                for work, next_property, step in price_step(
                        index, relations, applied, state_property):
                    target = states.setdefault(
                        (next_relations, next_applied, next_property), [])
                    for old in frontier:
                        generated += 1
                        insert_nondominated(target, Candidate(
                            old.work + work, next_property,
                            old.relation_order + added, old.steps + (step,)))
    finals = tuple(
        candidate
        for (relations, applied, _), frontier in states.items()
        if len(applied) == len(edges) and all_aliases <= relations
        for candidate in frontier)
    return SearchResult(finals, len(states), generated)
