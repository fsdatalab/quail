"""Exact left deep join search over subsets of relations.

The state is a joined alias set and a caller supplied physical
property. Several work records can reach one state; a record is kept
unless another is no larger in every `Work` category, so the frontier
holds every candidate that could win under any monotone time formula.
The caller supplies the extension function and the physical property.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Iterable, Sequence, TypeVar

from quail.cost.work import Work

StateProperty = TypeVar("StateProperty", bound=Hashable)
Step = TypeVar("Step")


@dataclass(frozen=True)
class Extension(Generic[StateProperty, Step]):
    """One possible way to add one alias to a left deep plan."""

    work: Work
    state_property: StateProperty
    steps: tuple[Step, ...]


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


Extend = Callable[
    [frozenset[str], StateProperty, str],
    Iterable[Extension[StateProperty, Step]],
]


def _insert_nondominated(
    frontier: list[Candidate[StateProperty, Step]],
    candidate: Candidate[StateProperty, Step],
) -> bool:
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


def optimize_left_deep(
    aliases: Sequence[str],
    initial_property: StateProperty,
    base_work: Work,
    extend: Extend[StateProperty, Step],
) -> SearchResult[StateProperty, Step]:
    """Run subset DP over relation subsets and a physical property."""
    all_aliases = frozenset(aliases)
    states: dict[
        tuple[frozenset[str], StateProperty],
        list[Candidate[StateProperty, Step]],
    ] = {}
    for alias in aliases:
        states[(frozenset((alias,)), initial_property)] = [
            Candidate(base_work, initial_property, (alias,))
        ]

    generated = len(aliases)
    for size in range(1, len(aliases)):
        current_states = [
            (state, tuple(frontier))
            for state, frontier in states.items()
            if len(state[0]) == size
        ]
        for (relations, state_property), candidates in current_states:
            for added in sorted(all_aliases - relations):
                extensions = tuple(extend(relations, state_property, added))
                for candidate in candidates:
                    for extension in extensions:
                        generated += 1
                        next_relations = relations | {added}
                        next_candidate = Candidate(
                            work=candidate.work + extension.work,
                            state_property=extension.state_property,
                            relation_order=candidate.relation_order + (added,),
                            steps=candidate.steps + extension.steps,
                        )
                        frontier = states.setdefault(
                            (next_relations, extension.state_property), []
                        )
                        _insert_nondominated(frontier, next_candidate)

    finals = tuple(
        candidate
        for (relations, _), frontier in states.items()
        if relations == all_aliases
        for candidate in frontier
    )
    return SearchResult(finals, len(states), generated)
