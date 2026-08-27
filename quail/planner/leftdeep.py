"""Exact left deep join search over subsets of relations.

The state is (joined alias set, cached prefix alias set). Several
work records can reach one state; a record is kept unless another is
no larger in every `Work` category, so the frontier holds every
candidate that could win under any monotone time formula. The caller
supplies the extension function, which is where the cost model lives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from quail.planner.sol import Work


@dataclass(frozen=True)
class Extension:
    """One possible way to add one alias to a left deep plan."""

    work: Work
    cached: frozenset[str]
    steps: tuple[dict, ...]


@dataclass(frozen=True)
class Candidate:
    """One nondominated work record and the choices that produced it."""

    work: Work
    cached: frozenset[str]
    relation_order: tuple[str, ...]
    steps: tuple[dict, ...] = ()


@dataclass(frozen=True)
class SearchResult:
    """All final nondominated records and search accounting."""

    candidates: tuple[Candidate, ...]
    state_count: int
    generated_count: int


Extend = Callable[
    [frozenset[str], frozenset[str], str], Iterable[Extension]
]


def _insert_nondominated(
    frontier: list[Candidate], candidate: Candidate
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
    initially_cached: Iterable[str],
    base_work: Work,
    extend: Extend,
) -> SearchResult:
    """Run subset DP over relation subsets and cached prefix aliases."""

    all_aliases = frozenset(aliases)
    initial_cache = frozenset(initially_cached)
    states: dict[
        tuple[frozenset[str], frozenset[str]], list[Candidate]
    ] = {}
    for alias in aliases:
        states[(frozenset((alias,)), initial_cache)] = [
            Candidate(base_work, initial_cache, (alias,))
        ]

    generated = len(aliases)
    for size in range(1, len(aliases)):
        current_states = [
            (state, tuple(frontier))
            for state, frontier in states.items()
            if len(state[0]) == size
        ]
        for (relations, cached), candidates in current_states:
            for added in sorted(all_aliases - relations):
                extensions = tuple(extend(relations, cached, added))
                for candidate in candidates:
                    for extension in extensions:
                        generated += 1
                        next_relations = relations | {added}
                        next_candidate = Candidate(
                            work=candidate.work + extension.work,
                            cached=extension.cached,
                            relation_order=candidate.relation_order + (added,),
                            steps=candidate.steps + extension.steps,
                        )
                        frontier = states.setdefault(
                            (next_relations, extension.cached), []
                        )
                        _insert_nondominated(frontier, next_candidate)

    finals = tuple(
        candidate
        for (relations, _), frontier in states.items()
        if relations == all_aliases
        for candidate in frontier
    )
    return SearchResult(finals, len(states), generated)
