"""Exact left deep join search for the QUAIL-B speed of light model."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

from quail.planner.work import Work


@dataclass(frozen=True)
class PairRelation:
    """The exact passing row pairs for one binary join predicate."""

    left: str
    right: str
    pairs: frozenset[tuple[int, int]]


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


def exact_live_rows(
    base_rows: Mapping[str, Sequence[int]],
    relations: Sequence[PairRelation],
) -> dict[str, tuple[int, ...]]:
    """Return rows that occur in at least one exact satisfying assignment.

    Tree components use repeated semijoin reduction. Cyclic components use
    exact backtracking because pairwise reduction alone is not exact there.
    """

    aliases = tuple(base_rows)
    combined = _combine_parallel_relations(relations)
    neighbors: dict[str, set[str]] = defaultdict(set)
    for relation in combined:
        neighbors[relation.left].add(relation.right)
        neighbors[relation.right].add(relation.left)

    components: list[set[str]] = []
    unseen = set(aliases)
    while unseen:
        root = min(unseen)
        component = {root}
        stack = [root]
        unseen.remove(root)
        while stack:
            alias = stack.pop()
            for other in neighbors[alias] & unseen:
                unseen.remove(other)
                component.add(other)
                stack.append(other)
        components.append(component)

    output: dict[str, tuple[int, ...]] = {}
    for component in components:
        component_relations = [
            relation
            for relation in combined
            if relation.left in component and relation.right in component
        ]
        if len(component_relations) <= len(component) - 1:
            live = _forest_live_rows(base_rows, component_relations, component)
        else:
            live = _cyclic_live_rows(base_rows, component_relations, component)
        output.update({alias: tuple(sorted(rows)) for alias, rows in live.items()})
    return output


def _combine_parallel_relations(
    relations: Sequence[PairRelation],
) -> tuple[PairRelation, ...]:
    oriented: dict[tuple[str, str], set[tuple[int, int]]] = {}
    for relation in relations:
        key = tuple(sorted((relation.left, relation.right)))
        pairs = set(relation.pairs)
        if (relation.left, relation.right) != key:
            pairs = {(right, left) for left, right in pairs}
        if key in oriented:
            oriented[key].intersection_update(pairs)
        else:
            oriented[key] = pairs
    return tuple(
        PairRelation(left, right, frozenset(pairs))
        for (left, right), pairs in sorted(oriented.items())
    )


def _adjacency(
    relation: PairRelation,
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    left_to_right: dict[int, set[int]] = defaultdict(set)
    right_to_left: dict[int, set[int]] = defaultdict(set)
    for left, right in relation.pairs:
        left_to_right[left].add(right)
        right_to_left[right].add(left)
    return left_to_right, right_to_left


def _forest_live_rows(
    base_rows: Mapping[str, Sequence[int]],
    relations: Sequence[PairRelation],
    aliases: set[str],
) -> dict[str, set[int]]:
    live = {alias: set(base_rows[alias]) for alias in aliases}
    adjacency = {
        (relation.left, relation.right): _adjacency(relation)
        for relation in relations
    }
    changed = True
    while changed:
        changed = False
        for relation in relations:
            left_to_right, right_to_left = adjacency[
                (relation.left, relation.right)
            ]
            left = {
                row
                for row in live[relation.left]
                if left_to_right.get(row, set()) & live[relation.right]
            }
            right = {
                row
                for row in live[relation.right]
                if right_to_left.get(row, set()) & live[relation.left]
            }
            if left != live[relation.left]:
                live[relation.left] = left
                changed = True
            if right != live[relation.right]:
                live[relation.right] = right
                changed = True
    return live


def _cyclic_live_rows(
    base_rows: Mapping[str, Sequence[int]],
    relations: Sequence[PairRelation],
    aliases: set[str],
) -> dict[str, set[int]]:
    pair_sets: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    neighbors: dict[str, set[str]] = defaultdict(set)
    for relation in relations:
        pair_sets[(relation.left, relation.right)] = relation.pairs
        pair_sets[(relation.right, relation.left)] = frozenset(
            (right, left) for left, right in relation.pairs
        )
        neighbors[relation.left].add(relation.right)
        neighbors[relation.right].add(relation.left)

    found = {alias: set() for alias in aliases}

    def visit(assignment: dict[str, int]) -> None:
        if len(assignment) == len(aliases):
            for alias, row in assignment.items():
                found[alias].add(row)
            return
        remaining = aliases - assignment.keys()
        alias = min(
            remaining,
            key=lambda name: (
                -len(neighbors[name] & assignment.keys()),
                len(base_rows[name]),
                name,
            ),
        )
        for row in base_rows[alias]:
            if all(
                (row, assignment[other]) in pair_sets[(alias, other)]
                for other in neighbors[alias] & assignment.keys()
            ):
                assignment[alias] = row
                visit(assignment)
                del assignment[alias]

    visit({})
    return found
