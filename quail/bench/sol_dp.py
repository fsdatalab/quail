"""Exact join results used by the QUAIL-B speed of light model."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PairRelation:
    """The exact passing row pairs for one binary join predicate."""

    left: str
    right: str
    pairs: frozenset[tuple[int, int]]


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
