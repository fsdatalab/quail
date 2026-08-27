"""Exact small-instance oracle for KV victim policy tests."""

from dataclasses import dataclass
from math import inf
from typing import Hashable, Iterable


@dataclass(frozen=True)
class Retained:
    """One evictable document prefix."""

    key: Hashable
    pages: int
    value: float


@dataclass(frozen=True)
class Victims:
    """The least expensive resident set that frees enough pages."""

    keys: tuple[Hashable, ...]
    pages: int
    value: float


@dataclass(frozen=True)
class _Path:
    key: Hashable
    previous: "_Path | None"


def minimum_loss_victims(
    residents: Iterable[Retained], pages_needed: int
) -> Victims | None:
    """Choose the minimum value set that frees at least pages_needed.

    The state is capped at pages_needed, so one admission uses at most
    len(residents) times pages_needed updates.
    """

    if pages_needed <= 0:
        return Victims((), 0, 0.0)
    entries = tuple(residents)
    if any(entry.pages <= 0 for entry in entries):
        raise ValueError("retained entries must use at least one page")
    if any(entry.value < 0 for entry in entries):
        raise ValueError("retained entry values must be nonnegative")

    values = [inf] * (pages_needed + 1)
    actual_pages = [0] * (pages_needed + 1)
    paths: list[_Path | None] = [None] * (pages_needed + 1)
    values[0] = 0.0

    for entry in entries:
        for have in range(pages_needed - 1, -1, -1):
            if values[have] == inf:
                continue
            reached = min(pages_needed, have + entry.pages)
            candidate_value = values[have] + entry.value
            candidate_pages = actual_pages[have] + entry.pages
            better = candidate_value < values[reached]
            tied = candidate_value == values[reached]
            if better or (tied and candidate_pages < actual_pages[reached]):
                values[reached] = candidate_value
                actual_pages[reached] = candidate_pages
                paths[reached] = _Path(entry.key, paths[have])

    if values[pages_needed] == inf:
        return None
    keys = []
    path = paths[pages_needed]
    while path is not None:
        keys.append(path.key)
        path = path.previous
    keys.reverse()
    return Victims(tuple(keys), actual_pages[pages_needed],
                   values[pages_needed])
