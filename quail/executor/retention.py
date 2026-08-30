"""Retention policy for KV kept across operators.

RetainedPool is the runtime policy the filter loop uses; the oracle
below it exists only for victim-policy tests.
"""

import heapq
from dataclasses import dataclass
from math import inf
from typing import Hashable, Iterable


class RetainedPool:
    """Fixed-capacity pool of document prefixes kept for a later operator.

    The capacity is what the arena holds beside the filter loop's
    working reservation, so retention can never starve admission.
    While the pool has room, every offered prefix is kept. Once full,
    the residents with the least saved recompute per page are the
    candidates to make room, and the newcomer replaces them only when
    its value strictly exceeds what they lose together: total retained
    value only rises, and equal value never swaps. Saved recompute per
    page rises with prefix length, so longer documents displace
    shorter ones.
    """

    def __init__(self, cap_pages: int):
        if cap_pages < 0:
            raise ValueError("cap_pages must be nonnegative")
        self.cap_pages = cap_pages
        self.pages = 0
        self.value = 0.0
        self._entries = {}    # key -> (pages, value)
        self._heap = []       # (value per page, seq, key); one live
        #                       record per key - victims leave the heap
        #                       when popped, rejected pops go back
        self._seq = 0

    def _push(self, key, pages: int, value: float):
        self._entries[key] = (pages, value)
        self._seq += 1
        heapq.heappush(self._heap, (value / pages, self._seq, key))

    def offer(self, key, pages: int, value: float):
        """Consider one prefix for retention.

        Args:
            key: The prefix's identity, unique per offer.
            pages: KV pages the retained prefix occupies.
            value: Seconds of recompute its next use saves.

        Returns:
            (kept, victims): whether the prefix is retained, and the
            evicted keys the caller must free before retaining it.
        """
        if pages <= 0:
            raise ValueError("pages must be positive")
        if value < 0:
            raise ValueError("value must be nonnegative")
        if key in self._entries:
            raise KeyError(f"{key!r} already retained")
        if self.pages + pages <= self.cap_pages:
            self._push(key, pages, value)
            self.pages += pages
            self.value += value
            return True, ()
        if pages > self.cap_pages:
            return False, ()
        popped = []
        freed = 0
        loss = 0.0
        while (self.pages - freed + pages > self.cap_pages
               and loss < value and self._heap):
            record = heapq.heappop(self._heap)
            entry = self._entries.get(record[2])
            if entry is None:
                continue        # evicted through discard, stale record
            victim_pages, victim_value = entry
            popped.append(record)
            freed += victim_pages
            loss += victim_value
        if self.pages - freed + pages <= self.cap_pages and loss < value:
            for _, _, k in popped:
                del self._entries[k]
            self.pages -= freed
            self.value -= loss
            self._push(key, pages, value)
            self.pages += pages
            self.value += value
            return True, tuple(k for _, _, k in popped)
        for record in popped:
            heapq.heappush(self._heap, record)
        return False, ()

    def discard(self, key) -> None:
        """Forget a prefix evicted outside the pool's own policy."""
        entry = self._entries.pop(key, None)
        if entry is not None:
            self.pages -= entry[0]
            self.value -= entry[1]

    def __contains__(self, key):
        return key in self._entries

    def __len__(self):
        return len(self._entries)


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
