"""Retention policy for KV kept across operators."""

import heapq


class RetainedPool:
    """Fixed-capacity pool of document prefixes kept for a later operator.

    Keeps every offer while capacity lasts. Once full, the lowest
    value-per-page residents make room only when the newcomer's value
    strictly exceeds theirs combined, so total retained value only
    rises and equal value never swaps.
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

