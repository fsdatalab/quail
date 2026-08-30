"""Retention policy for KV kept across operators."""

import heapq


class RetainedPool:
    """Fixed-capacity pool of document prefixes kept for a later operator.

    Keeps every offer while capacity lasts. Once full, residents with
    the fewest prefix tokens per page make room only when the newcomer's
    prefix tokens strictly exceed theirs combined. Total retained prefix
    tokens only rise, and equal token counts never swap.
    """

    def __init__(self, cap_pages: int):
        if cap_pages < 0:
            raise ValueError("cap_pages must be nonnegative")
        self.cap_pages = cap_pages
        self.pages = 0
        self.prefix_tokens = 0
        self._entries = {}    # key -> (pages, prefix tokens)
        self._heap = []       # (prefix tokens per page, seq, key)
        #                       record per key - victims leave the heap
        #                       when popped, rejected pops go back
        self._seq = 0

    def _push(self, key, pages: int, prefix_tokens: int):
        self._entries[key] = (pages, prefix_tokens)
        self._seq += 1
        heapq.heappush(
            self._heap, (prefix_tokens / pages, self._seq, key))

    def offer(self, key, pages: int, prefix_tokens: int):
        """Consider one prefix for retention.

        Args:
            key: The prefix's identity, unique per offer.
            pages: KV pages the retained prefix occupies.
            prefix_tokens: Reusable tokens in the prefix.

        Returns:
            (kept, victims): whether the prefix is retained, and the
            evicted keys the caller must free before retaining it.
        """
        if pages <= 0:
            raise ValueError("pages must be positive")
        if prefix_tokens <= 0:
            raise ValueError("prefix_tokens must be positive")
        if key in self._entries:
            raise KeyError(f"{key!r} already retained")
        if self.pages + pages <= self.cap_pages:
            self._push(key, pages, prefix_tokens)
            self.pages += pages
            self.prefix_tokens += prefix_tokens
            return True, ()
        if pages > self.cap_pages:
            return False, ()
        popped = []
        freed = 0
        lost_prefix_tokens = 0
        while (self.pages - freed + pages > self.cap_pages
               and lost_prefix_tokens < prefix_tokens and self._heap):
            record = heapq.heappop(self._heap)
            entry = self._entries.get(record[2])
            if entry is None:
                continue        # evicted through discard, stale record
            victim_pages, victim_prefix_tokens = entry
            popped.append(record)
            freed += victim_pages
            lost_prefix_tokens += victim_prefix_tokens
        if (self.pages - freed + pages <= self.cap_pages
                and lost_prefix_tokens < prefix_tokens):
            for _, _, k in popped:
                del self._entries[k]
            self.pages -= freed
            self.prefix_tokens -= lost_prefix_tokens
            self._push(key, pages, prefix_tokens)
            self.pages += pages
            self.prefix_tokens += prefix_tokens
            return True, tuple(k for _, _, k in popped)
        for record in popped:
            heapq.heappush(self._heap, record)
        return False, ()

    def discard(self, key) -> None:
        """Forget a prefix evicted outside the pool's own policy."""
        entry = self._entries.pop(key, None)
        if entry is not None:
            self.pages -= entry[0]
            self.prefix_tokens -= entry[1]

    def __contains__(self, key):
        return key in self._entries

    def __len__(self):
        return len(self._entries)
