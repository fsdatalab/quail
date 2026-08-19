"""The pinned CPU KV store: cross-query document KV, one pool per
container.

What it holds: document-prefix KV only, in the arena's dtype, keyed by
(content hash, document index). The kept question preamble is NOT
stored - it belongs to a query, not a document - so a warm query with
new questions still restores every document.

Layout: one pinned host tensor of token rows, each row = every
layer's K and V for that token (row width = kv_elements_per_token).
A document is one contiguous extent of rows, so save and restore are
ONE transfer per document staged through a GPU ring buffer on a side
stream - the plain path, no connector indirection (stock vLLM's
connector ran 10 of 55 GB/s; the committed batched path restored at
~15 GB/s effective end to end).

Eviction is a length threshold decided at plan time, never LRU: a
scanning query thrashes LRU; it cannot thrash a length cutoff. When
the pool is full or fragmented, save simply skips - the store is a
cache, correctness never depends on it.

ExtentAllocator is pure Python (CPU-tested); PinnedStore runs only
where torch and a GPU exist.
"""

import bisect


def alloc_with_reclaim(allocs, extents, tokens, keep_hash):
    """An extent for a new document, reclaiming idle datasets' space
    when the pool is full. Pure accounting, CPU-tested.

    The rule that keeps the store scan-proof: the running query's own
    corpus (keep_hash) is NEVER evicted - that is where LRU-style
    thrash lives. Other datasets' extents are dead capital while this
    dataset queries; they yield, shortest documents first (least
    recompute value lost), and re-store if their dataset returns.

    Returns (slab, offset) or None."""
    def try_alloc():
        for slab, alloc in enumerate(allocs):
            off = alloc.alloc(tokens)
            if off is not None:
                return slab, off
        return None

    got = try_alloc()
    if got is not None:
        return got
    victims = sorted((k for k in extents if k[0] != keep_hash),
                     key=lambda k: extents[k][2])
    for k in victims:
        slab, off, tk = extents.pop(k)
        allocs[slab].free(off, tk)
        got = try_alloc()
        if got is not None:
            return got
    return None


class ExtentAllocator:
    """First-fit contiguous extents with free-list coalescing."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.free_list = [(0, capacity)]     # (offset, length), sorted
        self.used = 0

    def alloc(self, n: int):
        """Offset of a free extent of n rows, or None."""
        if n <= 0:
            raise ValueError("extent size must be positive")
        for i, (off, length) in enumerate(self.free_list):
            if length >= n:
                if length == n:
                    self.free_list.pop(i)
                else:
                    self.free_list[i] = (off + n, length - n)
                self.used += n
                return off
        return None

    def free(self, off: int, n: int) -> None:
        """Return an extent, coalescing with adjacent free extents."""
        i = bisect.bisect_left(self.free_list, (off, 0))
        # merge with the previous and/or next neighbor when contiguous
        start, end = off, off + n
        if i > 0 and self.free_list[i - 1][0] + self.free_list[i - 1][1] \
                == start:
            start = self.free_list[i - 1][0]
            self.free_list.pop(i - 1)
            i -= 1
        if i < len(self.free_list) and self.free_list[i][0] == end:
            end = self.free_list[i][0] + self.free_list[i][1]
            self.free_list.pop(i)
        self.free_list.insert(i, (start, end - start))
        self.used -= n

    @property
    def free_rows(self) -> int:
        return self.capacity - self.used


class PinnedStore:
    """The pinned host pool plus the staged transfers.

    The pool is a list of 8 GiB pinned slabs, not one slab: a single
    280 GB cudaHostAlloc failed with cudaErrorMemoryAllocation where
    the committed persist run's chunked pool pinned 242 GB fine. A
    document's extent never spans slabs (a document is at most ~2 GB
    of KV even at 4x document length)."""

    STAGING_SLOTS = 4
    SLAB_BYTES = 8 << 30

    def __init__(self, capacity_tokens: int, n_layers: int, n_kv: int,
                 d_head: int, max_doc_tokens: int, dtype=None):
        import torch
        self.torch = torch
        dtype = dtype or torch.bfloat16
        self.n_layers = n_layers
        self.kv_width = n_kv * d_head
        self.row_width = n_layers * 2 * self.kv_width
        row_bytes = self.row_width * torch.tensor([], dtype=dtype).element_size()
        self.slab_tokens = self.SLAB_BYTES // row_bytes
        n_slabs = max(1, -(-capacity_tokens // self.slab_tokens))
        self.pools = [torch.empty((self.slab_tokens, self.row_width),
                                  dtype=dtype, pin_memory=True)
                      for _ in range(n_slabs)]
        self.allocs = [ExtentAllocator(self.slab_tokens)
                       for _ in range(n_slabs)]
        self.extents = {}       # key -> (slab, offset, tokens)
        self.stream = torch.cuda.Stream()
        self._staging = [torch.empty((max_doc_tokens, self.row_width),
                                     dtype=dtype, device="cuda")
                         for _ in range(self.STAGING_SLOTS)]
        self._staging_events = [torch.cuda.Event()
                                for _ in range(self.STAGING_SLOTS)]
        for e in self._staging_events:
            e.record()          # all slots start available
        self._next_slot = 0

    def __contains__(self, key) -> bool:
        return key in self.extents

    @property
    def stored_tokens(self) -> int:
        return sum(a.used for a in self.allocs)


    def _slot(self):
        i = self._next_slot
        self._next_slot = (i + 1) % self.STAGING_SLOTS
        self._staging_events[i].synchronize()   # previous use finished
        return i

    def _k_cols(self, layer):
        a = layer * 2 * self.kv_width
        return a, a + self.kv_width

    def _v_cols(self, layer):
        a = layer * 2 * self.kv_width + self.kv_width
        return a, a + self.kv_width

    def save(self, key, arena, arena_key, tokens, after_event):
        """Copy a document's first `tokens` arena rows to the pool.

        Runs on the side stream after `after_event` (the compute that
        wrote the pages). Returns the completion event - the caller
        must not free the document's pages before it fires - or None
        when the pool has no room (the cache skips, never evicts a
        longer document for a shorter one)."""
        torch = self.torch
        if key in self.extents or tokens > self._staging[0].shape[0]:
            return None
        if tokens > self.slab_tokens:
            return None
        got = alloc_with_reclaim(self.allocs, self.extents, tokens,
                                 keep_hash=key[0])
        if got is None:
            return None
        slab, off = got
        self.extents[key] = (slab, off, tokens)
        rows = arena.rows_gpu(arena_key)[:tokens]
        slot = self._slot()
        stage = self._staging[slot]
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(after_event)
            for layer in range(self.n_layers):
                k0, k1 = self._k_cols(layer)
                v0, v1 = self._v_cols(layer)
                stage[:tokens, k0:k1].copy_(
                    arena.k[layer].index_select(0, rows)
                    .view(tokens, self.kv_width))
                stage[:tokens, v0:v1].copy_(
                    arena.v[layer].index_select(0, rows)
                    .view(tokens, self.kv_width))
            self.pools[slab][off:off + tokens].copy_(stage[:tokens],
                                                     non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.stream)
            self._staging_events[slot].record(self.stream)
        return done

    def load(self, key, arena, arena_key):
        """Copy a stored document into its (already allocated) arena
        pages. Returns the completion event; the chunk that reads the
        pages must wait on it."""
        torch = self.torch
        slab, off, tokens = self.extents[key]
        rows = arena.rows_gpu(arena_key)[:tokens]
        slot = self._slot()
        stage = self._staging[slot]
        n_kv = arena.k[0].shape[-2]
        d = arena.k[0].shape[-1]
        with torch.cuda.stream(self.stream):
            stage[:tokens].copy_(self.pools[slab][off:off + tokens],
                                 non_blocking=True)
            for layer in range(self.n_layers):
                k0, k1 = self._k_cols(layer)
                v0, v1 = self._v_cols(layer)
                arena.k[layer].index_copy_(
                    0, rows, stage[:tokens, k0:k1].view(tokens, n_kv, d))
                arena.v[layer].index_copy_(
                    0, rows, stage[:tokens, v0:v1].view(tokens, n_kv, d))
            done = torch.cuda.Event()
            done.record(self.stream)
            self._staging_events[slot].record(self.stream)
        return done

    def flush(self) -> None:
        """Drop everything (the benchmark's cold pass is a store
        flush, not a restart)."""
        self.torch.cuda.synchronize()
        self.extents.clear()
        self.allocs = [ExtentAllocator(self.slab_tokens)
                       for _ in self.pools]
