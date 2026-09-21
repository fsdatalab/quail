"""Paged KV arena: preallocated per-layer buffers in fixed-size pages.

Pages are handed out from a free list and reached through block tables.

PageArena is the accounting (pure Python, CPU-tested); KVArena is the
tensor backing and runs only where torch and a GPU exist.

KVArena is the whole surface the chunk loop, the packer, the attention
paths, and the operator runtimes use. Everything they need is a method
or property on it; nothing outside this module reads the accounting
object or the K and V tensors directly. A replacement arena implements
the same methods:

- allocation: alloc, activate, alloc_temporary, free_key, pin, grow
- retention: retain, evict_retained, evict_key, configure_retention,
  retention_cap_pages, retained_keys, retained_pages,
  retained_prefix_tokens
- residency: is_resident, resident_keys, owned_pages, capacity_rows,
  pages_needed, page_cost, held_cost, page_tokens, n_pages, free_pages
- the sliding window: has_sliding, origin, sliding_start, trim_window,
  owned_sliding_pages, capacity_rows_sliding, n_sliding_pages, resize
- attention inputs: layer_kv, paged_kv, block_table, block_table_rows
- counters: reset_stats, evicted_keys, evicted_pages,
  evicted_prefix_tokens
"""

import heapq

import numpy as np

from quail.progress import answer_sink


class PageArena:
    """Page accounting: a free list and per-document page lists."""

    def __init__(self, n_pages: int, page_tokens: int):
        self.n_pages = n_pages
        self.page_tokens = page_tokens
        self.free = list(range(n_pages - 1, -1, -1))    # stack
        self.owned = {}       # key -> list of page ids
        self.tokens = {}      # key -> resident token count
        self.pinned = set()    # current operators depend on these keys
        self._retained_sizes = {}
        self._retained_pages = 0
        self.retained = {}     # key -> reusable prefix tokens
        self._retained_heap = []
        self._retained_versions = {}
        self._retention_version = 0
        self.retention_policy = None
        self.retention_cap_pages = None

    def pages_needed(self, tokens: int) -> int:
        return -(-tokens // self.page_tokens)

    def alloc(self, key, tokens: int, capacity_tokens: int | None = None):
        """The document's pages, or None when the free list is short.

        The admission scheduler treats None as "wait".
        """
        if key in self.owned:
            raise KeyError(f"{key!r} already resident")
        capacity_tokens = tokens if capacity_tokens is None else capacity_tokens
        if capacity_tokens < tokens:
            raise ValueError("capacity_tokens must cover logical tokens")
        need = self.pages_needed(capacity_tokens)
        if need > len(self.free):
            return None
        pages = [self.free.pop() for _ in range(need)]
        self.owned[key] = pages
        self.tokens[key] = tokens
        return pages

    def free_key(self, key) -> int:
        """Return a document's pages to the free list."""
        pages = self.owned.pop(key)
        self.tokens.pop(key)
        self.pinned.discard(key)
        self._forget_retained(key)
        self.free.extend(pages)
        return len(pages)

    def pin(self, key) -> None:
        """Protect a resident key while an operator uses it."""
        if key not in self.owned:
            raise KeyError(key)
        self._forget_retained(key)
        self.pinned.add(key)

    def _forget_retained(self, key):
        self._retained_pages -= self._retained_sizes.pop(key, 0)
        self.retained.pop(key, None)
        self._retained_versions.pop(key, None)

    def retain(self, key, priority=None) -> None:
        """Make a resident key evictable after its current use."""
        if key not in self.owned:
            raise KeyError(key)
        self.pinned.discard(key)
        prefix_tokens = self.tokens[key]
        self._forget_retained(key)
        self.retained[key] = prefix_tokens
        self._retained_sizes[key] = len(self.owned[key])
        self._retained_pages += len(self.owned[key])
        self._retention_version += 1
        version = self._retention_version
        self._retained_versions[key] = version
        pages = len(self.owned[key])
        if priority is None:
            priority = (self.retention_policy.priority(key, prefix_tokens, pages)
                        if self.retention_policy else (prefix_tokens / pages,))
        heapq.heappush(
            self._retained_heap,
            (priority, version, key, pages, prefix_tokens),
        )

    def configure_retention(self, policy, cap_pages: int) -> None:
        """Update priorities without making active prefixes evictable."""
        if not 0 <= cap_pages <= self.n_pages:
            raise ValueError("retention capacity must fit inside the arena")
        self.retention_policy = policy
        self.retention_cap_pages = cap_pages
        keys = list(self.retained)
        self._retained_heap.clear()
        for key in keys:
            self.retain(key)

    def pop_retained_victim(self):
        """Remove the retained prefix with the lowest planned reuse priority."""
        while self._retained_heap:
            _, version, key, pages, prefix_tokens = heapq.heappop(
                self._retained_heap)
            if self._retained_versions.get(key) != version:
                continue
            self._forget_retained(key)
            return key, pages, prefix_tokens
        return None

    def rewind(self, key, tokens: int) -> int:
        """Keep the first tokens and return unused trailing pages."""
        if key not in self.owned:
            raise KeyError(key)
        if tokens < 0 or tokens > len(self.owned[key]) * self.page_tokens:
            raise ValueError("rewind tokens exceed the resident capacity")
        keep = self.pages_needed(tokens)
        released = self.owned[key][keep:]
        self.owned[key] = self.owned[key][:keep]
        self.tokens[key] = tokens
        self.free.extend(released)
        if key in self.retained:
            self.retain(key)
        return len(released)

    def grow(self, key, capacity_tokens: int) -> int | None:
        """Add pages for capacity_tokens without changing logical tokens."""
        if key not in self.owned:
            raise KeyError(key)
        need = self.pages_needed(capacity_tokens) - len(self.owned[key])
        if need <= 0:
            return 0
        if need > len(self.free):
            return None
        self.owned[key].extend(self.free.pop() for _ in range(need))
        return need

    def row_indices(self, key, tokens=None):
        """Flat row positions of the document's tokens inside a pool.

        The pool is viewed as (n_pages * page_tokens, ...): page p holds
        rows p*page_tokens .. p*page_tokens + page_tokens.
        """
        rows = []
        left = self.tokens[key] if tokens is None else tokens
        for p in self.owned[key]:
            take = min(left, self.page_tokens)
            base = p * self.page_tokens
            rows.extend(range(base, base + take))
            left -= take
        return rows

    @property
    def free_pages(self) -> int:
        return len(self.free)

    @property
    def resident_tokens(self) -> int:
        return sum(self.tokens.values())

    @property
    def retained_pages(self) -> int:
        return self._retained_pages

    @property
    def retained_prefix_tokens(self) -> int:
        return sum(self.tokens[key] for key in self.retained)


def _evict_event(key, prefix_tokens):
    """The answer-sink entry for one evicted document prefix.

    Returns None when the key is not an alias and a document index.
    """
    if not (isinstance(key, tuple) and len(key) == 2):
        return None
    alias, document = key
    if not isinstance(alias, str):
        return None
    try:
        document = int(document)
    except (TypeError, ValueError):
        return None
    return {"kind": "evict", "alias": alias, "document": document,
            "tokens": int(prefix_tokens)}


class KVArena:
    """The tensor backing for the arena, plus the accounting above.

    Per-layer K and V pools of shape (n_pages, page_tokens, n_kv,
    d_head). Torch is imported when the tensor pools are created.

    layer_kv gives each layer its own (n_kv, d_head) when the layers
    differ, as Gemma 4's sliding and full-attention layers do.

    Sliding layers (sliding_layers, with sliding_window tokens behind
    each row) keep a second, smaller pool of n_sliding_pages. A key
    holds pages in both: its whole prefix on the every-token pool, and
    on the sliding pool only the rows from its window origin on. The
    origin is the window's start below base_tokens (the document the
    key may be rewound to), rounded down to a page. A fresh key takes
    sliding pages for its whole prefix during its one pass, since the
    prefix rows attend to each other; trim_window then releases the
    pages before the origin. Page counts the schedulers see are in
    every-token pages, with sliding pages converted at the pools' size
    ratio, so a count that fits admits in both pools.
    """

    def __init__(self, n_layers: int, n_pages: int, page_tokens: int,
                 n_kv: int, d_head: int, dtype=None, device="cuda",
                 layer_kv=None, sliding_layers=(), sliding_window=0,
                 n_sliding_pages=0):
        import torch
        self.torch = torch
        self.dtype = dtype or torch.bfloat16
        self.device = device
        self.pinned = str(device).startswith("cuda")
        shapes = ([(n_kv, d_head)] * n_layers if layer_kv is None
                  else [tuple(shape) for shape in layer_kv])
        if len(shapes) != n_layers:
            raise ValueError(
                f"layer_kv names {len(shapes)} layers, the arena has "
                f"{n_layers}")
        self.shapes = shapes
        self.sliding_layers = (frozenset(sliding_layers) if sliding_window
                               else frozenset())
        self.window = sliding_window if self.sliding_layers else 0
        if self.sliding_layers and n_sliding_pages <= 0:
            raise ValueError("sliding layers need a sliding pool")
        self._build(n_pages, page_tokens, n_sliding_pages)
        self.reset_stats()

    def _build(self, n_pages, page_tokens, n_sliding_pages):
        torch = self.torch
        self.accounting = PageArena(n_pages, page_tokens)
        self.sliding = (PageArena(n_sliding_pages, page_tokens)
                        if self.sliding_layers else None)
        self.k, self.v = [], []
        for layer, (heads, dim) in enumerate(self.shapes):
            pages = n_sliding_pages if layer in self.sliding_layers else n_pages
            shape = (pages * page_tokens, heads, dim)
            self.k.append(torch.empty(shape, dtype=self.dtype, device=self.device))
            self.v.append(torch.empty(shape, dtype=self.dtype, device=self.device))
        # row indices stay on the host: pageable H2D copies block the
        # CPU behind the running stream
        self._rows = {}       # key -> row-index tensor on CPU
        self._capacity_rows = {}  # key -> every row in the claimed pages
        self._sliding_rows = {}   # key -> every row in its sliding pages
        self._base = {}       # key -> tokens the window is anchored below
        self._sliding_start = {}  # key -> logical row of its first sliding page

    def resize(self, n_pages: int, n_sliding_pages: int = 0, *,
               free_resident: bool = False) -> None:
        """Rebuild both pools at new sizes.

        Nothing survives a rebuild; free_resident frees every resident
        key first, and without it a resident key is an error.
        """
        if free_resident:
            for key in self.resident_keys():
                self.free_key(key)
        if self.accounting.owned:
            raise RuntimeError("the arena holds keys; free them before resizing")
        if (n_pages, n_sliding_pages) == (self.n_pages, self.n_sliding_pages):
            return
        page_tokens = self.page_tokens
        self.k = self.v = None
        self._build(n_pages, page_tokens, n_sliding_pages)

    def reset_stats(self):
        self.evicted_keys = 0
        self.evicted_pages = 0
        self.evicted_prefix_tokens = 0

    # ---- the window --------------------------------------------------

    @property
    def has_sliding(self) -> bool:
        return self.sliding is not None

    def origin(self, base_tokens: int) -> int:
        """The logical row a key's sliding pages start at once trimmed."""
        if not self.window or base_tokens <= self.window:
            return 0
        return (base_tokens - self.window) // self.page_tokens * self.page_tokens

    def sliding_start(self, key) -> int:
        """The logical row of the key's first sliding page."""
        return self._sliding_start[key]

    def trim_window(self, key) -> int:
        """Release a key's sliding pages before its window origin.

        Returns the pages freed, in every-token pages.
        """
        if self.sliding is None:
            return 0
        origin = self.origin(self._base[key])
        start = self._sliding_start[key]
        if origin <= start:
            return 0
        before = self.free_pages
        drop = (origin - start) // self.page_tokens
        owned = self.sliding.owned[key]
        self.sliding.free.extend(owned[:drop])
        self.sliding.owned[key] = owned[drop:]
        self.sliding.tokens[key] -= origin - start
        self._sliding_start[key] = origin
        self._refresh_rows(key)
        return self.free_pages - before

    # ---- allocation ----------------------------------------------------

    def alloc(self, key, tokens: int, capacity_tokens: int | None = None,
              base_tokens: int | None = None, sliding_tokens=None):
        """Pages for a fresh key in both pools, or None when either is short.

        sliding_tokens sizes the sliding pages when they differ from
        capacity_tokens, as a temporary's do.
        """
        capacity = tokens if capacity_tokens is None else capacity_tokens
        pages = self.accounting.alloc(key, tokens, capacity)
        if pages is None:
            return None
        if self.sliding is not None:
            want = capacity if sliding_tokens is None else sliding_tokens
            got = self.sliding.alloc(key, min(tokens, want), want)
            if got is None:
                self.accounting.free_key(key)
                return None
        self._base[key] = tokens if base_tokens is None else base_tokens
        self._sliding_start[key] = 0
        # the logical rows are the first `tokens` entries of the
        # capacity rows (same pages, same order), so build once and
        # slice instead of walking the pages twice
        self._refresh_rows(key, tokens)
        return pages

    def _page_rows(self, arena, key, tokens):
        pages = np.asarray(arena.owned[key], dtype=np.int64)
        rows = (pages[:, None] * self.page_tokens
                + np.arange(self.page_tokens, dtype=np.int64))
        return self.torch.from_numpy(rows.reshape(-1)[:tokens])

    def _refresh_rows(self, key, logical_tokens=None):
        logical = (self.accounting.tokens[key]
                   if logical_tokens is None else logical_tokens)
        capacity = len(self.accounting.owned[key]) * self.page_tokens
        cap = self._capacity_rows.get(key)
        if cap is None or cap.numel() != capacity:
            cap = self._page_rows(self.accounting, key, capacity)
        self._capacity_rows[key] = cap
        self._rows[key] = cap[:logical]
        if self.sliding is not None:
            capacity_s = len(self.sliding.owned[key]) * self.page_tokens
            self._sliding_rows[key] = self._page_rows(self.sliding, key, capacity_s)

    def pin(self, key):
        self.accounting.pin(key)

    def retain(self, key, tokens: int, priority=None):
        """Rewind a prefix and make it available for a later operator.

        Retention decisions are keyed by (alias, document), so only
        such pairs may be retained; score and temporary keys cannot.
        A key never rewinds below the base its window was anchored at.
        """
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError(f"retained keys are (alias, document) pairs, "
                            f"got {key!r}")
        if self.sliding is not None and tokens < self._base[key]:
            raise ValueError(
                f"{key!r}: rewinding to {tokens} tokens drops rows of "
                f"the sliding window anchored at {self._base[key]}")
        self.accounting.rewind(key, tokens)
        if self.sliding is not None:
            self.sliding.rewind(key, tokens - self._sliding_start[key])
        self._refresh_rows(key, tokens)
        self.accounting.retain(key, priority)
        cap = self.accounting.retention_cap_pages
        before = self.free_pages
        if cap is not None:
            self.evict_retained(max(0, self.accounting.retained_pages - cap))
        return self.free_pages - before

    def evict_retained(self, pages_needed: int) -> tuple:
        """Evict retained prefixes in planned reuse priority order.

        Stops once pages_needed have come free, counted in every-token
        pages. Returns the evicted keys.
        """
        start = self.free_pages
        return self._evict_until(
            lambda: self.free_pages - start >= pages_needed)

    def _evict_until(self, satisfied) -> tuple:
        """Evict retained prefixes in priority order until satisfied() holds."""
        keys = []
        while not satisfied():
            victim = self.accounting.pop_retained_victim()
            if victim is None:
                break
            keys.append(victim[0])
            self.evict_key(victim[0])
        return tuple(keys)

    def _evict_for_pages(self, need, need_sliding):
        """Evict retained prefixes until both pools have the pages."""
        self._evict_until(lambda: (
            self.accounting.free_pages >= need
            and (self.sliding is None
                 or self.sliding.free_pages >= need_sliding)))

    def evict_key(self, key):
        """Free one retained prefix, record the lost KV, and report it.

        When an answer sink is listening, it is called with kind
        ``evict``, the alias, the document index, and the prefix
        length in tokens. Keys that are not an alias and a document
        index are not reported.
        """
        pages = len(self.accounting.owned[key])
        prefix_tokens = self.accounting.tokens[key]
        self.free_key(key)
        self.evicted_keys += 1
        self.evicted_pages += pages
        self.evicted_prefix_tokens += prefix_tokens
        sink = answer_sink()
        if sink is not None:
            event = _evict_event(key, prefix_tokens)
            if event is not None:
                sink(event)
        return pages

    def activate(self, key, tokens: int, capacity_tokens: int | None = None,
                 base_tokens: int | None = None):
        """Make a prefix active, evicting retained KV when required.

        base_tokens anchors a fresh key's sliding window; a resident
        key keeps the base it was given.
        """
        capacity = tokens if capacity_tokens is None else capacity_tokens
        if key in self.accounting.owned:
            self.accounting.pin(key)
            need = max(0, self.accounting.pages_needed(capacity)
                       - len(self.accounting.owned[key]))
            capacity_s = capacity - self._sliding_start[key]
            need_s = (max(0, self.sliding.pages_needed(capacity_s)
                          - len(self.sliding.owned[key]))
                      if self.sliding is not None else 0)
            self._evict_for_pages(need, need_s)
            # neither pool grows unless both can, so a refusal leaves
            # the key as it was
            if need > self.accounting.free_pages or (
                    self.sliding is not None
                    and need_s > self.sliding.free_pages):
                return None
            self.accounting.grow(key, capacity)
            if self.sliding is not None:
                self.sliding.grow(key, capacity_s)
            self.accounting.tokens[key] = tokens
            if self.sliding is not None:
                self.sliding.tokens[key] = tokens - self._sliding_start[key]
            self._refresh_rows(key, tokens)
            return self.accounting.owned[key]

        need = self.accounting.pages_needed(capacity)
        self._evict_for_pages(need, need)
        pages = self.alloc(key, tokens, capacity, base_tokens)
        if pages is not None:
            self.accounting.pin(key)
        return pages

    def alloc_temporary(self, tokens: int, sliding_tokens: int | None = None):
        key = object()
        pages = self.alloc(key, tokens, base_tokens=0,
                           sliding_tokens=sliding_tokens)
        return None if pages is None else (key, pages)

    def free_key(self, key):
        self._rows.pop(key)
        self._capacity_rows.pop(key)
        self._sliding_rows.pop(key, None)
        self._base.pop(key, None)
        self._sliding_start.pop(key, None)
        if self.sliding is not None:
            self.sliding.free_key(key)
        return self.accounting.free_key(key)

    # ---- residency and retention, read by the loop and operators ------

    @property
    def page_tokens(self) -> int:
        return self.accounting.page_tokens

    @property
    def n_pages(self) -> int:
        return self.accounting.n_pages

    @property
    def n_sliding_pages(self) -> int:
        return 0 if self.sliding is None else self.sliding.n_pages

    @property
    def free_pages(self) -> int:
        """Free pages in every-token pages: the tighter pool decides."""
        free = self.accounting.free_pages
        if self.sliding is None:
            return free
        return min(free, self.sliding.free_pages * self.accounting.n_pages
                   // self.sliding.n_pages)

    def page_cost(self, tokens: int, base_tokens: int | None = None) -> int:
        """Pages a key of `tokens` rows takes, in every-token pages.

        base_tokens prices the trimmed key: only the rows from its
        window origin on sit on the sliding pool. Without it the key is
        priced untrimmed, as a fresh key is admitted.
        """
        pages = self.accounting.pages_needed(tokens)
        if self.sliding is None:
            return pages
        origin = 0 if base_tokens is None else self.origin(base_tokens)
        pages_s = self.sliding.pages_needed(max(0, tokens - origin))
        return max(pages, self._as_every_token_pages(pages_s))

    def held_cost(self, key) -> int:
        """Pages a resident key holds, in every-token pages."""
        pages = len(self.accounting.owned[key])
        if self.sliding is None:
            return pages
        return max(pages, self._as_every_token_pages(
            len(self.sliding.owned[key])))

    def _as_every_token_pages(self, sliding_pages: int) -> int:
        """Sliding pages converted at the pools' size ratio, rounded up."""
        return -(-sliding_pages * self.accounting.n_pages
                 // self.sliding.n_pages)

    @property
    def retained_pages(self) -> int:
        return self.accounting.retained_pages

    @property
    def retained_prefix_tokens(self) -> int:
        return self.accounting.retained_prefix_tokens

    @property
    def retention_cap_pages(self):
        return self.accounting.retention_cap_pages

    @retention_cap_pages.setter
    def retention_cap_pages(self, pages) -> None:
        self.accounting.retention_cap_pages = pages

    def pages_needed(self, tokens: int) -> int:
        return self.accounting.pages_needed(tokens)

    def configure_retention(self, policy, cap_pages: int) -> None:
        """Set the eviction priority rule and the retained-page cap."""
        self.accounting.configure_retention(policy, cap_pages)

    def is_resident(self, key) -> bool:
        return key in self.accounting.owned

    def resident_keys(self) -> list:
        """Every key holding pages, in allocation order."""
        return list(self.accounting.owned)

    def retained_keys(self) -> list:
        """Every key whose KV is kept for a later operator."""
        return list(self.accounting.retained)

    def owned_pages(self, key) -> list:
        """The page ids a resident key holds, in logical order."""
        return self.accounting.owned[key]

    def owned_sliding_pages(self, key) -> list:
        """The sliding-pool page ids a resident key holds, in logical order."""
        return self.sliding.owned[key]

    def capacity_rows(self, key):
        """Row index tensor (CPU) over every row in the key's pages."""
        return self._capacity_rows[key]

    def capacity_rows_sliding(self, key):
        """Row index tensor (CPU) over every row in the key's sliding pages."""
        return self._sliding_rows[key]

    # ---- tensors, read by the attention paths ------------------------

    def layer_kv(self, layer: int):
        """Flat K and V pools of one layer, shape (rows, n_kv, d_head)."""
        return self.k[layer], self.v[layer]

    def paged_kv(self, layer: int):
        """Pools as (n_pages, page_tokens, n_kv, d_head) for paged attention."""
        n_kv, d = self.k[layer].shape[-2], self.k[layer].shape[-1]
        shape = (-1, self.page_tokens, n_kv, d)
        return self.k[layer].view(shape), self.v[layer].view(shape)

    def block_table(self, keys, pad_to=None):
        """Block table and seqused_k for groups reading these documents' KV.

        Built flat on the host and staged through pinned memory.
        """
        pages = [self.accounting.owned[k] for k in keys]
        table = self.block_table_rows(pages, pad_to=pad_to)
        torch = self.torch
        used = torch.tensor([self.accounting.tokens[k] for k in keys],
                            dtype=torch.int32, pin_memory=self.pinned) \
            .to(self.device, non_blocking=self.pinned)
        return table, used

    def block_table_rows(self, pages, pad_to=None):
        """A block table from explicit physical page rows."""
        torch = self.torch
        width = max(len(p) for p in pages)
        if pad_to:
            width = max(width, pad_to)
        flat = []
        for p in pages:
            flat.extend(p)
            flat.extend([0] * (width - len(p)))
        table = torch.tensor(flat, dtype=torch.int32,
                             pin_memory=self.pinned).view(len(pages), width) \
            .to(self.device, non_blocking=self.pinned)
        return table
