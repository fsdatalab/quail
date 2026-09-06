"""Paged KV arena: preallocated per-layer buffers divided into
fixed-size pages with a free list and block-table access.

PageArena is the accounting (pure Python, CPU-tested); KVArena is the
tensor backing and runs only where torch and a GPU exist.
"""

import heapq


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
        """The document's pages, or None when the free list is short
        (the admission scheduler treats None as "wait")."""
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
        """Flat row positions of the document's tokens inside a pool
        viewed as (n_pages * page_tokens, ...): page p holds rows
        p*page_tokens .. p*page_tokens + page_tokens."""
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


class KVArena:
    """The tensor backing: per-layer K and V pools of shape
    (n_pages, page_tokens, n_kv, d_head), plus the accounting above.
    Import-time torch dependency is deliberate here; this class only
    exists inside the Modal image."""

    def __init__(self, n_layers: int, n_pages: int, page_tokens: int,
                 n_kv: int, d_head: int, dtype=None, device="cuda"):
        import torch
        self.torch = torch
        dtype = dtype or torch.bfloat16
        self.page_tokens = page_tokens
        self.accounting = PageArena(n_pages, page_tokens)
        shape = (n_pages * page_tokens, n_kv, d_head)
        self.k = [torch.empty(shape, dtype=dtype, device=device)
                  for _ in range(n_layers)]
        self.v = [torch.empty(shape, dtype=dtype, device=device)
                  for _ in range(n_layers)]
        # row indices stay on the host: pageable H2D copies block the
        # CPU behind the running stream
        self._rows = {}       # key -> row-index tensor on CPU
        self._capacity_rows = {}  # key -> every row in the claimed pages
        self.device = device
        self.reset_stats()

    def reset_stats(self):
        self.evicted_keys = 0
        self.evicted_pages = 0
        self.evicted_prefix_tokens = 0

    def alloc(self, key, tokens: int, capacity_tokens: int | None = None):
        pages = self.accounting.alloc(key, tokens, capacity_tokens)
        if pages is None:
            return None
        # the logical rows are the first `tokens` entries of the
        # capacity rows (same pages, same order), so build once and
        # slice instead of walking the pages twice
        cap = self.torch.tensor(
            self.accounting.row_indices(key, capacity_tokens),
            dtype=self.torch.int64)
        self._capacity_rows[key] = cap
        self._rows[key] = cap[:tokens]
        return pages

    def _refresh_rows(self, key, logical_tokens=None):
        logical = (self.accounting.tokens[key]
                   if logical_tokens is None else logical_tokens)
        capacity = len(self.accounting.owned[key]) * self.page_tokens
        cap = self.torch.tensor(
            self.accounting.row_indices(key, capacity),
            dtype=self.torch.int64)
        self._capacity_rows[key] = cap
        self._rows[key] = cap[:logical]

    def pin(self, key):
        self.accounting.pin(key)

    def retain(self, key, tokens: int, priority=None):
        """Rewind a prefix and make it available for a later operator."""

        self.accounting.rewind(key, tokens)
        self._refresh_rows(key, tokens)
        self.accounting.retain(key, priority)
        cap = self.accounting.retention_cap_pages
        before = self.accounting.free_pages
        if cap is not None:
            self.evict_retained(max(0, self.accounting.retained_pages - cap))
        return self.accounting.free_pages - before

    def evict_retained(self, pages_needed: int) -> tuple:
        """Evict retained prefixes in planned reuse priority order."""

        keys = []
        pages = 0
        while pages < pages_needed:
            victim = self.accounting.pop_retained_victim()
            if victim is None:
                break
            key, victim_pages, _ = victim
            keys.append(key)
            pages += victim_pages
            self.evict_key(key)
        return tuple(keys)

    def evict_key(self, key):
        """Free one retained prefix and record the lost KV."""

        pages = len(self.accounting.owned[key])
        prefix_tokens = self.accounting.tokens[key]
        self.free_key(key)
        self.evicted_keys += 1
        self.evicted_pages += pages
        self.evicted_prefix_tokens += prefix_tokens
        return pages

    def activate(self, key, tokens: int, capacity_tokens: int | None = None):
        """Make a prefix active, evicting retained KV when required."""

        capacity = tokens if capacity_tokens is None else capacity_tokens
        if key in self.accounting.owned:
            self.accounting.pin(key)
            current = len(self.accounting.owned[key])
            need = max(0, self.accounting.pages_needed(capacity) - current)
            if need > self.accounting.free_pages:
                self.evict_retained(need - self.accounting.free_pages)
            grown = self.accounting.grow(key, capacity)
            if grown is None:
                return None
            self.accounting.tokens[key] = tokens
            self._refresh_rows(key, tokens)
            return self.accounting.owned[key]

        need = self.accounting.pages_needed(capacity)
        if need > self.accounting.free_pages:
            self.evict_retained(need - self.accounting.free_pages)
        pages = self.alloc(key, tokens, capacity)
        if pages is not None:
            self.accounting.pin(key)
        return pages

    def alloc_temporary(self, tokens: int):
        key = object()
        pages = self.alloc(key, tokens)
        return None if pages is None else (key, pages)

    def free_key(self, key):
        self._rows.pop(key)
        self._capacity_rows.pop(key)
        return self.accounting.free_key(key)

    def paged_kv(self, layer: int):
        """The pools viewed as (n_pages, page_tokens, n_kv, d_head)
        for the paged attention call."""
        n_kv, d = self.k[layer].shape[-2], self.k[layer].shape[-1]
        shape = (-1, self.page_tokens, n_kv, d)
        return self.k[layer].view(shape), self.v[layer].view(shape)

    def block_table(self, keys, pad_to=None):
        """Block table and seqused_k for the groups reading these
        documents' KV, built flat on the host and staged through
        pinned memory."""
        pages = [self.accounting.owned[k] for k in keys]
        table = self.block_table_rows(pages, pad_to=pad_to)
        torch = self.torch
        used = torch.tensor([self.accounting.tokens[k] for k in keys],
                            dtype=torch.int32, pin_memory=True) \
            .to(self.device, non_blocking=True)
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
                             pin_memory=True).view(len(pages), width) \
            .to(self.device, non_blocking=True)
        return table
