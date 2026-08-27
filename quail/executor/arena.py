"""Paged KV arena: preallocated per-layer buffers divided into
fixed-size pages with a free list and block-table access.

PageArena is the accounting (pure Python, CPU-tested); KVArena is the
tensor backing and runs only where torch and a GPU exist.
"""

class PageArena:
    """Page accounting: a free list and per-document page lists."""

    def __init__(self, n_pages: int, page_tokens: int):
        self.n_pages = n_pages
        self.page_tokens = page_tokens
        self.free = list(range(n_pages - 1, -1, -1))    # stack
        self.owned = {}       # key -> list of page ids
        self.tokens = {}      # key -> resident token count
        self.pinned = set()    # current operators depend on these keys
        self.retained = {}     # key -> ideal seconds saved at next use

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
        self.retained.pop(key, None)
        self.free.extend(pages)
        return len(pages)

    def pin(self, key) -> None:
        """Protect a resident key while an operator uses it."""

        if key not in self.owned:
            raise KeyError(key)
        self.retained.pop(key, None)
        self.pinned.add(key)

    def retain(self, key, value: float) -> None:
        """Make a resident key evictable after its current use."""

        if key not in self.owned:
            raise KeyError(key)
        if value < 0:
            raise ValueError("retention value must be nonnegative")
        self.pinned.discard(key)
        self.retained[key] = value

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
        return sum(len(self.owned[key]) for key in self.retained)


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
        self._rows_dev = {}   # key -> device copy, built on first use
        self.device = device
        self.reset_stats()

    def reset_stats(self):
        self.evicted_keys = 0
        self.evicted_pages = 0
        self.evicted_value = 0.0

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
        self._rows_dev.pop(key, None)

    def pin(self, key):
        self.accounting.pin(key)

    def retain(self, key, tokens: int, value: float):
        """Rewind a prefix and make it available for a later operator."""

        self.accounting.rewind(key, tokens)
        self._refresh_rows(key, tokens)
        self.accounting.retain(key, value)

    def evict_retained(self, pages_needed: int) -> tuple:
        """Evict the minimum value set that frees pages_needed pages."""

        from quail.executor.retention import Retained, minimum_loss_victims

        entries = (
            Retained(key, len(self.accounting.owned[key]), value)
            for key, value in self.accounting.retained.items()
        )
        victims = minimum_loss_victims(entries, pages_needed)
        if victims is None:
            return ()
        self.evicted_keys += len(victims.keys)
        self.evicted_pages += victims.pages
        self.evicted_value += victims.value
        for key in victims.keys:
            self.free_key(key)
        return victims.keys

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
        self._rows_dev.pop(key, None)
        return self.accounting.free_key(key)

    def rows_gpu(self, key):
        """The document's row indices on device, cached per residency."""
        r = self._rows_dev.get(key)
        if r is None:
            r = self._rows[key].to(self.device)
            self._rows_dev[key] = r
        return r

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
