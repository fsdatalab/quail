"""The paged KV arena.

One preallocated buffer per layer, sized to the admission budget,
divided into fixed 16-token pages with a free list. A resident
document owns a list of pages; the pages return to the free list the
instant the document fails a stage or answers its last one. The paged
attention kernels read this layout natively (block tables), so there
are no copies and no compaction - and since admission never lets
page-rounded resident tokens exceed the arena, allocation cannot fail.

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

    def pages_needed(self, tokens: int) -> int:
        return -(-tokens // self.page_tokens)

    def alloc(self, key, tokens: int):
        """The document's pages, or None when the free list is short
        (the admission scheduler treats None as "wait")."""
        if key in self.owned:
            raise KeyError(f"{key!r} already resident")
        need = self.pages_needed(tokens)
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
        self.free.extend(pages)
        return len(pages)

    def row_indices(self, key):
        """Flat row positions of the document's tokens inside a pool
        viewed as (n_pages * page_tokens, ...): page p holds rows
        p*page_tokens .. p*page_tokens + page_tokens."""
        rows = []
        left = self.tokens[key]
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
        self._rows = {}       # key -> row-index tensor on device
        self.device = device

    def alloc(self, key, tokens: int):
        pages = self.accounting.alloc(key, tokens)
        if pages is None:
            return None
        self._rows[key] = self.torch.tensor(
            self.accounting.row_indices(key), dtype=self.torch.int64,
            device=self.device)
        return pages

    def free_key(self, key):
        self._rows.pop(key)
        return self.accounting.free_key(key)

    def write(self, layer: int, key, k_rows, v_rows, offset: int = 0):
        """Scatter fresh K/V rows into the document's pages, starting
        `offset` tokens into its allocation (the shared question
        preamble lands after the document's own rows)."""
        rows = self._rows[key][offset:offset + k_rows.shape[0]]
        self.k[layer].index_copy_(0, rows, k_rows)
        self.v[layer].index_copy_(0, rows, v_rows)

    def paged_kv(self, layer: int):
        """The pools viewed as (n_pages, page_tokens, n_kv, d_head)
        for the paged attention call."""
        n_kv, d = self.k[layer].shape[-2], self.k[layer].shape[-1]
        shape = (-1, self.page_tokens, n_kv, d)
        return self.k[layer].view(shape), self.v[layer].view(shape)

    def block_table(self, keys, pad_to=None):
        """(block_table int32 (len(keys), max_pages), seqused_k int32)
        for the groups reading these documents' KV, in order."""
        pages = [self.accounting.owned[k] for k in keys]
        width = max(len(p) for p in pages)
        if pad_to:
            width = max(width, pad_to)
        table = self.torch.zeros((len(keys), width),
                                 dtype=self.torch.int32,
                                 device=self.device)
        for i, p in enumerate(pages):
            table[i, :len(p)] = self.torch.tensor(
                p, dtype=self.torch.int32, device=self.device)
        used = self.torch.tensor(
            [self.accounting.tokens[k] for k in keys],
            dtype=self.torch.int32, device=self.device)
        return table, used

    def gather(self, layer: int, key):
        """A document's (K, V) rows, contiguous - the copy fallback if
        the paged kernel path fails a parity gate."""
        rows = self._rows[key]
        return (self.k[layer].index_select(0, rows),
                self.v[layer].index_select(0, rows))
