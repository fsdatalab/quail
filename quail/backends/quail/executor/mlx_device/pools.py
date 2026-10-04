"""KV pools for the MLX device implementation."""

from quail.backends.quail.executor.mlx_device.kernels import write_rows


class MlxKVPools:
    """One K pool and one V pool of KV pages per layer, as MLX arrays.

    A pool has shape (pages, page tokens, KV heads, head dim). A slot is
    a pool row: page * page tokens + the offset in the page, the row
    index the page accounting hands out.

    Writes happen in place and replace the pool arrays with the arrays
    the write returns. Read a layer's pools with paged_kv after its
    writes, not before.

    Args:
        n_layers: Layers of the model.
        n_pages: Pages per pool.
        page_tokens: Tokens per page.
        n_kv: KV heads.
        d_head: Head dimension.
        dtype: MLX dtype of K and V.
    """

    def __init__(self, n_layers: int, n_pages: int, page_tokens: int,
                 n_kv: int, d_head: int, dtype):
        self.n_layers = n_layers
        self.page_tokens = page_tokens
        self.n_kv = n_kv
        self.d_head = d_head
        self.dtype = dtype
        self._build(n_pages)

    def _build(self, n_pages: int) -> None:
        import mlx.core as mx

        self.n_pages = n_pages
        shape = (n_pages, self.page_tokens, self.n_kv, self.d_head)
        self.k = [mx.zeros(shape, dtype=self.dtype) for _ in range(self.n_layers)]
        self.v = [mx.zeros(shape, dtype=self.dtype) for _ in range(self.n_layers)]
        mx.eval(self.k, self.v)

    def resize(self, n_pages: int) -> None:
        """Rebuild the pools at a new size; nothing survives."""
        if n_pages == self.n_pages:
            return
        self.k = self.v = None
        self._build(n_pages)

    @property
    def nbytes(self) -> int:
        """Bytes the pools hold."""
        return sum(pool.nbytes for pool in self.k + self.v)

    def paged_kv(self, layer: int):
        """Return the K and V pools of one layer for paged attention."""
        return self.k[layer], self.v[layer]

    def write(self, layer: int, k_rows, v_rows, slots) -> None:
        """Write K and V rows into one layer's slots.

        Args:
            layer: The layer.
            k_rows: K rows, (rows, KV heads, head dim), contiguous.
            v_rows: V rows, same shape, contiguous.
            slots: int64 slot of each row.
        """
        self.k[layer], self.v[layer] = write_rows(
            k_rows, v_rows, self.k[layer], self.v[layer], slots)

    def copy(self, layer: int, source_slots, slots) -> None:
        """Copy rows of one layer from source slots to other slots.

        Args:
            layer: The layer.
            source_slots: int32 slots to read, written before this call.
            slots: int64 slots to write, as many.
        """
        rows = (-1, self.n_kv, self.d_head)
        self.write(layer,
                   self.k[layer].reshape(rows)[source_slots],
                   self.v[layer].reshape(rows)[source_slots],
                   slots)

    def arrays(self) -> list:
        """Return every pool array, to evaluate with a chunk's answers."""
        return self.k + self.v
