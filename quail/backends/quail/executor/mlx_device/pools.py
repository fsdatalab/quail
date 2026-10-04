"""KV pools for the MLX device implementation."""

from quail.backends.quail.executor.mlx_device.kernels import write_rows


class MlxKVPools:
    """One K pool and one V pool of KV pages per layer, as MLX arrays.

    A layer's pool has shape (pages, page tokens, KV heads, head dim). A
    slot is a pool row: page * page tokens + the offset in the page, the
    row index the page accounting hands out. The KVArena builds the
    pools.

    Writes happen in place and replace the pool arrays with the arrays
    the write returns. Read a layer's pools with paged_kv after its
    writes, not before.

    Args:
        dtype: MLX dtype of K and V.
    """

    def __init__(self, dtype):
        self.dtype = dtype
        self.page_tokens = None
        self.k = self.v = None

    def build(self, page_tokens: int, layers) -> None:
        """Allocate the pools; nothing of earlier pools survives.

        Args:
            page_tokens: Tokens per page.
            layers: Per layer, (pages, KV heads, head dim).
        """
        import mlx.core as mx

        # the earlier pools go before the new ones are allocated
        self.k = self.v = None
        self.page_tokens = page_tokens
        shapes = [(pages, page_tokens, heads, dim) for pages, heads, dim in layers]
        self.k = [mx.zeros(shape, dtype=self.dtype) for shape in shapes]
        self.v = [mx.zeros(shape, dtype=self.dtype) for shape in shapes]
        mx.eval(self.k, self.v)

    @property
    def nbytes(self) -> int:
        """Bytes the pools hold."""
        return sum(pool.nbytes for pool in self.k + self.v)

    def layer_kv(self, layer: int):
        """Flat K and V pools of one layer, shape (rows, n_kv, d_head)."""
        rows = (-1, *self.k[layer].shape[2:])
        return self.k[layer].reshape(rows), self.v[layer].reshape(rows)

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
        k_rows, v_rows = self.layer_kv(layer)
        self.write(layer, k_rows[source_slots], v_rows[source_slots], slots)

    def arrays(self) -> list:
        """Return every pool array, to evaluate with a chunk's answers."""
        return self.k + self.v
