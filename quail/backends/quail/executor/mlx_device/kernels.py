"""vllm-metal's paged KV write and paged attention, as Quail calls them.

vllm-metal is the vLLM project's Apple silicon plugin. Quail uses two
functions of its Metal kernel extension and nothing else of it. Both
take a KV pool of shape (pages, page tokens, KV heads, head dim). A
slot is a pool row: page * page tokens + the offset in the page.
"""

# The value vllm-metal takes for full attention. 0 is a window of no
# rows and returns wrong values without an error.
NO_WINDOW = -1
NO_SOFTCAP = 0.0

WHEEL = ("https://github.com/vllm-project/vllm-metal/releases/download/v0.30.0/"
         "vllm_metal-0.30.0-cp312-cp312-macosx_15_0_arm64.whl")
INSTALL_TEXT = (
    "MLX execution needs mlx and vllm-metal's Metal kernels on Apple "
    "silicon with macOS 15 or later. Install them with "
    "`pip install 'quail-engine[mlx]'` and "
    f"`pip install --no-deps {WHEEL}`."
)


def metal_ops():
    """Return vllm-metal's kernel module.

    Raises:
        RuntimeError: mlx or vllm-metal is not installed.
    """
    try:
        from vllm_metal.metal import get_ops
    except ImportError as error:
        raise RuntimeError(INSTALL_TEXT) from error
    return get_ops()


def write_rows(k_rows, v_rows, k_pool, v_pool, slots):
    """Write K and V rows into pool slots in place.

    The returned pools share the given pools' memory. Later reads must
    use the returned arrays: MLX orders the write before a read only
    when the read takes the write's output.

    Args:
        k_rows: K rows, (rows, KV heads, head dim), contiguous.
        v_rows: V rows, same shape, contiguous.
        k_pool: K pool, (pages, page tokens, KV heads, head dim).
        v_pool: V pool, same shape.
        slots: int64 slot of each row.

    Returns:
        The K pool and the V pool holding the rows.
    """
    return metal_ops().reshape_and_cache(k_rows, v_rows, k_pool, v_pool, slots)


def paged_attention(q, k_pool, v_pool, *, table, used, cu_q, max_used, scale):
    """Causal attention of packed query rows over paged KV.

    The rows are grouped into sequences. A sequence's rows are the last
    rows of its context, and each row reads the context up to itself.

    Args:
        q: Query rows, (rows, heads, head dim), contiguous.
        k_pool: K pool, (pages, page tokens, KV heads, head dim), holding
            every context row including the query rows' own.
        v_pool: V pool, same shape.
        table: int32 block table, (sequences, pages): the pages of each
            sequence's context in order, padded with 0.
        used: int32 context tokens of each sequence.
        cu_q: int32 cumulative query rows, (sequences + 1,).
        max_used: The largest entry of used.
        scale: Attention logit scale.

    Returns:
        (rows, heads, head dim) attention output in q's dtype.
    """
    import mlx.core as mx

    out = mx.array(0)
    page_tokens, n_kv = k_pool.shape[1], k_pool.shape[2]
    metal_ops().paged_attention_primitive(
        q, k_pool, v_pool, n_kv, scale, NO_SOFTCAP, table, used, cu_q,
        page_tokens, int(max_used), NO_WINDOW, out)
    return out


def check_geometry(n_q, n_kv, head_dim, page_tokens, dtype) -> None:
    """Run one row through both kernels to check they exist for a geometry.

    vllm-metal builds its kernels for a fixed set of head sizes, page
    sizes, and dtypes; a geometry outside it fails when first run.

    Raises:
        ValueError: vllm-metal has no kernel for the geometry.
    """
    import mlx.core as mx

    pool = mx.zeros((1, page_tokens, n_kv, head_dim), dtype=dtype)
    row = mx.zeros((1, n_kv, head_dim), dtype=dtype)
    try:
        k_pool, v_pool = write_rows(row, row, pool, pool + 0,
                                    mx.array([0], dtype=mx.int64))
        out = paged_attention(
            mx.zeros((1, n_q, head_dim), dtype=dtype), k_pool, v_pool,
            table=mx.array([[0]], dtype=mx.int32),
            used=mx.array([1], dtype=mx.int32),
            cu_q=mx.array([0, 1], dtype=mx.int32), max_used=1,
            scale=1.0)
        mx.eval(out)
    except RuntimeError as error:
        raise ValueError(
            f"vllm-metal has no paged attention kernel for {n_q} heads, "
            f"{n_kv} KV heads, head dim {head_dim}, {page_tokens}-token "
            f"pages, {dtype}") from error
