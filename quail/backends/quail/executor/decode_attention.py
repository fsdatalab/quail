"""Triton Split-KV causal paged attention for short queries on head-256 FA4 GPUs.

Blackwell's head-256 FA4 kernel splits neither the KV nor the query heads that
share a KV head, leaving short-query decode passes underutilized. A dynamic
split-KV Triton kernel saturates all SMs without mid-query compilations.
"""

from quail.backends.quail.executor.triton_launch import (
    launch_triton,
    num_multiprocessors,
    pdl_enabled,
)

_KERNELS = {}
# Maximum query tokens per sequence routed to Triton Split-KV decode.
TRITON_DECODE_MAX_Q = 16
_SPLITKV_WAVES_PER_SM = 4
_SPLITKV_MAX_SPLITS = 16
_SPLITKV_TILE_N = 32


def supports_splitkv_decode(fa_version: int, head_dim: int, page_tokens: int,
                            num_q_heads: int, num_kv_heads: int, *,
                            causal: bool, max_q: int, dtype=None) -> bool:
    """Return whether a paged attention call should run Triton Split-KV decode.

    Shared by Engine._paged, mixed prefill/decode chunk splitting, and boot
    warmup so eligibility cannot drift across call sites.

    Args:
        fa_version: Active FlashAttention version for the GPU.
        head_dim: Query and KV head dimension.
        page_tokens: Tokens per physical KV arena page.
        num_q_heads: Query head count.
        num_kv_heads: KV head count.
        causal: Whether the attention call is causal.
        max_q: Maximum query tokens per sequence in the batch.
        dtype: Optional tensor dtype; when given, must be bfloat16.

    Returns:
        True when Triton Split-KV decode is valid and preferred over FA4.
    """
    if fa_version != 4 or head_dim != 256 or not causal:
        return False
    if not (0 < max_q <= TRITON_DECODE_MAX_Q):
        return False
    if page_tokens <= 0 or page_tokens % _SPLITKV_TILE_N != 0:
        return False
    if head_dim & (head_dim - 1):
        return False
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        return False
    if dtype is not None:
        import torch

        if dtype != torch.bfloat16:
            return False
    return True


def _kernels() -> dict:
    """Compile the split-KV decode Triton kernels on first use."""
    if _KERNELS:
        return _KERNELS
    import triton
    import triton.language as tl

    # PDL invariant: loads before gdc_wait read only host-staged metadata
    # (cu_q, seq_lens); early-exit stores touch only split-KV buffers whose
    # last reader finished before the previous layer's non-PDL GEMMs.
    @triton.jit(
        do_not_specialize=[
            "stride_bt", "max_q", "num_splits", "window_left", "window_right",
        ],
        do_not_specialize_on_alignment=[
            "cu_q_ptr", "seq_lens_ptr", "block_tables_ptr",
        ],
    )
    def splitkv_decode_part(
        q_ptr, k_ptr, v_ptr, cu_q_ptr, seq_lens_ptr, block_tables_ptr,
        out_ptr, part_out_ptr, part_lse_ptr,
        stride_q0, stride_q1, stride_o0, stride_o1,
        stride_k0, stride_k1, stride_k2, stride_v0, stride_v1, stride_v2,
        stride_bt, stride_po_q, stride_po_kv, stride_po_s, stride_po_g,
        stride_pl_q, stride_pl_kv, stride_pl_s,
        scale_log2, max_q, num_splits, window_left, window_right,
        GROUP: tl.constexpr, GROUP_PAD: tl.constexpr,
        HEAD_DIM: tl.constexpr, PAGE_TOKENS: tl.constexpr,
        TILE_N: tl.constexpr, HAS_WINDOW: tl.constexpr,
        SINGLE_SPLIT: tl.constexpr, PDL: tl.constexpr,
    ):
        pid_q = tl.program_id(0)
        kv_head = tl.program_id(1)
        split_id = tl.program_id(2)
        seq = pid_q // max_q
        q_local = pid_q % max_q
        q_start = tl.load(cu_q_ptr + seq).to(tl.int64)
        q_end = tl.load(cu_q_ptr + seq + 1).to(tl.int64)
        q_len = q_end - q_start
        if q_local >= q_len:
            if PDL:
                tl.extra.cuda.gdc_launch_dependents()
            return
        q_row = q_start + q_local
        seq_len = tl.load(seq_lens_ptr + seq).to(tl.int64)
        visible_k = seq_len - q_len + q_local + 1
        k_lo = visible_k - visible_k
        if HAS_WINDOW:
            if window_left >= 0:
                k_lo = tl.maximum(k_lo, visible_k - 1 - window_left)
            if window_right >= 0:
                visible_k = tl.minimum(visible_k, visible_k + window_right)
        first_tile = k_lo // TILE_N
        last_tile = (visible_k + TILE_N - 1) // TILE_N
        total_tiles = tl.maximum(0, last_tile - first_tile)
        tiles_per = (total_tiles + num_splits - 1) // num_splits
        tile_start = first_tile + split_id * tiles_per
        tile_end = tl.minimum(last_tile, tile_start + tiles_per)

        offs_g = tl.arange(0, GROUP_PAD)
        offs_d = tl.arange(0, HEAD_DIM)
        g_mask = offs_g < GROUP

        if tile_start >= tile_end:
            if not SINGLE_SPLIT:
                pl_base = (part_lse_ptr + q_row * stride_pl_q
                           + kv_head * stride_pl_kv + split_id * stride_pl_s)
                if PDL:
                    tl.extra.cuda.gdc_launch_dependents()
                tl.store(pl_base + offs_g, -1e38, mask=g_mask)
            else:
                o_base = (out_ptr + q_row * stride_o0
                          + (kv_head * GROUP + offs_g[:, None]) * stride_o1
                          + offs_d[None, :])
                if PDL:
                    tl.extra.cuda.gdc_launch_dependents()
                tl.store(o_base, tl.zeros([GROUP_PAD, HEAD_DIM], dtype=tl.bfloat16),
                         mask=g_mask[:, None])
            return

        if PDL:
            tl.extra.cuda.gdc_wait()
        q_ptrs = (q_ptr + q_row * stride_q0
                  + (kv_head * GROUP + offs_g[:, None]) * stride_q1
                  + offs_d[None, :])
        q_tile = tl.load(q_ptrs, mask=g_mask[:, None], other=0.0)

        offs_k_tile = tl.arange(0, TILE_N)
        m_i = tl.full([GROUP_PAD], -1e38, dtype=tl.float32)
        l_i = tl.zeros([GROUP_PAD], dtype=tl.float32)
        acc = tl.zeros([GROUP_PAD, HEAD_DIM], dtype=tl.float32)
        bt_row = block_tables_ptr + seq * stride_bt

        for tile_index in range(tile_start, tile_end):
            position = tile_index * TILE_N
            page_index = position // PAGE_TOKENS
            in_page = position - page_index * PAGE_TOKENS
            phys_page = tl.load(bt_row + page_index).to(tl.int64)
            k_ptrs = (k_ptr + phys_page * stride_k0
                      + (in_page + offs_k_tile[:, None]) * stride_k1
                      + kv_head * stride_k2 + offs_d[None, :])
            v_ptrs = (v_ptr + phys_page * stride_v0
                      + (in_page + offs_k_tile[:, None]) * stride_v1
                      + kv_head * stride_v2 + offs_d[None, :])
            key_position = position + offs_k_tile
            key_valid = (key_position < visible_k) & (key_position >= k_lo)
            k_tile = tl.load(k_ptrs, mask=key_valid[:, None], other=0.0)
            v_tile = tl.load(v_ptrs, mask=key_valid[:, None], other=0.0)
            qk = tl.dot(q_tile, tl.trans(k_tile)) * scale_log2
            qk = tl.where(key_valid[None, :], qk, -1e38)
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            valid_row = m_new > -1e30
            alpha = tl.where(valid_row, tl.math.exp2(m_i - m_new), 0.0)
            p = tl.where(valid_row[:, None] & key_valid[None, :],
                         tl.math.exp2(qk - m_new[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v_tile)
            m_i = m_new

        inv_l = tl.where(l_i > 0.0, 1.0 / l_i, 0.0)
        norm_acc = acc * inv_l[:, None]
        if PDL:
            tl.extra.cuda.gdc_launch_dependents()
        if SINGLE_SPLIT:
            o_ptrs = (out_ptr + q_row * stride_o0
                      + (kv_head * GROUP + offs_g[:, None]) * stride_o1
                      + offs_d[None, :])
            tl.store(o_ptrs, norm_acc.to(out_ptr.dtype.element_ty),
                     mask=g_mask[:, None])
        else:
            lse = tl.where(l_i > 0.0, m_i + tl.math.log2(l_i), -1e38)
            po_ptrs = (part_out_ptr + q_row * stride_po_q
                       + kv_head * stride_po_kv + split_id * stride_po_s
                       + offs_g[:, None] * stride_po_g + offs_d[None, :])
            pl_ptrs = (part_lse_ptr + q_row * stride_pl_q
                       + kv_head * stride_pl_kv + split_id * stride_pl_s
                       + offs_g)
            tl.store(po_ptrs, norm_acc, mask=g_mask[:, None])
            tl.store(pl_ptrs, lse, mask=g_mask)

    @triton.jit(do_not_specialize=["num_splits"])
    def splitkv_decode_reduce(
        part_out_ptr, part_lse_ptr, out_ptr,
        stride_po_q, stride_po_kv, stride_po_s, stride_po_g,
        stride_pl_q, stride_pl_kv, stride_pl_s,
        stride_o0, stride_o1, num_splits,
        GROUP: tl.constexpr, HEAD_DIM: tl.constexpr,
        MAX_SPLITS: tl.constexpr, PDL: tl.constexpr,
    ):
        q_row = tl.program_id(0).to(tl.int64)
        q_head = tl.program_id(1)
        kv_head = q_head // GROUP
        g = q_head % GROUP
        offs_s = tl.arange(0, MAX_SPLITS)
        offs_d = tl.arange(0, HEAD_DIM)
        s_mask = offs_s < num_splits
        pl_base = (part_lse_ptr + q_row * stride_pl_q
                   + kv_head * stride_pl_kv + g)
        if PDL:
            tl.extra.cuda.gdc_wait()
        lse_all = tl.load(pl_base + offs_s * stride_pl_s, mask=s_mask,
                          other=-1e38)
        m_max = tl.max(lse_all, axis=0)
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
        w_sum = 0.0
        po_base = (part_out_ptr + q_row * stride_po_q
                   + kv_head * stride_po_kv + g * stride_po_g + offs_d)
        for split_id in range(0, num_splits):
            lse_s = tl.load(pl_base + split_id * stride_pl_s)
            if lse_s > -1e30:
                w = tl.math.exp2(lse_s - m_max)
                part = tl.load(po_base + split_id * stride_po_s)
                acc += w * part
                w_sum += w
        inv_w = tl.where(w_sum > 0.0, 1.0 / w_sum, 0.0)
        out_val = (acc * inv_w).to(out_ptr.dtype.element_ty)
        if PDL:
            tl.extra.cuda.gdc_launch_dependents()
        tl.store(out_ptr + q_row * stride_o0 + q_head * stride_o1 + offs_d,
                 out_val)

    _KERNELS.update(
        splitkv_decode_part=splitkv_decode_part,
        splitkv_decode_reduce=splitkv_decode_reduce,
    )
    return _KERNELS


def triton_splitkv_decode(q3, kp, vp, cu_q, max_q, seqused_k, max_seqused_k,
                          block_table, *, softmax_scale=None, window=None,
                          output=None, workspace=None):
    """Causal paged attention for short queries with dynamic Split-KV and PDL.

    Targets four CTA waves per GPU multiprocessor and compiles one kernel per
    head geometry regardless of batch size, block-table width, or index
    alignment.

    Args:
        q3: (total_q, num_q_heads, head_dim) bfloat16 queries, unit inner stride.
        kp: (num_pages, page_tokens, num_kv_heads, head_dim) bfloat16 paged keys.
        vp: (num_pages, page_tokens, num_kv_heads, head_dim) bfloat16 paged values.
        cu_q: (num_seqs + 1,) cumulative query offsets, int32.
        max_q: Maximum query length in the batch.
        seqused_k: (num_seqs,) visible KV tokens per sequence, int32.
        max_seqused_k: Maximum entry of seqused_k.
        block_table: (num_seqs, max_pages) physical page indices, int32.
        softmax_scale: Logit scale; None uses head_dim ** -0.5.
        window: Optional (left, right) sliding window; None is full causal.
        output: Optional (total_q, num_q_heads, head_dim) bfloat16 output tensor.
        workspace: Optional engine-local dict caching split-KV partial buffers.

    Returns:
        The (total_q, num_q_heads, head_dim) attention output.

    Raises:
        ValueError: Input layout, dtype, page size, or head geometry is invalid.
    """
    import torch

    total_q, num_q_heads, head_dim = q3.shape
    page_tokens, num_kv_heads = kp.shape[1], kp.shape[2]
    num_seqs = seqused_k.shape[0]
    if page_tokens % _SPLITKV_TILE_N != 0:
        raise ValueError(
            f"triton_splitkv_decode requires page_tokens divisible by "
            f"{_SPLITKV_TILE_N}, got {page_tokens}")
    if (q3.dtype != torch.bfloat16 or kp.dtype != torch.bfloat16
            or vp.dtype != torch.bfloat16):
        raise ValueError("triton_splitkv_decode requires bfloat16 Q, K, and V")
    if head_dim & (head_dim - 1):
        raise ValueError(f"head_dim must be a power of two, got {head_dim}")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    out = output if output is not None else torch.empty_like(q3)
    if (q3.stride(-1) != 1 or kp.stride(-1) != 1 or vp.stride(-1) != 1
            or out.stride(-1) != 1):
        raise ValueError("Q, K, V, and out must have unit inner stride")
    if total_q == 0 or num_seqs == 0:
        return out

    group = num_q_heads // num_kv_heads
    group_pad = max(16, 1 << (group - 1).bit_length())
    scale = (head_dim ** -0.5 if softmax_scale is None
             else float(softmax_scale))
    scale_log2 = scale * 1.4426950408889634
    has_window = window is not None and (window[0] >= 0 or window[1] >= 0)
    window_left = window[0] if has_window else -1
    window_right = window[1] if has_window else -1

    target_ctas = _SPLITKV_WAVES_PER_SM * num_multiprocessors(q3.device)
    base_ctas = num_seqs * max_q * num_kv_heads
    max_useful = max(1, (max_seqused_k + page_tokens - 1) // page_tokens)
    want_splits = (target_ctas + base_ctas - 1) // base_ctas
    num_splits = min(_SPLITKV_MAX_SPLITS, max_useful, max(1, want_splits))
    single_split = num_splits == 1

    if single_split:
        part_out = out
        part_lse = out
        po_strides = (0, 0, 0, 0)
        pl_strides = (0, 0, 0)
    else:
        buf_key = (q3.device, num_kv_heads, group, head_dim)
        cached = None if workspace is None else workspace.get(buf_key)
        if (cached is None or cached[0].shape[0] < total_q
                or cached[0].shape[2] < num_splits):
            cap_q = max(total_q, 64 if cached is None else cached[0].shape[0])
            cap_s = _SPLITKV_MAX_SPLITS
            buf_out = torch.empty((cap_q, num_kv_heads, cap_s, group, head_dim),
                                  dtype=torch.float32, device=q3.device)
            buf_lse = torch.empty((cap_q, num_kv_heads, cap_s, group),
                                  dtype=torch.float32, device=q3.device)
            if workspace is not None:
                workspace[buf_key] = (buf_out, buf_lse)
        else:
            buf_out, buf_lse = cached
        part_out = buf_out[:total_q]
        part_lse = buf_lse[:total_q]
        po_strides = part_out.stride()[:4]
        pl_strides = part_lse.stride()[:3]

    pdl = pdl_enabled(q3.device)
    part_args = (
        q3, kp, vp, cu_q, seqused_k, block_table, out, part_out, part_lse,
        q3.stride(0), q3.stride(1), out.stride(0), out.stride(1),
        kp.stride(0), kp.stride(1), kp.stride(2),
        vp.stride(0), vp.stride(1), vp.stride(2),
        block_table.stride(0), *po_strides, *pl_strides,
        scale_log2, int(max_q), int(num_splits),
        int(window_left), int(window_right),
    )
    # Fixed data tensor strides and alignments stay in the compile key so any
    # non-standard caller still compiles a matching binary.
    part_key = (
        group, group_pad, head_dim, page_tokens, _SPLITKV_TILE_N,
        has_window, single_split, pdl,
        (q3.data_ptr() & 15) == 0, (out.data_ptr() & 15) == 0,
        (kp.data_ptr() & 15) == 0, (vp.data_ptr() & 15) == 0,
        (q3.stride(0) & 15) == 0, (out.stride(0) & 15) == 0,
        (kp.stride(0) & 15) == 0, (kp.stride(1) & 15) == 0,
        (kp.stride(2) & 15) == 0,
    )
    part_kwargs = dict(
        GROUP=group, GROUP_PAD=group_pad, HEAD_DIM=head_dim,
        PAGE_TOKENS=page_tokens, TILE_N=_SPLITKV_TILE_N,
        HAS_WINDOW=has_window, SINGLE_SPLIT=single_split, PDL=pdl,
        num_warps=4, num_stages=3, launch_pdl=pdl,
    )
    launch_triton(
        _kernels()["splitkv_decode_part"],
        (num_seqs * max_q, num_kv_heads, num_splits), part_args,
        key=part_key, kwargs=part_kwargs, exact_args=False)

    if not single_split:
        reduce_args = (
            part_out, part_lse, out, *po_strides, *pl_strides,
            out.stride(0), out.stride(1), int(num_splits),
        )
        reduce_key = (
            group, head_dim, _SPLITKV_MAX_SPLITS, pdl,
            (out.data_ptr() & 15) == 0, (out.stride(0) & 15) == 0,
        )
        reduce_kwargs = dict(
            GROUP=group, HEAD_DIM=head_dim, MAX_SPLITS=_SPLITKV_MAX_SPLITS,
            PDL=pdl, num_warps=4, launch_pdl=pdl,
        )
        launch_triton(
            _kernels()["splitkv_decode_reduce"], (total_q, num_q_heads),
            reduce_args, key=reduce_key, kwargs=reduce_kwargs,
            exact_args=False)
    return out
