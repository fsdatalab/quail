"""Triton kernels shared by Gated DeltaNet phase execution and decode."""

from quail.backends.quail.executor.triton_launch import launch_triton, pdl_enabled

_KERNELS = {}


def _kernels() -> dict:
    """Compile the Gated DeltaNet Triton kernels on first use."""
    if _KERNELS:
        return _KERNELS
    import triton
    import triton.language as tl

    @triton.jit
    def gather_gdn_inputs(
        qkv_ptr, a_ptr, b_ptr, rows_ptr, qkv_out_ptr, a_out_ptr, b_out_ptr,
        stride_qkv, stride_a, stride_b, stride_qo, stride_ao, stride_bo,
        CONV_DIM: tl.constexpr, HV: tl.constexpr, QKV_BLOCKS: tl.constexpr,
        BLOCK: tl.constexpr, HV_BLOCK: tl.constexpr,
    ):
        dst_row = tl.program_id(0).to(tl.int64)
        col_block = tl.program_id(1)
        src_row = tl.load(rows_ptr + dst_row).to(tl.int64)
        if col_block < QKV_BLOCKS:
            offs = col_block * BLOCK + tl.arange(0, BLOCK)
            mask = offs < CONV_DIM
            vals = tl.load(qkv_ptr + src_row * stride_qkv + offs,
                           mask=mask, other=0.0)
            tl.store(qkv_out_ptr + dst_row * stride_qo + offs, vals, mask=mask)
        else:
            hv_offs = tl.arange(0, HV_BLOCK)
            hv_mask = hv_offs < HV
            a_vals = tl.load(a_ptr + src_row * stride_a + hv_offs,
                             mask=hv_mask, other=0.0)
            b_vals = tl.load(b_ptr + src_row * stride_b + hv_offs,
                             mask=hv_mask, other=0.0)
            tl.store(a_out_ptr + dst_row * stride_ao + hv_offs,
                     a_vals, mask=hv_mask)
            tl.store(b_out_ptr + dst_row * stride_bo + hv_offs,
                     b_vals, mask=hv_mask)

    @triton.jit
    def scatter_gdn_rows(
        src_ptr, rows_ptr, dst_ptr, stride_src, stride_dst,
        WIDTH: tl.constexpr, BLOCK: tl.constexpr,
    ):
        src_row = tl.program_id(0).to(tl.int64)
        col_block = tl.program_id(1)
        dst_row = tl.load(rows_ptr + src_row).to(tl.int64)
        offs = col_block * BLOCK + tl.arange(0, BLOCK)
        mask = offs < WIDTH
        vals = tl.load(src_ptr + src_row * stride_src + offs,
                       mask=mask, other=0.0)
        tl.store(dst_ptr + dst_row * stride_dst + offs, vals, mask=mask)

    @triton.jit
    def copy_slots(pool_ptr, src_ptr, dst_ptr, stride_slot,
                   WIDTH: tl.constexpr, BLOCK: tl.constexpr):
        pair = tl.program_id(0)
        offs = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < WIDTH
        src = tl.load(src_ptr + pair).to(tl.int64)
        dst = tl.load(dst_ptr + pair).to(tl.int64)
        vals = tl.load(pool_ptr + src * stride_slot + offs, mask=mask)
        tl.store(pool_ptr + dst * stride_slot + offs, vals, mask=mask)

    # PDL invariant: loads before gdc_wait read only model weights (a_log,
    # dt_bias) and host-staged index arrays (state_rows, out_rows).
    @triton.jit
    def gdn_decode_step(
        x_ptr, a_ptr, b_ptr, a_log_ptr, dt_bias_ptr, state_ptr, state_rows_ptr,
        out_ptr, out_rows_ptr, stride_x, stride_a, stride_b, stride_state,
        stride_out, scale, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
        V: tl.constexpr, BV: tl.constexpr, HAS_OUT_ROWS: tl.constexpr,
        PDL: tl.constexpr,
    ):
        v_block = tl.program_id(0)
        seq_head = tl.program_id(1)
        seq = (seq_head // HV).to(tl.int64)
        hv = seq_head % HV
        h = hv // (HV // H)
        state_row = tl.load(state_rows_ptr + seq).to(tl.int64)
        dt_bias = tl.load(dt_bias_ptr + hv).to(tl.float32)
        a_log = tl.load(a_log_ptr + hv).to(tl.float32)
        out_row = tl.load(out_rows_ptr + seq).to(tl.int64) if HAS_OUT_ROWS else seq
        offs_k = tl.arange(0, K)
        offs_v = v_block * BV + tl.arange(0, BV)
        p_state = (state_ptr + state_row * stride_state + hv * V * K
                   + offs_v[:, None] * K + offs_k[None, :])
        if PDL:
            tl.extra.cuda.gdc_wait()
        row = x_ptr + seq * stride_x
        q = tl.load(row + h * K + offs_k).to(tl.float32)
        k = tl.load(row + H * K + h * K + offs_k).to(tl.float32)
        v = tl.load(row + 2 * H * K + hv * V + offs_v).to(tl.float32)
        # The bf16 rounding matches the separate prep kernel's stored q and k.
        q = (q / tl.sqrt(tl.sum(q * q) + 1e-6)).to(tl.bfloat16).to(tl.float32)
        k = (k / tl.sqrt(tl.sum(k * k) + 1e-6)).to(tl.bfloat16).to(tl.float32)
        q = q * scale
        x = tl.load(a_ptr + seq * stride_a + hv).to(tl.float32) + dt_bias
        softplus = tl.where(x > 0, x + tl.log(1.0 + tl.exp(-x)),
                            tl.log(1.0 + tl.exp(x)))
        softplus = tl.where(x <= 20.0, softplus, x)
        g = -tl.exp(a_log) * softplus
        beta = tl.sigmoid(tl.load(b_ptr + seq * stride_b + hv).to(tl.float32))
        state = tl.load(p_state) * tl.exp(g)
        v = (v - tl.sum(state * k[None, :], 1)) * beta
        state += v[:, None] * k[None, :]
        o = tl.sum(state * q[None, :], 1)
        if PDL:
            tl.extra.cuda.gdc_launch_dependents()
        tl.store(p_state, state)
        tl.store(out_ptr + out_row * stride_out + hv * V + offs_v,
                 o.to(out_ptr.dtype.element_ty))

    _KERNELS.update(
        gather_gdn_inputs=gather_gdn_inputs,
        scatter_gdn_rows=scatter_gdn_rows,
        copy_slots=copy_slots,
        gdn_decode_step=gdn_decode_step,
    )
    return _KERNELS


def gather_gdn_inputs(qkv, a, b, rows):
    """Gather qkv, a, and b at selected row indices in one vectorized launch."""
    import torch

    count = rows.shape[0]
    conv_dim = qkv.shape[1]
    hv = a.shape[1]
    qkv_out = torch.empty((count, conv_dim), dtype=qkv.dtype, device=qkv.device)
    a_out = torch.empty((count, hv), dtype=a.dtype, device=a.device)
    b_out = torch.empty((count, hv), dtype=b.dtype, device=b.device)
    if count == 0:
        return qkv_out, a_out, b_out
    block = 1024
    qkv_blocks = (conv_dim + block - 1) // block
    hv_block = max(16, 1 << (hv - 1).bit_length())
    args = (
        qkv, a, b, rows, qkv_out, a_out, b_out, qkv.stride(0), a.stride(0),
        b.stride(0), qkv_out.stride(0), a_out.stride(0), b_out.stride(0),
    )
    key = (conv_dim, hv, qkv_blocks, block, hv_block)
    kwargs = dict(
        CONV_DIM=conv_dim, HV=hv, QKV_BLOCKS=qkv_blocks, BLOCK=block,
        HV_BLOCK=hv_block, num_warps=4,
    )
    launch_triton(
        _kernels()["gather_gdn_inputs"], (count, qkv_blocks + 1), args,
        key=key, kwargs=kwargs)
    return qkv_out, a_out, b_out


def scatter_gdn_rows(core, rows, out):
    """Scatter 3D core recurrence outputs into out at selected row indices."""
    count = rows.shape[0]
    if count == 0:
        return out
    core_2d = core.reshape(count, -1)
    out_2d = out.view(out.shape[0], -1)
    width = core_2d.shape[1]
    block = 1024
    grid = (count, (width + block - 1) // block)
    args = (core_2d, rows, out_2d, core_2d.stride(0), out_2d.stride(0))
    launch_triton(
        _kernels()["scatter_gdn_rows"], grid, args,
        key=(width, block), kwargs=dict(WIDTH=width, BLOCK=block, num_warps=4))
    return out


def copy_slots(pool, src, dst):
    """Copy pool[src[i]] into pool[dst[i]]; no dst may also be a src.

    Args:
        pool: (slots, ...) tensor whose slots hold contiguous values; only
            dimension 0 may be strided.
        src: (pairs,) source slots.
        dst: (pairs,) destination slots.

    Returns:
        pool.
    """
    count = src.shape[0]
    if count == 0:
        return pool
    width = pool[0].numel()
    if not pool[0].is_contiguous():
        raise ValueError("each slot must hold contiguous values")
    block = 4096
    args = (pool, src, dst, pool.stride(0))
    launch_triton(
        _kernels()["copy_slots"], (count, (width + block - 1) // block), args,
        key=(width, block), kwargs=dict(WIDTH=width, BLOCK=block, num_warps=8))
    return pool


def gdn_decode_step(conv_out, a, b, a_log, dt_bias, state_pool, state_rows, out,
                    *, num_key_heads, scale, out_rows=None, block_v=8,
                    num_warps=1):
    """Advance one-token Gated DeltaNet sequences and write their outputs.

    One launch reads the convolved q, k, and v columns in place, L2-normalizes
    q and k, computes the decay and write gates, updates each sequence's fp32
    state row in place, and writes its output row.

    Args:
        conv_out: (sequences, conv channels) convolution output, unit inner
            stride.
        a: (sequences, value heads) decay gate inputs.
        b: (sequences, value heads) write gate inputs.
        a_log: (value heads,) log decay rates.
        dt_bias: (value heads,) decay biases.
        state_pool: (state rows, value heads, value dim, key dim) fp32 states
            of one layer; only dimension 0 may be strided.
        state_rows: (sequences,) state row of each sequence.
        out: (rows, value heads, value dim) layer output.
        num_key_heads: Query and key heads.
        scale: Query scale.
        out_rows: Optional (sequences,) output row of each sequence; None
            writes sequence s to row s.
        block_v: Value dims each program updates.
        num_warps: Warps per program.

    Returns:
        out.
    """
    count = state_rows.shape[0]
    if count == 0:
        return out
    _, value_heads, value_dim, key_dim = state_pool.shape
    if state_pool.stride()[1:] != (value_dim * key_dim, key_dim, 1):
        raise ValueError("each state row must hold contiguous heads")
    out_2d = out.view(out.shape[0], -1)
    has_out_rows = out_rows is not None
    pdl = pdl_enabled(conv_out.device)
    grid = (value_dim // block_v, count * value_heads)
    args = (
        conv_out, a, b, a_log, dt_bias, state_pool, state_rows, out_2d,
        state_rows if not has_out_rows else out_rows,
        conv_out.stride(0), a.stride(0), b.stride(0), state_pool.stride(0),
        out_2d.stride(0), scale,
    )
    key = (num_key_heads, value_heads, key_dim, value_dim, block_v,
           has_out_rows, num_warps, pdl)
    kwargs = dict(
        H=num_key_heads, HV=value_heads, K=key_dim, V=value_dim, BV=block_v,
        HAS_OUT_ROWS=has_out_rows, PDL=pdl, num_warps=num_warps, launch_pdl=pdl,
    )
    launch_triton(
        _kernels()["gdn_decode_step"], grid, args, key=key, kwargs=kwargs)
    return out
