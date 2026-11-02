"""Triton kernels for the LFM3 layers' work around their matrix multiplies.

Each kernel reads its inputs once and writes its outputs once, in
place of several PyTorch operations that each read and write the
whole hidden state. Each rounds where the PyTorch code it replaces
rounds to bf16, so a pass's answers stay those of that code.

- gated_conv: a short convolution block's B * x gate, its causal
  depthwise convolution within each sequence, and its C gate.
- post_norm_add_norm: a block's output norm, the residual add, and the
  next norm of the new residual.
- gate_heads: the sigmoid gate of each attention head's output.
- merge_questions: under set attention, a question row's attention over
  its state and over itself, merged by log-sum-exp.
"""

_KERNELS = {}


def _kernels() -> dict:
    """Compile the kernels on first use; Triton needs a GPU process."""
    if _KERNELS:
        return _KERNELS
    import triton
    import triton.language as tl

    # a row count is not specialized, so each chunk size reuses one
    # compiled kernel
    @triton.jit(do_not_specialize=["rows"])
    def gated_conv(proj_ptr, taps_ptr, starts_ptr, out_ptr, rows, stride_proj,
                   stride_out, C: tl.constexpr, WIDTH: tl.constexpr,
                   BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr):
        # proj holds [B | C | x] per row; out = C * conv(B * x)
        row_block = tl.program_id(0)
        channel_block = tl.program_id(1)
        t = row_block * BLOCK_T + tl.arange(0, BLOCK_T)
        c = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
        row_ok = t < rows
        start = tl.load(starts_ptr + t, mask=row_ok, other=0)
        acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
        for tap in tl.static_range(WIDTH):
            source = t - (WIDTH - 1 - tap)
            ok = row_ok & (source >= start)
            mask = ok[:, None] & (c[None, :] < C)
            offsets = source[:, None].to(tl.int64) * stride_proj + c[None, :]
            b = tl.load(proj_ptr + offsets, mask=mask, other=0.0)
            x = tl.load(proj_ptr + offsets + 2 * C, mask=mask, other=0.0)
            gated = (b * x).to(tl.bfloat16).to(tl.float32)
            weight = tl.load(taps_ptr + c * WIDTH + tap, mask=c < C,
                             other=0.0).to(tl.float32)
            acc += gated * weight[None, :]
        mask = row_ok[:, None] & (c[None, :] < C)
        here = t[:, None].to(tl.int64) * stride_proj + c[None, :]
        gate = tl.load(proj_ptr + here + C, mask=mask, other=0.0)
        out = gate * acc.to(tl.bfloat16)
        tl.store(out_ptr + t[:, None].to(tl.int64) * stride_out + c[None, :],
                 out.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def post_norm_add_norm(x_ptr, hidden_ptr, post_ptr, next_ptr, out_ptr,
                           stride_x, stride_hidden, stride_out, eps,
                           H: tl.constexpr, BLOCK: tl.constexpr,
                           HAS_NEXT: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        mask = offs < H
        x = tl.load(x_ptr + row * stride_x + offs, mask=mask,
                    other=0.0).to(tl.float32)
        weight = tl.load(post_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        # as vLLM's rms_norm: the scaled row is rounded before the weight
        scaled = x * tl.math.rsqrt(tl.sum(x * x, axis=0) / H + eps)
        normed = (scaled.to(tl.bfloat16).to(tl.float32) * weight).to(tl.bfloat16)
        hidden = tl.load(hidden_ptr + row * stride_hidden + offs, mask=mask,
                         other=0.0).to(tl.float32)
        hidden = (hidden + normed.to(tl.float32)).to(tl.bfloat16)
        tl.store(hidden_ptr + row * stride_hidden + offs, hidden, mask=mask)
        if HAS_NEXT:
            total = hidden.to(tl.float32)
            weight = tl.load(next_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            scaled = total * tl.math.rsqrt(tl.sum(total * total, axis=0) / H + eps)
            out = scaled.to(tl.bfloat16).to(tl.float32) * weight
            tl.store(out_ptr + row * stride_out + offs,
                     out.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def gate_heads(out_ptr, gate_ptr, stride_out, stride_gate,
                   HEADS: tl.constexpr, D: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        heads = tl.arange(0, HEADS)
        dims = tl.arange(0, D)
        offsets = row * stride_out + heads[:, None] * D + dims[None, :]
        values = tl.load(out_ptr + offsets).to(tl.float32)
        gate = tl.load(gate_ptr + row * stride_gate + heads).to(tl.float32)
        scale = (1.0 / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + offsets,
                 (values.to(tl.bfloat16).to(tl.float32) * scale[:, None])
                 .to(out_ptr.dtype.element_ty))

    @triton.jit(do_not_specialize=["count"])
    def merge_questions(out_ptr, rows_ptr, a_ptr, lse_a_ptr, b_ptr, lse_b_ptr,
                        count, stride_out, stride_a, stride_b, stride_lse_a,
                        stride_lse_b, HEADS: tl.constexpr, D: tl.constexpr):
        index = tl.program_id(0)
        heads = tl.arange(0, HEADS)
        dims = tl.arange(0, D)
        lse_a = tl.load(lse_a_ptr + heads * stride_lse_a + index)
        lse_b = tl.load(lse_b_ptr + heads * stride_lse_b + index)
        top = tl.maximum(lse_a, lse_b)
        wa = tl.exp(lse_a - top)
        wb = tl.exp(lse_b - top)
        within = heads[:, None] * D + dims[None, :]
        a = tl.load(a_ptr + index * stride_a + within).to(tl.float32)
        b = tl.load(b_ptr + index * stride_b + within).to(tl.float32)
        merged = (a * wa[:, None] + b * wb[:, None]) / (wa + wb)[:, None]
        row = tl.load(rows_ptr + index).to(tl.int64)
        tl.store(out_ptr + row * stride_out + within,
                 merged.to(out_ptr.dtype.element_ty))

    _KERNELS.update(gated_conv=gated_conv, post_norm_add_norm=post_norm_add_norm,
                    gate_heads=gate_heads, merge_questions=merge_questions)
    return _KERNELS


def sequence_starts(torch, lengths, device):
    """Each packed row's first row of its sequence, as int32 on the device."""
    import numpy as np

    lengths = np.asarray(lengths, dtype=np.int64)
    firsts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    return torch.from_numpy(np.repeat(firsts, lengths).astype(np.int32)).to(device)


def gated_conv(projected, taps, starts):
    """C * conv(B * x) of a short convolution block, within each sequence.

    Args:
        projected: (rows, 3 * channels) input projection: B, C, then x.
        taps: (channels, width) convolution weights.
        starts: Per row, the first row of its sequence (sequence_starts).

    Returns:
        (rows, channels) bf16 outputs, B * x rounded to bf16 before the
        convolution and the convolution rounded before the C gate.
    """
    import torch

    rows, width3 = projected.shape
    channels = width3 // 3
    out = torch.empty((rows, channels), dtype=projected.dtype,
                      device=projected.device)
    block_t, block_c = 32, 128
    grid = ((rows + block_t - 1) // block_t, (channels + block_c - 1) // block_c)
    _kernels()["gated_conv"][grid](
        projected, taps, starts, out, rows, projected.stride(0), out.stride(0),
        C=channels, WIDTH=taps.shape[1], BLOCK_T=block_t, BLOCK_C=block_c,
        num_warps=4)
    return out


def post_norm_add_norm(x, hidden, post_weight, next_weight, eps):
    """Hidden += norm(x) in place; return the next norm of hidden.

    Args:
        x: (rows, width) block output.
        hidden: (rows, width) residual stream, updated in place.
        post_weight: The block output norm's weight.
        next_weight: The next norm's weight, or None to skip it.
        eps: The norms' variance epsilon.

    Returns:
        The next norm's output, or None.
    """
    import torch

    rows, width = x.shape
    out = None if next_weight is None else torch.empty_like(hidden)
    _kernels()["post_norm_add_norm"][(rows,)](
        x, hidden, post_weight,
        post_weight if next_weight is None else next_weight,
        hidden if out is None else out, x.stride(0), hidden.stride(0),
        hidden.stride(0) if out is None else out.stride(0), eps, H=width,
        BLOCK=1 << (width - 1).bit_length(), HAS_NEXT=next_weight is not None,
        num_warps=8)
    return out


def gate_heads(out, gate):
    """Out (rows, heads * dim) *= sigmoid(gate) per head, in place.

    Args:
        out: Contiguous attention output.
        gate: (rows, heads) gate logits; any row stride.
    """
    rows, heads = gate.shape
    _kernels()["gate_heads"][(rows,)](
        out, gate, out.stride(0), gate.stride(0), HEADS=heads,
        D=out.shape[1] // heads, num_warps=4)
    return out


def merge_questions(out, rows, over_state, lse_state, own, lse_own):
    """Write each question row's two attention reads, merged, into out.

    Args:
        out: (all rows, heads, dim) attention output.
        rows: The question rows' indices in out.
        over_state: (questions, heads, dim) reads of the state.
        lse_state: (heads, questions) their log-sum-exp.
        own: (questions, heads, dim) reads of the question itself.
        lse_own: (heads, questions) their log-sum-exp.
    """
    count, heads, dim = over_state.shape
    if not count:
        return out
    _kernels()["merge_questions"][(count,)](
        out, rows, over_state, lse_state, own, lse_own, count, out.stride(0),
        over_state.stride(0), own.stride(0), lse_state.stride(0),
        lse_own.stride(0), HEADS=heads, D=dim, num_warps=4)
    return out
