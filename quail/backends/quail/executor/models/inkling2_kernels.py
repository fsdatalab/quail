"""Triton kernels for the Inkling 2 Mini forward pass.

Each kernel rounds to bf16 where the transformers reference rounds, so
the pass computes what the reference computes up to accumulation order.

- mix: a block attention residual. Per row, a softmax over the
  completed blocks and the running partial block, scored by a learned
  query against each state's RMS-normed values, then the weighted sum
  of the normed states, optionally RMS-normed with a weight.
- gate_out: a loop message gate. The message is a softmax mix of the
  previous pass's last block and output; the gate mixes it into the
  hidden state with tanh gains from a rank-64 projection of both, then
  RMS-norms the result. The projection is rebuilt from products a loop
  pass forms once and each row's norms.
- short_conv: a static causal depthwise convolution of four taps with
  its residual and scale, whose earlier taps come from a tap table
  (conv_taps), optionally added to a residual stream.
- kv_conv: the key or value convolution of one attention layer, static
  or with per-token taps, then the key's L2 clip or the value's RMS norm.
- clip_heads: a query's or relative state's scaled L2 clip per head,
  with log scaling of long positions on the full-attention layers.
- rel_attention: causal paged attention with a learned relative bias
  over the last 1,024 distances, a per-token sink logit, and an
  optional sliding window.
- prefill_attention: the same attention of longer sequences on
  FlashAttention 4, patched to read the bias tiles bias_tiles writes,
  then the sink's share of each row (sink_gate).
- gate_attention: the sigmoid output gate of every head.
- route: each token's top experts and their normalized scores.
- quantize_tokens, swiglu_quantize, combine: the expert path's
  activation quantization (four-over-six scales), its softcapped
  SwiGLU, and the weighted sum of each token's experts, for passes
  that do not run FlashInfer's fused MoE.
"""

import math

# a tap source in a conv tap table: a chunk row (>= 0), zero (-1), or a
# saved state column of slot s, tap j (-(2 + 3 * s + j))
ZERO_TAP = -1

_KERNELS = {}
# the FA4 score_mod of each KV head count
_MARKERS = {}


def _pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def _kernels() -> dict:
    """Compile the kernels on first use; Triton needs a GPU process."""
    if _KERNELS:
        return _KERNELS
    import triton
    import triton.language as tl

    @triton.jit
    def _tap(x_ptr, stride_x, state_ptr, stride_slot, stride_tap, src, cols,
             mask):
        is_row = src >= 0
        old = -src - 2
        is_old = src <= -2
        slot = tl.where(is_old, old // 3, 0).to(tl.int64)
        tap = tl.where(is_old, old % 3, 0).to(tl.int64)
        row = tl.where(is_row, src, 0).to(tl.int64)
        value = tl.load(x_ptr + row * stride_x + cols, mask=mask & is_row,
                        other=0.0).to(tl.float32)
        value += tl.load(state_ptr + slot * stride_slot + tap * stride_tap
                         + cols, mask=mask & is_old, other=0.0).to(tl.float32)
        return value

    @triton.jit
    def _state(blocks_ptr, stride_bb, partial_ptr, addend_ptr, row_b, row_p,
               row_a, cols, J: tl.constexpr, NB: tl.constexpr,
               NS: tl.constexpr, HAS_ADDEND: tl.constexpr):
        # state J of one row: a completed block, the partial (+ addend), or
        # zero past the last state
        if J < NB:
            x = tl.load(blocks_ptr + row_b + J * stride_bb + cols)
            x = x.to(tl.float32)
        elif J < NS:
            x = tl.load(partial_ptr + row_p + cols).to(tl.float32)
            if HAS_ADDEND:
                x += tl.load(addend_ptr + row_a + cols).to(tl.float32)
        else:
            x = tl.zeros(cols.shape, dtype=tl.float32)
        return x

    @triton.jit
    def _softmax5(l0, l1, l2, l3, l4, NS: tl.constexpr):
        top = l0
        if NS > 1:
            top = tl.maximum(top, l1)
        if NS > 2:
            top = tl.maximum(top, l2)
        if NS > 3:
            top = tl.maximum(top, l3)
        if NS > 4:
            top = tl.maximum(top, l4)
        e0 = tl.exp(l0 - top)
        e1 = tl.exp(l1 - top) if NS > 1 else 0.0
        e2 = tl.exp(l2 - top) if NS > 2 else 0.0
        e3 = tl.exp(l3 - top) if NS > 3 else 0.0
        e4 = tl.exp(l4 - top) if NS > 4 else 0.0
        total = e0 + e1 + e2 + e3 + e4
        return e0 / total, e1 / total, e2 / total, e3 / total, e4 / total

    @triton.jit
    def mix(blocks_ptr, stride_bb, stride_br, partial_ptr, stride_p,
            addend_ptr, stride_a, query_ptr, weight_ptr, out_ptr, stride_oq,
            stride_or, squares_ptr, stride_sq, eps, H: tl.constexpr,
            CH: tl.constexpr, NB: tl.constexpr, NS: tl.constexpr,
            NQ: tl.constexpr, HAS_ADDEND: tl.constexpr, NORM: tl.constexpr,
            SQUARES: tl.constexpr):
        # One row of up to five states and three queries. The first pass
        # accumulates the states' pairwise products and each query's
        # products with each state elementwise, reduced once at the end;
        # the RMS norms, the softmax weights, and each output's norm
        # follow from those sums, so the second pass writes finished rows.
        row = tl.program_id(0).to(tl.int64)
        row_b = row * stride_br
        row_p = row * stride_p
        row_a = row * stride_a
        offs = tl.arange(0, CH)
        g00 = tl.zeros((CH,), dtype=tl.float32)
        g01 = tl.zeros((CH,), dtype=tl.float32)
        g02 = tl.zeros((CH,), dtype=tl.float32)
        g03 = tl.zeros((CH,), dtype=tl.float32)
        g04 = tl.zeros((CH,), dtype=tl.float32)
        g11 = tl.zeros((CH,), dtype=tl.float32)
        g12 = tl.zeros((CH,), dtype=tl.float32)
        g13 = tl.zeros((CH,), dtype=tl.float32)
        g14 = tl.zeros((CH,), dtype=tl.float32)
        g22 = tl.zeros((CH,), dtype=tl.float32)
        g23 = tl.zeros((CH,), dtype=tl.float32)
        g24 = tl.zeros((CH,), dtype=tl.float32)
        g33 = tl.zeros((CH,), dtype=tl.float32)
        g34 = tl.zeros((CH,), dtype=tl.float32)
        g44 = tl.zeros((CH,), dtype=tl.float32)
        d00 = tl.zeros((CH,), dtype=tl.float32)
        d01 = tl.zeros((CH,), dtype=tl.float32)
        d02 = tl.zeros((CH,), dtype=tl.float32)
        d03 = tl.zeros((CH,), dtype=tl.float32)
        d04 = tl.zeros((CH,), dtype=tl.float32)
        d10 = tl.zeros((CH,), dtype=tl.float32)
        d11 = tl.zeros((CH,), dtype=tl.float32)
        d12 = tl.zeros((CH,), dtype=tl.float32)
        d13 = tl.zeros((CH,), dtype=tl.float32)
        d14 = tl.zeros((CH,), dtype=tl.float32)
        d20 = tl.zeros((CH,), dtype=tl.float32)
        d21 = tl.zeros((CH,), dtype=tl.float32)
        d22 = tl.zeros((CH,), dtype=tl.float32)
        d23 = tl.zeros((CH,), dtype=tl.float32)
        d24 = tl.zeros((CH,), dtype=tl.float32)
        for c in range(0, H, CH):
            cols = c + offs
            x0 = _state(blocks_ptr, stride_bb, partial_ptr, addend_ptr, row_b,
                        row_p, row_a, cols, 0, NB, NS, HAS_ADDEND)
            x1 = _state(blocks_ptr, stride_bb, partial_ptr, addend_ptr, row_b,
                        row_p, row_a, cols, 1, NB, NS, HAS_ADDEND)
            x2 = _state(blocks_ptr, stride_bb, partial_ptr, addend_ptr, row_b,
                        row_p, row_a, cols, 2, NB, NS, HAS_ADDEND)
            x3 = _state(blocks_ptr, stride_bb, partial_ptr, addend_ptr, row_b,
                        row_p, row_a, cols, 3, NB, NS, HAS_ADDEND)
            x4 = _state(blocks_ptr, stride_bb, partial_ptr, addend_ptr, row_b,
                        row_p, row_a, cols, 4, NB, NS, HAS_ADDEND)
            if NQ > 0:
                q0 = tl.load(query_ptr + 0 * H + cols)
                if NS > 0:
                    d00 += q0 * x0
                if NS > 1:
                    d01 += q0 * x1
                if NS > 2:
                    d02 += q0 * x2
                if NS > 3:
                    d03 += q0 * x3
                if NS > 4:
                    d04 += q0 * x4
            if NQ > 1:
                q1 = tl.load(query_ptr + 1 * H + cols)
                if NS > 0:
                    d10 += q1 * x0
                if NS > 1:
                    d11 += q1 * x1
                if NS > 2:
                    d12 += q1 * x2
                if NS > 3:
                    d13 += q1 * x3
                if NS > 4:
                    d14 += q1 * x4
            if NQ > 2:
                q2 = tl.load(query_ptr + 2 * H + cols)
                if NS > 0:
                    d20 += q2 * x0
                if NS > 1:
                    d21 += q2 * x1
                if NS > 2:
                    d22 += q2 * x2
                if NS > 3:
                    d23 += q2 * x3
                if NS > 4:
                    d24 += q2 * x4
            if NS > 0:
                g00 += x0 * x0
            if NS > 1:
                g01 += x0 * x1
            if NS > 2:
                g02 += x0 * x2
            if NS > 3:
                g03 += x0 * x3
            if NS > 4:
                g04 += x0 * x4
            if NS > 1:
                g11 += x1 * x1
            if NS > 2:
                g12 += x1 * x2
            if NS > 3:
                g13 += x1 * x3
            if NS > 4:
                g14 += x1 * x4
            if NS > 2:
                g22 += x2 * x2
            if NS > 3:
                g23 += x2 * x3
            if NS > 4:
                g24 += x2 * x4
            if NS > 3:
                g33 += x3 * x3
            if NS > 4:
                g34 += x3 * x4
            if NS > 4:
                g44 += x4 * x4
        # the reductions, once
        s00 = tl.sum(g00, axis=0)
        s01 = tl.sum(g01, axis=0)
        s02 = tl.sum(g02, axis=0)
        s03 = tl.sum(g03, axis=0)
        s04 = tl.sum(g04, axis=0)
        s11 = tl.sum(g11, axis=0)
        s12 = tl.sum(g12, axis=0)
        s13 = tl.sum(g13, axis=0)
        s14 = tl.sum(g14, axis=0)
        s22 = tl.sum(g22, axis=0)
        s23 = tl.sum(g23, axis=0)
        s24 = tl.sum(g24, axis=0)
        s33 = tl.sum(g33, axis=0)
        s34 = tl.sum(g34, axis=0)
        s44 = tl.sum(g44, axis=0)
        i0 = tl.math.rsqrt(s00 / H + eps)
        i1 = tl.math.rsqrt(s11 / H + eps)
        i2 = tl.math.rsqrt(s22 / H + eps)
        i3 = tl.math.rsqrt(s33 / H + eps)
        i4 = tl.math.rsqrt(s44 / H + eps)
        if NQ > 0:
            p00, p01, p02, p03, p04 = _softmax5(
                tl.sum(d00, axis=0) * i0, tl.sum(d01, axis=0) * i1,
                tl.sum(d02, axis=0) * i2, tl.sum(d03, axis=0) * i3,
                tl.sum(d04, axis=0) * i4, NS)
            w00 = p00 * i0
            w01 = p01 * i1
            w02 = p02 * i2
            w03 = p03 * i3
            w04 = p04 * i4
            if NORM or SQUARES:
                norm_sq = (w00 * w00 * s00
                           + w00 * w01 * s01
                           + w00 * w02 * s02
                           + w00 * w03 * s03
                           + w00 * w04 * s04
                           + w01 * w00 * s01
                           + w01 * w01 * s11
                           + w01 * w02 * s12
                           + w01 * w03 * s13
                           + w01 * w04 * s14
                           + w02 * w00 * s02
                           + w02 * w01 * s12
                           + w02 * w02 * s22
                           + w02 * w03 * s23
                           + w02 * w04 * s24
                           + w03 * w00 * s03
                           + w03 * w01 * s13
                           + w03 * w02 * s23
                           + w03 * w03 * s33
                           + w03 * w04 * s34
                           + w04 * w00 * s04
                           + w04 * w01 * s14
                           + w04 * w02 * s24
                           + w04 * w03 * s34
                           + w04 * w04 * s44)
            if SQUARES:
                tl.store(squares_ptr + 0 * stride_sq + row, norm_sq)
            if NORM:
                scale = tl.math.rsqrt(norm_sq / H + eps)
                w00 = w00 * scale
                w01 = w01 * scale
                w02 = w02 * scale
                w03 = w03 * scale
                w04 = w04 * scale
        if NQ > 1:
            p10, p11, p12, p13, p14 = _softmax5(
                tl.sum(d10, axis=0) * i0, tl.sum(d11, axis=0) * i1,
                tl.sum(d12, axis=0) * i2, tl.sum(d13, axis=0) * i3,
                tl.sum(d14, axis=0) * i4, NS)
            w10 = p10 * i0
            w11 = p11 * i1
            w12 = p12 * i2
            w13 = p13 * i3
            w14 = p14 * i4
            if NORM or SQUARES:
                norm_sq = (w10 * w10 * s00
                           + w10 * w11 * s01
                           + w10 * w12 * s02
                           + w10 * w13 * s03
                           + w10 * w14 * s04
                           + w11 * w10 * s01
                           + w11 * w11 * s11
                           + w11 * w12 * s12
                           + w11 * w13 * s13
                           + w11 * w14 * s14
                           + w12 * w10 * s02
                           + w12 * w11 * s12
                           + w12 * w12 * s22
                           + w12 * w13 * s23
                           + w12 * w14 * s24
                           + w13 * w10 * s03
                           + w13 * w11 * s13
                           + w13 * w12 * s23
                           + w13 * w13 * s33
                           + w13 * w14 * s34
                           + w14 * w10 * s04
                           + w14 * w11 * s14
                           + w14 * w12 * s24
                           + w14 * w13 * s34
                           + w14 * w14 * s44)
            if SQUARES:
                tl.store(squares_ptr + 1 * stride_sq + row, norm_sq)
            if NORM:
                scale = tl.math.rsqrt(norm_sq / H + eps)
                w10 = w10 * scale
                w11 = w11 * scale
                w12 = w12 * scale
                w13 = w13 * scale
                w14 = w14 * scale
        if NQ > 2:
            p20, p21, p22, p23, p24 = _softmax5(
                tl.sum(d20, axis=0) * i0, tl.sum(d21, axis=0) * i1,
                tl.sum(d22, axis=0) * i2, tl.sum(d23, axis=0) * i3,
                tl.sum(d24, axis=0) * i4, NS)
            w20 = p20 * i0
            w21 = p21 * i1
            w22 = p22 * i2
            w23 = p23 * i3
            w24 = p24 * i4
            if NORM or SQUARES:
                norm_sq = (w20 * w20 * s00
                           + w20 * w21 * s01
                           + w20 * w22 * s02
                           + w20 * w23 * s03
                           + w20 * w24 * s04
                           + w21 * w20 * s01
                           + w21 * w21 * s11
                           + w21 * w22 * s12
                           + w21 * w23 * s13
                           + w21 * w24 * s14
                           + w22 * w20 * s02
                           + w22 * w21 * s12
                           + w22 * w22 * s22
                           + w22 * w23 * s23
                           + w22 * w24 * s24
                           + w23 * w20 * s03
                           + w23 * w21 * s13
                           + w23 * w22 * s23
                           + w23 * w23 * s33
                           + w23 * w24 * s34
                           + w24 * w20 * s04
                           + w24 * w21 * s14
                           + w24 * w22 * s24
                           + w24 * w23 * s34
                           + w24 * w24 * s44)
            if SQUARES:
                tl.store(squares_ptr + 2 * stride_sq + row, norm_sq)
            if NORM:
                scale = tl.math.rsqrt(norm_sq / H + eps)
                w20 = w20 * scale
                w21 = w21 * scale
                w22 = w22 * scale
                w23 = w23 * scale
                w24 = w24 * scale
        for c in range(0, H, CH):
            cols = c + offs
            x0 = _state(blocks_ptr, stride_bb, partial_ptr, addend_ptr, row_b,
                        row_p, row_a, cols, 0, NB, NS, HAS_ADDEND)
            x1 = _state(blocks_ptr, stride_bb, partial_ptr, addend_ptr, row_b,
                        row_p, row_a, cols, 1, NB, NS, HAS_ADDEND)
            x2 = _state(blocks_ptr, stride_bb, partial_ptr, addend_ptr, row_b,
                        row_p, row_a, cols, 2, NB, NS, HAS_ADDEND)
            x3 = _state(blocks_ptr, stride_bb, partial_ptr, addend_ptr, row_b,
                        row_p, row_a, cols, 3, NB, NS, HAS_ADDEND)
            x4 = _state(blocks_ptr, stride_bb, partial_ptr, addend_ptr, row_b,
                        row_p, row_a, cols, 4, NB, NS, HAS_ADDEND)
            if NQ > 0:
                out = (w00 * x0 + w01 * x1 + w02 * x2 + w03 * x3
                       + w04 * x4)
                if NORM:
                    out *= tl.load(weight_ptr + 0 * H + cols).to(tl.float32)
                tl.store(out_ptr + 0 * stride_oq + row * stride_or + cols,
                         out.to(out_ptr.dtype.element_ty))
            if NQ > 1:
                out = (w10 * x0 + w11 * x1 + w12 * x2 + w13 * x3
                       + w14 * x4)
                if NORM:
                    out *= tl.load(weight_ptr + 1 * H + cols).to(tl.float32)
                tl.store(out_ptr + 1 * stride_oq + row * stride_or + cols,
                         out.to(out_ptr.dtype.element_ty))
            if NQ > 2:
                out = (w20 * x0 + w21 * x1 + w22 * x2 + w23 * x3
                       + w24 * x4)
                if NORM:
                    out *= tl.load(weight_ptr + 2 * H + cols).to(tl.float32)
                tl.store(out_ptr + 2 * stride_oq + row * stride_or + cols,
                         out.to(out_ptr.dtype.element_ty))

    @triton.jit
    def _blend(aa, bb, ab, qa, qb, eps, H: tl.constexpr):
        # the message (alpha a + beta b) of the previous pass's last block a
        # and output b: their RMS-normed softmax blend by query score;
        # returns alpha, beta, and the message's squared sum
        inv_a = tl.math.rsqrt(aa / H + eps)
        inv_b = tl.math.rsqrt(bb / H + eps)
        sa = qa * inv_a
        sb = qb * inv_b
        top = tl.maximum(sa, sb)
        ea = tl.exp(sa - top)
        eb = tl.exp(sb - top)
        alpha = ea * inv_a / (ea + eb)
        beta = eb * inv_b / (ea + eb)
        mm = alpha * alpha * aa + 2.0 * alpha * beta * ab + beta * beta * bb
        return alpha, beta, mm

    @triton.jit
    def _gate_tile(d, gains, hidden, block, prev, alpha, beta, cols, offs_r,
                   ok, H: tl.constexpr, R: tl.constexpr):
        # (1 - tanh f) h + tanh(g) (alpha a + beta b) over columns cols
        f = tl.dot(d, tl.load(gains + offs_r[:, None] * (2 * H)
                              + cols[None, :]))
        g = tl.dot(d, tl.load(gains + offs_r[:, None] * (2 * H) + H
                              + cols[None, :]))
        h = tl.load(hidden + cols[None, :], mask=ok[:, None],
                    other=0.0).to(tl.float32)
        a = tl.load(block + cols[None, :], mask=ok[:, None],
                    other=0.0).to(tl.float32)
        b = tl.load(prev + cols[None, :], mask=ok[:, None],
                    other=0.0).to(tl.float32)
        m = alpha[:, None] * a + beta[:, None] * b
        # the row is rounded to bf16, so the approximate tanh suffices
        return (1.0 - _tanh_approx(f)) * h + _tanh_approx(g) * m

    @triton.jit
    def gate_out(hidden_ptr, stride_hg, stride_h, squares_ptr, stride_sg,
                 prev_block_ptr, prev_out_ptr, stride_prev, ph_ptr, stride_pg,
                 stride_p, pa_ptr, pb_ptr, stride_m, scores_ptr, stride_s,
                 stats_ptr, gains_ptr, weight_ptr, out_ptr, stride_og,
                 stride_o, rows, eps, H: tl.constexpr, R: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr):
        # One gate (program_id(1)) of BM rows. Its down projection is
        # rebuilt from h @ Dh, a @ Dm, and b @ Dm with each row's norms;
        # the gains GEMM (K = R) then runs on two passes: the first sums
        # the gated row's squares, the second writes it RMS-normed.
        r = tl.program_id(0) * BM + tl.arange(0, BM)
        gate = tl.program_id(1)
        ok = r < rows
        r64 = r.to(tl.int64)
        offs_c = tl.arange(0, BN)
        offs_r = tl.arange(0, R)
        hidden = hidden_ptr + gate * stride_hg + r64[:, None] * stride_h
        block = prev_block_ptr + r64[:, None] * stride_prev
        prev = prev_out_ptr + r64[:, None] * stride_prev
        gains = gains_ptr + gate * (R * 2 * H)
        stats = stats_ptr + r64 * 3
        aa = tl.load(stats, mask=ok, other=1.0)
        bb = tl.load(stats + 1, mask=ok, other=1.0)
        ab = tl.load(stats + 2, mask=ok, other=0.0)
        scores = scores_ptr + r64 * stride_s + gate * 2
        alpha, beta, mm = _blend(aa, bb, ab,
                                 tl.load(scores, mask=ok, other=0.0),
                                 tl.load(scores + 1, mask=ok, other=0.0),
                                 eps, H)
        hh = tl.load(squares_ptr + gate * stride_sg + r64, mask=ok, other=1.0)
        inv_h = tl.math.rsqrt(hh / H + eps)
        inv_m = tl.math.rsqrt(mm / H + eps)
        cols_r = gate * R + offs_r
        ph = tl.load(ph_ptr + gate * stride_pg + r64[:, None] * stride_p
                     + offs_r[None, :], mask=ok[:, None], other=0.0)
        pa = tl.load(pa_ptr + r64[:, None] * stride_m + cols_r[None, :],
                     mask=ok[:, None], other=0.0)
        pb = tl.load(pb_ptr + r64[:, None] * stride_m + cols_r[None, :],
                     mask=ok[:, None], other=0.0)
        # [norm(h) | norm(m)] @ down, rounded to bf16 as the reference's is
        d = (inv_h[:, None] * ph + inv_m[:, None] * (
            alpha[:, None] * pa + beta[:, None] * pb)).to(tl.bfloat16)
        total = tl.zeros((BM,), dtype=tl.float32)
        for n in range(0, H, BN):
            y = _gate_tile(d, gains, hidden, block, prev, alpha, beta,
                           n + offs_c, offs_r, ok, H, R)
            total += tl.sum(y * y, 1)
        scale = tl.math.rsqrt(total / H + eps)
        out = out_ptr + gate * stride_og + r64[:, None] * stride_o
        for n in range(0, H, BN):
            cols = n + offs_c
            y = _gate_tile(d, gains, hidden, block, prev, alpha, beta, cols,
                           offs_r, ok, H, R)
            w = tl.load(weight_ptr + gate * H + cols).to(tl.float32)
            tl.store(out + cols[None, :],
                     (y * scale[:, None] * w[None, :]).to(tl.bfloat16),
                     mask=ok[:, None])

    @triton.jit
    def rms_norm(x_ptr, stride_x, weight_ptr, out_ptr, stride_o, eps,
                 H: tl.constexpr, HP: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, HP)
        mask = offs < H
        x = tl.load(x_ptr + row * stride_x + offs, mask=mask,
                    other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x = x * tl.math.rsqrt(tl.sum(x * x, axis=0) / H + eps) * w
        tl.store(out_ptr + row * stride_o + offs, x.to(tl.bfloat16),
                 mask=mask)

    @triton.jit(do_not_specialize=["rows"])
    def short_conv(x_ptr, stride_x, taps_ptr, state_ptr, stride_slot,
                   stride_tap, w_ptr, base_ptr, stride_base, base2_ptr,
                   stride_base2, out_ptr, stride_out, rows, C, inv_scale,
                   BT: tl.constexpr, BC: tl.constexpr,
                   HAS_BASE: tl.constexpr, HAS_BASE2: tl.constexpr):
        # w is (C, 4) with w[:, 3] on the current row
        r = tl.program_id(0) * BT + tl.arange(0, BT)
        cols = tl.program_id(1) * BC + tl.arange(0, BC)
        row_ok = r < rows
        col_ok = cols < C
        mask = row_ok[:, None] & col_ok[None, :]
        r64 = r.to(tl.int64)[:, None]
        x = tl.load(x_ptr + r64 * stride_x + cols[None, :], mask=mask,
                    other=0.0).to(tl.float32)
        acc = x * tl.load(w_ptr + cols * 4 + 3, mask=col_ok, other=0.0)[None, :]
        for j in tl.static_range(3):
            src = tl.load(taps_ptr + r * 3 + j, mask=row_ok, other=-1)
            acc += tl.load(w_ptr + cols * 4 + j, mask=col_ok,
                           other=0.0)[None, :] * _tap(
                x_ptr, stride_x, state_ptr, stride_slot, stride_tap,
                src[:, None], cols[None, :], mask)
        y = ((acc + x) * inv_scale).to(tl.bfloat16)
        if HAS_BASE:
            base = tl.load(base_ptr + r64 * stride_base + cols[None, :],
                           mask=mask, other=0.0).to(tl.float32)
            if HAS_BASE2:
                base += tl.load(base2_ptr + r64 * stride_base2
                                + cols[None, :], mask=mask,
                                other=0.0).to(tl.float32)
            y = (base + y.to(tl.float32)).to(tl.bfloat16)
        tl.store(out_ptr + r64 * stride_out + cols[None, :], y, mask=mask)

    @triton.jit
    def _taps(x_ptr, stride_x, state_ptr, stride_slot, stride_tap, src, cols,
              ok):
        # (rows, cols) values of each row's tap source src (rows,): a row
        # of x, zero, or a saved state column
        is_row = (src >= 0)[:, None] & ok[:, None]
        old = -src - 2
        is_old = (src <= -2)[:, None] & ok[:, None]
        slot = tl.where(src <= -2, old // 3, 0).to(tl.int64)
        tap = tl.where(src <= -2, old % 3, 0).to(tl.int64)
        row = tl.where(src >= 0, src, 0).to(tl.int64)
        value = tl.load(x_ptr + row[:, None] * stride_x + cols[None, :],
                        mask=is_row, other=0.0).to(tl.float32)
        value += tl.load(state_ptr + slot[:, None] * stride_slot
                         + tap[:, None] * stride_tap + cols[None, :],
                         mask=is_old, other=0.0).to(tl.float32)
        return value

    @triton.jit(do_not_specialize=["rows"])
    def kv_conv(x_ptr, stride_x, dyn_ptr, stride_dyn, taps_ptr, state_ptr,
                stride_slot, stride_tap, w_ptr, out_ptr, stride_o0, rows, eps,
                clip, inv_scale, HEADS: tl.constexpr, D: tl.constexpr,
                BT: tl.constexpr, DYNAMIC: tl.constexpr,
                IS_KEY: tl.constexpr):
        # BT rows, every head. static: w is (C, 4), w[:, 3] current;
        # dynamic: w is (4, C) by backward distance, plus each row's
        # (heads, 4) taps at dyn_ptr
        C: tl.constexpr = HEADS * D
        r = tl.program_id(0) * BT + tl.arange(0, BT)
        ok = r < rows
        r64 = r.to(tl.int64)
        cols = tl.arange(0, C)
        x = tl.load(x_ptr + r64[:, None] * stride_x + cols[None, :],
                    mask=ok[:, None], other=0.0).to(tl.float32)
        src0 = tl.load(taps_ptr + r64 * 3, mask=ok, other=-1)
        src1 = tl.load(taps_ptr + r64 * 3 + 1, mask=ok, other=-1)
        src2 = tl.load(taps_ptr + r64 * 3 + 2, mask=ok, other=-1)
        t3 = _taps(x_ptr, stride_x, state_ptr, stride_slot, stride_tap, src0,
                   cols, ok)
        t2 = _taps(x_ptr, stride_x, state_ptr, stride_slot, stride_tap, src1,
                   cols, ok)
        t1 = _taps(x_ptr, stride_x, state_ptr, stride_slot, stride_tap, src2,
                   cols, ok)
        if DYNAMIC:
            dyn = dyn_ptr + r64[:, None] * stride_dyn + (cols // D)[None, :] * 4
            w0 = tl.load(w_ptr + cols).to(tl.float32)[None, :] + tl.load(
                dyn, mask=ok[:, None], other=0.0).to(tl.float32)
            w1 = tl.load(w_ptr + C + cols).to(tl.float32)[None, :] + tl.load(
                dyn + 1, mask=ok[:, None], other=0.0).to(tl.float32)
            w2 = tl.load(w_ptr + 2 * C + cols).to(tl.float32)[None, :] + tl.load(
                dyn + 2, mask=ok[:, None], other=0.0).to(tl.float32)
            w3 = tl.load(w_ptr + 3 * C + cols).to(tl.float32)[None, :] + tl.load(
                dyn + 3, mask=ok[:, None], other=0.0).to(tl.float32)
            acc = w0 * x + w1 * t1 + w2 * t2 + w3 * t3
        else:
            acc = (tl.load(w_ptr + cols * 4 + 3)[None, :] * x
                   + tl.load(w_ptr + cols * 4 + 2)[None, :] * t1
                   + tl.load(w_ptr + cols * 4 + 1)[None, :] * t2
                   + tl.load(w_ptr + cols * 4)[None, :] * t3)
        y = ((acc + x) * inv_scale).to(tl.bfloat16).to(tl.float32)
        heads = tl.reshape(y, (BT, HEADS, D))
        sq = tl.sum(heads * heads, axis=2)
        if IS_KEY:
            factor = tl.minimum(clip / (tl.sqrt(sq) + 1e-6), 1.0)
        else:
            factor = tl.math.rsqrt(sq / D + eps)
        y = tl.reshape(heads * factor[:, :, None], (BT, C))
        tl.store(out_ptr + r64[:, None] * stride_o0 + cols[None, :],
                 y.to(tl.bfloat16), mask=ok[:, None])

    @triton.jit(do_not_specialize=["rows"])
    def route(logits_ptr, bias_ptr, ids_ptr, weights_ptr, rows, scale,
              E: tl.constexpr, K: tl.constexpr, BT: tl.constexpr):
        # each row's K experts by sigmoid score plus bias, and their scores
        # normalized to sum to scale; ties go to the lower expert
        r = tl.program_id(0) * BT + tl.arange(0, BT)
        ok = r < rows
        r64 = r.to(tl.int64)
        experts = tl.arange(0, E)
        logits = tl.load(logits_ptr + r64[:, None] * E + experts[None, :],
                         mask=ok[:, None], other=0.0)
        scores = tl.sigmoid(logits)
        choice = scores + tl.load(bias_ptr + experts)[None, :]
        slots = tl.arange(0, K)
        ids = tl.zeros((BT, K), dtype=tl.int32)
        picked = tl.zeros((BT, K), dtype=tl.float32)
        for k in tl.static_range(K):
            best = tl.max(choice, axis=1)
            index = tl.min(tl.where(choice == best[:, None], experts[None, :],
                                    E), axis=1)
            hit = experts[None, :] == index[:, None]
            score = tl.sum(tl.where(hit, scores, 0.0), axis=1)
            ids = tl.where(slots[None, :] == k, index[:, None], ids)
            picked = tl.where(slots[None, :] == k, score[:, None], picked)
            choice = tl.where(hit, -float("inf"), choice)
        picked = picked / (tl.sum(picked, axis=1)[:, None] + 1e-20) * scale
        out = r64[:, None] * K + slots[None, :]
        tl.store(ids_ptr + out, ids, mask=ok[:, None])
        tl.store(weights_ptr + out, picked, mask=ok[:, None])

    @triton.jit
    def clip_heads(x_ptr, stride_x, scale_ptr, positions_ptr, out_ptr,
                   stride_o0, stride_o1, clip, alpha, floor,
                   H: tl.constexpr, HP: tl.constexpr, D: tl.constexpr,
                   LOG_SCALE: tl.constexpr):
        # every head of one row
        row = tl.program_id(0).to(tl.int64)
        heads = tl.arange(0, HP)
        offs = tl.arange(0, D)
        mask = (heads < H)[:, None]
        cols = heads[:, None] * D + offs[None, :]
        x = tl.load(x_ptr + row * stride_x + cols, mask=mask,
                    other=0.0).to(tl.float32)
        x = x * tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        norm = tl.sqrt(tl.sum(x * x, axis=1))
        y = (x * tl.minimum(clip / (norm + 1e-6), 1.0)[:, None]).to(
            tl.bfloat16)
        if LOG_SCALE:
            position = tl.load(positions_ptr + row).to(tl.float32)
            tau = 1.0 + alpha * tl.log(tl.maximum((position + 1.0) / floor,
                                                  1.0))
            y = (y.to(tl.float32) * tau).to(tl.bfloat16)
        tl.store(out_ptr + row * stride_o0 + heads[:, None] * stride_o1
                 + offs[None, :], y, mask=mask)

    @triton.jit
    def gate_attention(attn_ptr, gate_ptr, stride_gate, out_ptr, N,
                       BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < N
        a = tl.load(attn_ptr + row * N + cols, mask=mask, other=0.0)
        g = tl.load(gate_ptr + row * stride_gate + cols, mask=mask,
                    other=0.0).to(tl.float32)
        out = a.to(tl.float32) * tl.sigmoid(g)
        tl.store(out_ptr + row * N + cols, out.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _attend_tile(q, k_ptr, v_ptr, bt_row, n0, kv_head, qpos, row_ok,
                     used, window, bias_rows, m_i, l_i, acc, scale_log2,
                     stride_k0, stride_k1, stride_k2, stride_v0, stride_v1,
                     stride_v2, BN: tl.constexpr, D: tl.constexpr,
                     EXTENT: tl.constexpr, PAGE: tl.constexpr,
                     HAS_WINDOW: tl.constexpr, NEAR: tl.constexpr):
        # one key tile of the online softmax; a far tile's keys are all
        # earlier than every query, past the bias and any window
        LOG2E: tl.constexpr = 1.4426950408889634
        offs_key = tl.arange(0, BN)
        offs_d = tl.arange(0, D)
        page = tl.load(bt_row + n0 // PAGE).to(tl.int64)
        in_page = n0 % PAGE
        kpos = n0 + offs_key
        key_ok = kpos < used
        k = tl.load(k_ptr + page * stride_k0
                    + (in_page + offs_key[:, None]) * stride_k1
                    + kv_head * stride_k2 + offs_d[None, :],
                    mask=key_ok[:, None], other=0.0)
        v = tl.load(v_ptr + page * stride_v0
                    + (in_page + offs_key[:, None]) * stride_v1
                    + kv_head * stride_v2 + offs_d[None, :],
                    mask=key_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale_log2
        if NEAR:
            dist = qpos[:, None] - kpos[None, :]
            valid = (dist >= 0) & key_ok[None, :] & row_ok[:, None]
            if HAS_WINDOW:
                valid = valid & (dist < window)
            # bias_rows points at each row's bias for this tile's first key
            bias = tl.load(bias_rows[:, None] + offs_key[None, :])
            s += tl.where(dist < EXTENT, bias.to(tl.float32) * LOG2E, 0.0)
            s = tl.where(valid, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        # a later split's row that has seen no key keeps m at -inf, and
        # exp2(-inf - -inf) would be NaN
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        return m_new, l_i, acc

    @triton.jit(do_not_specialize=["window", "splits"])
    def rel_attention(q_ptr, k_ptr, v_ptr, rel_ptr, proj_ptr, sink_ptr,
                      out_ptr, scratch_ptr, part_acc_ptr, part_ml_ptr,
                      tile_seq_ptr, tile_q0_ptr, cu_q_ptr, used_ptr,
                      table_ptr, stride_q0, stride_q1, stride_r0, stride_r1,
                      stride_s0, stride_o0, stride_o1, stride_k0, stride_k1,
                      stride_k2, stride_v0, stride_v1, stride_v2, stride_bt,
                      scale_log2, window, splits, GROUP: tl.constexpr,
                      HEADS: tl.constexpr, BM: tl.constexpr,
                      BN: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
                      EXTENT: tl.constexpr, PAGE: tl.constexpr,
                      SPAN: tl.constexpr, WCH: tl.constexpr,
                      HAS_WINDOW: tl.constexpr, SPLIT: tl.constexpr):
        # each program first writes its rows' bias by key position over
        # its near keys to its own (BM, SPAN) scratch, so every near key
        # tile reads an aligned rectangle; reads that start at a different
        # column in each row do not vectorize
        LOG2E: tl.constexpr = 1.4426950408889634
        TQ: tl.constexpr = BM // GROUP
        tile = tl.program_id(0) // splits
        split = tl.program_id(0) % splits
        kv_head = tl.program_id(1)
        seq = tl.load(tile_seq_ptr + tile)
        q0 = tl.load(tile_q0_ptr + tile)
        q_start = tl.load(cu_q_ptr + seq)
        q_len = tl.load(cu_q_ptr + seq + 1) - q_start
        used = tl.load(used_ptr + seq)
        base = used - q_len
        offs_m = tl.arange(0, BM)
        tq = offs_m // GROUP
        g = offs_m % GROUP
        qi = q0 + tq
        row_ok = (tq < TQ) & (qi < q_len)
        q_row = (q_start + qi).to(tl.int64)
        head = kv_head * GROUP + g
        qpos = base + qi
        offs_d = tl.arange(0, D)
        q = tl.load(q_ptr + q_row[:, None] * stride_q0
                    + head[:, None] * stride_q1 + offs_d[None, :],
                    mask=row_ok[:, None], other=0.0)
        sink = tl.load(sink_ptr + q_row * stride_s0 + head, mask=row_ok,
                       other=0.0).to(tl.float32)
        # the sink is one more logit, with a zero value, counted once
        if split == 0:
            m_i = sink * LOG2E
            l_i = tl.zeros((BM,), dtype=tl.float32) + 1.0
        else:
            m_i = tl.zeros((BM,), dtype=tl.float32) - float("inf")
            l_i = tl.zeros((BM,), dtype=tl.float32)
        acc = tl.zeros((BM, D), dtype=tl.float32)
        first_q = base + q0
        last_q = base + tl.minimum(q0 + TQ, q_len) - 1
        lo = first_q * 0
        if HAS_WINDOW:
            lo = tl.maximum(first_q - (window - 1), 0)
        start = (lo // BN) * BN
        # the first tile some query of this tile is near: its last key is
        # within EXTENT - 1 of the first query
        behind = tl.maximum(first_q - EXTENT - BN + 2 - start, 0)
        near = start + tl.cdiv(behind, BN) * BN
        # this split's share of the key tiles
        per = tl.cdiv(tl.cdiv(last_q + 1 - start, BN), splits) * BN
        lo_key = start + split * per
        hi_key = tl.minimum(lo_key + per, last_q + 1)
        far_end = tl.minimum(near, hi_key)
        near_start = tl.maximum(near, lo_key)
        bt_row = table_ptr + seq.to(tl.int64) * stride_bt
        scratch = scratch_ptr + (tl.program_id(0) * tl.num_programs(1)
                                 + kv_head).to(tl.int64) * (BM * SPAN)
        bias_rows = scratch + offs_m * SPAN
        if near_start < hi_key:
            offs_r = tl.arange(0, R)
            offs_c = tl.arange(0, WCH)
            rel = tl.load(rel_ptr + q_row[:, None] * stride_r0
                          + head[:, None] * stride_r1 + offs_r[None, :],
                          mask=row_ok[:, None], other=0.0)
            # the near keys' distances, from the last key to the first
            d_lo = tl.maximum(first_q - hi_key + 1, 0) // WCH * WCH
            reach = tl.minimum(last_q - near_start + 1, EXTENT)
            for c0 in range(d_lo, reach, WCH):
                dists = c0 + offs_c
                part = tl.load(proj_ptr + offs_r[:, None] * EXTENT
                               + dists[None, :],
                               mask=(dists < EXTENT)[None, :], other=0.0)
                # rounded to bf16 as the reference's bias is
                values = tl.dot(rel, part).to(tl.bfloat16)
                cols = qpos[:, None] - dists[None, :] - near_start
                tl.store(bias_rows[:, None] + cols, values,
                         mask=(cols >= 0) & (cols < SPAN))
            tl.debug_barrier()
        for n0 in range(lo_key, far_end, BN):
            m_i, l_i, acc = _attend_tile(
                q, k_ptr, v_ptr, bt_row, n0, kv_head, qpos, row_ok, used,
                window, bias_rows, m_i, l_i, acc, scale_log2, stride_k0,
                stride_k1, stride_k2, stride_v0, stride_v1, stride_v2, BN, D,
                EXTENT, PAGE, HAS_WINDOW, False)
        for n0 in range(near_start, hi_key, BN):
            m_i, l_i, acc = _attend_tile(
                q, k_ptr, v_ptr, bt_row, n0, kv_head, qpos, row_ok, used,
                window, bias_rows + (n0 - near_start), m_i, l_i, acc,
                scale_log2, stride_k0,
                stride_k1, stride_k2, stride_v0, stride_v1, stride_v2, BN, D,
                EXTENT, PAGE, HAS_WINDOW, True)
        if SPLIT:
            slot = (q_row * HEADS + head) * splits + split
            tl.store(part_acc_ptr + slot[:, None] * D + offs_d[None, :], acc,
                     mask=row_ok[:, None])
            tl.store(part_ml_ptr + slot * 2, m_i, mask=row_ok)
            tl.store(part_ml_ptr + slot * 2 + 1, l_i, mask=row_ok)
        else:
            out = acc / l_i[:, None]
            tl.store(out_ptr + q_row[:, None] * stride_o0
                     + head[:, None] * stride_o1 + offs_d[None, :],
                     out.to(tl.bfloat16), mask=row_ok[:, None])

    @triton.jit(do_not_specialize=["window", "splits"])
    def decode_attention(q_ptr, k_ptr, v_ptr, rel_ptr, proj_t_ptr, sink_ptr,
                         out_ptr, part_acc_ptr, part_ml_ptr, token_seq_ptr,
                         token_q_ptr, num_real_ptr, cu_q_ptr, used_ptr,
                         table_ptr, stride_q0,
                         stride_q1, stride_r0, stride_r1, stride_s0,
                         stride_o0, stride_o1, stride_k0, stride_k1,
                         stride_k2, stride_v0, stride_v1, stride_v2,
                         stride_bt, scale_log2, window, splits,
                         GROUP: tl.constexpr, HEADS: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
                         R: tl.constexpr, EXTENT: tl.constexpr,
                         PAGE: tl.constexpr, HAS_WINDOW: tl.constexpr,
                         SPLIT: tl.constexpr, LIMIT: tl.constexpr):
        # One query token's GROUP heads against one KV head's keys. Every
        # row shares the token's position, so a key tile's bias is one dot:
        # the rows' relative states times the profile at the tile's
        # distances, read from the transposed (EXTENT, R) profile.
        LOG2E: tl.constexpr = 1.4426950408889634
        token = tl.program_id(0) // splits
        split = tl.program_id(0) % splits
        kv_head = tl.program_id(1)
        # a captured grid covers a bucket; tokens past the real ones exit
        if LIMIT:
            if token >= tl.load(num_real_ptr):
                return
        seq = tl.load(token_seq_ptr + token)
        qi = tl.load(token_q_ptr + token)
        q_start = tl.load(cu_q_ptr + seq)
        q_len = tl.load(cu_q_ptr + seq + 1) - q_start
        used = tl.load(used_ptr + seq)
        qpos = used - q_len + qi
        q_row = (q_start + qi).to(tl.int64)
        offs_m = tl.arange(0, BM)
        row_ok = offs_m < GROUP
        head = kv_head * GROUP + offs_m
        offs_d = tl.arange(0, D)
        offs_r = tl.arange(0, R)
        offs_key = tl.arange(0, BN)
        q = tl.load(q_ptr + q_row * stride_q0 + head[:, None] * stride_q1
                    + offs_d[None, :], mask=row_ok[:, None], other=0.0)
        rel = tl.load(rel_ptr + q_row * stride_r0 + head[:, None] * stride_r1
                      + offs_r[None, :], mask=row_ok[:, None], other=0.0)
        if split == 0:
            sink = tl.load(sink_ptr + q_row * stride_s0 + head, mask=row_ok,
                           other=0.0).to(tl.float32)
            m_i = sink * LOG2E
            l_i = tl.zeros((BM,), dtype=tl.float32) + 1.0
        else:
            m_i = tl.zeros((BM,), dtype=tl.float32) - float("inf")
            l_i = tl.zeros((BM,), dtype=tl.float32)
        acc = tl.zeros((BM, D), dtype=tl.float32)
        lo = qpos * 0
        if HAS_WINDOW:
            lo = tl.maximum(qpos - (window - 1), 0)
        start = lo // BN * BN
        per = tl.cdiv(tl.cdiv(qpos + 1 - start, BN), splits) * BN
        lo_key = start + split * per
        hi_key = tl.minimum(lo_key + per, qpos + 1)
        bt_row = table_ptr + seq.to(tl.int64) * stride_bt
        for n0 in range(lo_key, hi_key, BN):
            page = tl.load(bt_row + n0 // PAGE).to(tl.int64)
            in_page = n0 % PAGE
            kpos = n0 + offs_key
            k = tl.load(k_ptr + page * stride_k0
                        + (in_page + offs_key[:, None]) * stride_k1
                        + kv_head * stride_k2 + offs_d[None, :])
            v = tl.load(v_ptr + page * stride_v0
                        + (in_page + offs_key[:, None]) * stride_v1
                        + kv_head * stride_v2 + offs_d[None, :])
            s = tl.dot(q, tl.trans(k)) * scale_log2
            dist = qpos - kpos
            near = (dist >= 0) & (dist < EXTENT)
            profile = tl.load(proj_t_ptr + tl.minimum(tl.maximum(dist, 0),
                                                      EXTENT - 1)[:, None] * R
                              + offs_r[None, :], mask=near[:, None],
                              other=0.0)
            # rounded to bf16 as the reference's bias is
            bias = tl.dot(rel, tl.trans(profile)).to(tl.bfloat16)
            s += bias.to(tl.float32) * LOG2E
            valid = (kpos >= lo) & (dist >= 0)
            s = tl.where(valid[None, :], s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            alpha = tl.math.exp2(m_i - m_safe)
            p = tl.math.exp2(s - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            m_i = m_new
        if SPLIT:
            slot = (token * HEADS + head) * splits + split
            tl.store(part_acc_ptr + slot[:, None] * D + offs_d[None, :], acc,
                     mask=row_ok[:, None])
            tl.store(part_ml_ptr + slot * 2, m_i, mask=row_ok)
            tl.store(part_ml_ptr + slot * 2 + 1, l_i, mask=row_ok)
        else:
            out = acc / l_i[:, None]
            tl.store(out_ptr + q_row * stride_o0 + head[:, None] * stride_o1
                     + offs_d[None, :], out.to(tl.bfloat16),
                     mask=row_ok[:, None])

    @triton.jit(do_not_specialize=["splits"])
    def merge_tokens(part_acc_ptr, part_ml_ptr, rows_ptr, num_real_ptr,
                     out_ptr, stride_o0, stride_o1, splits,
                     HEADS: tl.constexpr, D: tl.constexpr,
                     MAX_SPLITS: tl.constexpr, LIMIT: tl.constexpr):
        # one decoded token and head: the splits' softmax states, merged
        # into the token's row
        token = tl.program_id(0).to(tl.int64)
        head = tl.program_id(1)
        if LIMIT:
            if token >= tl.load(num_real_ptr):
                return
        row = tl.load(rows_ptr + token).to(tl.int64)
        offs_s = tl.arange(0, MAX_SPLITS)
        offs_d = tl.arange(0, D)
        slot = (token * HEADS + head) * splits + offs_s
        ok = offs_s < splits
        m = tl.load(part_ml_ptr + slot * 2, mask=ok, other=-float("inf"))
        sums = tl.load(part_ml_ptr + slot * 2 + 1, mask=ok, other=0.0)
        top = tl.max(m, axis=0)
        w = tl.where(ok, tl.math.exp2(m - top), 0.0)
        acc = tl.load(part_acc_ptr + slot[:, None] * D + offs_d[None, :],
                      mask=ok[:, None], other=0.0)
        out = tl.sum(acc * w[:, None], axis=0) / tl.sum(sums * w, axis=0)
        tl.store(out_ptr + row * stride_o0 + head * stride_o1 + offs_d,
                 out.to(tl.bfloat16))

    @triton.jit(do_not_specialize=["splits"])
    def merge_splits(part_acc_ptr, part_ml_ptr, out_ptr, stride_o0,
                     stride_o1, splits, HEADS: tl.constexpr,
                     D: tl.constexpr, MAX_SPLITS: tl.constexpr):
        # one row and head: the splits' softmax states, merged
        row = tl.program_id(0).to(tl.int64)
        head = tl.program_id(1)
        offs_s = tl.arange(0, MAX_SPLITS)
        offs_d = tl.arange(0, D)
        slot = (row * HEADS + head) * splits + offs_s
        ok = offs_s < splits
        m = tl.load(part_ml_ptr + slot * 2, mask=ok, other=-float("inf"))
        sums = tl.load(part_ml_ptr + slot * 2 + 1, mask=ok, other=0.0)
        top = tl.max(m, axis=0)
        w = tl.where(ok, tl.math.exp2(m - top), 0.0)
        acc = tl.load(part_acc_ptr + slot[:, None] * D + offs_d[None, :],
                      mask=ok[:, None], other=0.0)
        out = tl.sum(acc * w[:, None], axis=0) / tl.sum(sums * w, axis=0)
        tl.store(out_ptr + row * stride_o0 + head * stride_o1 + offs_d,
                 out.to(tl.bfloat16))

    @triton.jit
    def _e2m1(x):
        a = tl.abs(x)
        code = tl.where(a <= 0.25, 0, tl.where(a < 0.75, 1, tl.where(
            a <= 1.25, 2, tl.where(a < 1.75, 3, tl.where(
                a <= 2.5, 4, tl.where(a < 3.5, 5, tl.where(
                    a <= 5.0, 6, 7))))))).to(tl.uint32)
        exponent = code >> 1
        mantissa = (code & 1).to(tl.float32)
        value = tl.where(exponent == 0, mantissa * 0.5, tl.where(
            exponent == 1, 1.0 + mantissa * 0.5, tl.where(
                exponent == 2, 2.0 + mantissa, 4.0 + 2.0 * mantissa)))
        sign = x.to(tl.uint32, bitcast=True) >> 31
        return code | (sign << 3), tl.where(sign != 0, -value, value)

    @triton.jit
    def _e2m1_pairs(x, HW: tl.constexpr):
        # (blocks, 8, 2) values -> (blocks, 8) bytes, low nibble first,
        # and the values rounded to e2m1, (blocks, 16)
        low, high = tl.split(x)
        if HW:
            packed, halves = tl.inline_asm_elementwise(
                "{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, $3, $2; "
                "cvt.u32.u8 $0, b; cvt.rn.f16x2.e2m1x2 $1, b; }",
                "=r,=r,f,f", [low, high], dtype=(tl.uint32, tl.uint32),
                is_pure=True, pack=1)
            low_v = (halves & 0xFFFF).to(tl.uint16).to(tl.float16,
                                                       bitcast=True)
            high_v = (halves >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
            rounded = tl.join(low_v, high_v).to(tl.float32)
            packed = packed.to(tl.uint8)
        else:
            low_code, low_v = _e2m1(low)
            high_code, high_v = _e2m1(high)
            packed = (low_code | (high_code << 4)).to(tl.uint8)
            rounded = tl.join(low_v, high_v)
        return packed, rounded

    @triton.jit
    def _nvfp4_chunk(x, amax, CH: tl.constexpr, HW: tl.constexpr):
        # four-over-six: per 16-value block, the 6.0- or 4.0-based fp8
        # scale with the smaller absolute error; returns packed bytes
        # (CH // 2,) and fp8 scales (CH // 16,). The row scale amax / 1536
        # leaves each block's fp8 scale headroom below 448.
        NBLK: tl.constexpr = CH // 16
        encoder = tl.minimum(1536.0 / amax, 3.4028234663852886e38)
        decoder = 1.0 / encoder
        outer = amax * (1.0 / 1536.0)
        blocks = tl.reshape(x, (NBLK, 16))
        block_rowsax = tl.max(tl.abs(blocks), axis=1)
        scale6 = tl.minimum(block_rowsax * (encoder * (1.0 / 6.0)), 448.0)
        scale4 = tl.minimum(block_rowsax * (encoder * 0.25), 448.0)
        scale6 = scale6.to(tl.float8e4nv).to(tl.float32)
        scale4 = scale4.to(tl.float8e4nv).to(tl.float32)
        inv6 = tl.minimum(1.0 / (scale6 * decoder), 3.4028234663852886e38)
        inv4 = tl.minimum(1.0 / (scale4 * decoder), 3.4028234663852886e38)
        code6, value6 = _e2m1_pairs(
            tl.reshape(blocks * inv6[:, None], (NBLK, 8, 2)), HW)
        code4, value4 = _e2m1_pairs(
            tl.reshape(blocks * inv4[:, None], (NBLK, 8, 2)), HW)
        value6 = tl.reshape(value6, (NBLK, 16))
        value4 = tl.reshape(value4, (NBLK, 16))
        err6 = tl.sum(tl.abs(value6 * scale6[:, None] * outer - blocks), 1)
        err4 = tl.sum(tl.abs(value4 * scale4[:, None] * outer - blocks), 1)
        choose4 = err4 < err6
        packed = tl.where(choose4[:, None], code4, code6)
        scales = tl.where(choose4, scale4, scale6)
        return tl.reshape(packed, (CH // 2,)), scales

    @triton.jit
    def _scale_offset(row, block, KB4):
        # CUTLASS's swizzled block-scale layout: 128 x 4 tiles
        return (((row // 128) * KB4 + block // 4) * 512 + (row % 32) * 16
                + ((row % 128) // 32) * 4 + block % 4)

    @triton.jit
    def _mxfp8_round(x, NBLK: tl.constexpr):
        groups = tl.reshape(x, (NBLK, 32))
        amax = tl.max(tl.abs(groups), 1)
        raw = amax * (1.0 / 448.0)
        # fp32 constants: a bare 2**-127 would be typed fp64
        smallest = tl.full((), 5.877471754111438e-39, tl.float32)
        largest = tl.full((), 1.7014118346046923e38, tl.float32)
        exponent = ((raw.to(tl.int32, bitcast=True) + 0x007FFFFF)
                    & 0x7F800000) >> 23
        exponent = tl.where(raw <= smallest, 0, tl.minimum(exponent, 254))
        scale = tl.maximum((exponent << 23).to(tl.float32, bitcast=True),
                           smallest)
        inverse = tl.where(exponent == 0, largest, 1.0 / scale)
        values = tl.minimum(tl.maximum(groups * inverse[:, None], -448.0),
                            448.0)
        values = values.to(tl.float8e4nv).to(tl.float32) * scale[:, None]
        return tl.reshape(values, (NBLK * 32,))

    @triton.jit
    def quantize_tokens(x_ptr, stride_x, c_map_ptr, ids_ptr, offsets_ptr,
                        sf_offsets_ptr, q_ptr, sf_ptr, row_scale_ptr,
                        bf16_ptr, K: tl.constexpr, CH: tl.constexpr,
                        TOPK: tl.constexpr, NVFP4: tl.constexpr,
                        HW: tl.constexpr, LINEAR: tl.constexpr):
        # one token, quantized once and stored to each of its TOPK
        # permuted rows; NVFP4 writes packed values, swizzled (or, with
        # LINEAR, row-major) scales, and the row scale, else the
        # MXFP8-rounded row in bf16
        token = tl.program_id(0).to(tl.int64)
        slots = token * TOPK + tl.arange(0, TOPK)
        rows = tl.load(c_map_ptr + slots).to(tl.int64)
        offs = tl.arange(0, CH)
        amax = 0.0
        if NVFP4:
            for c in tl.static_range(K // CH):
                x = tl.load(x_ptr + token * stride_x + c * CH + offs)
                amax = tl.maximum(amax, tl.max(tl.abs(x.to(tl.float32)), 0))
            experts = tl.load(ids_ptr + slots)
            sf_rows = (tl.load(sf_offsets_ptr + experts)
                       + rows - tl.load(offsets_ptr + experts))
        for c in tl.static_range(K // CH):
            x = tl.load(x_ptr + token * stride_x + c * CH + offs).to(
                tl.float32)
            if NVFP4:
                packed, scales = _nvfp4_chunk(x, amax, CH, HW)
                byte = c * (CH // 2) + tl.arange(0, CH // 2)
                tl.store(q_ptr + rows[:, None] * (K // 2) + byte[None, :],
                         tl.broadcast_to(packed[None, :], (TOPK, CH // 2)))
                block = c * (CH // 16) + tl.arange(0, CH // 16)
                if LINEAR:
                    offset = sf_rows[:, None] * (K // 16) + block[None, :]
                else:
                    offset = _scale_offset(sf_rows[:, None], block[None, :],
                                           (K // 16 + 3) // 4)
                tl.store(sf_ptr + offset,
                         tl.broadcast_to(scales.to(tl.float8e4nv)[None, :],
                                         (TOPK, CH // 16)))
            else:
                y = _mxfp8_round(x, CH // 32).to(tl.bfloat16)
                cols = c * CH + offs
                tl.store(bf16_ptr + rows[:, None] * K + cols[None, :],
                         tl.broadcast_to(y[None, :], (TOPK, CH)))
        if NVFP4:
            tl.store(row_scale_ptr + rows,
                     tl.zeros((TOPK,), tl.float32) + amax * (1.0 / 1536.0))

    @triton.jit
    def _tanh_approx(x):
        return tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;", "=f,f", [x], dtype=tl.float32,
            is_pure=True, pack=1)

    @triton.jit
    def _sigmoid_approx(x):
        return tl.inline_asm_elementwise(
            "{ .reg .f32 t; mul.rn.f32 t, $1, 0fBFB8AA3B; "
            "ex2.approx.ftz.f32 t, t; add.rn.f32 t, t, 0f3F800000; "
            "rcp.approx.ftz.f32 $0, t; }",
            "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _situ(gate_up_ptr, p, start, outer, gamma, gate_cap, up_cap,
              N: tl.constexpr, W: tl.constexpr, NVFP4: tl.constexpr):
        # the softcapped SwiGLU of columns [start, start + W), from 2N
        # columns that alternate 64 up and 64 gate columns; the result is
        # rounded to bf16, so approximate tanh and sigmoid suffice
        index = start + tl.arange(0, W)
        column = gate_up_ptr + p * 2 * N + (index // 64) * 128 + index % 64
        up = tl.load(column).to(tl.float32)
        gate = tl.load(column + 64).to(tl.float32)
        if NVFP4:
            gate = (gate * outer).to(tl.bfloat16).to(tl.float32)
            up = (up * outer).to(tl.bfloat16).to(tl.float32)
        y = (gate_cap * _tanh_approx(gate / gate_cap) * _sigmoid_approx(gate)
             * up_cap * _tanh_approx(up / up_cap))
        return (y * gamma).to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _store_quantized(y, amax, p, start, q_ptr, sf_ptr, sf_row, bf16_ptr,
                         N: tl.constexpr, W: tl.constexpr,
                         NVFP4: tl.constexpr, HW: tl.constexpr):
        if NVFP4:
            packed, scales = _nvfp4_chunk(y, amax, W, HW)
            tl.store(q_ptr + p * (N // 2) + start // 2
                     + tl.arange(0, W // 2), packed)
            block = start // 16 + tl.arange(0, W // 16)
            tl.store(sf_ptr + _scale_offset(sf_row, block, (N // 16 + 3) // 4),
                     scales.to(tl.float8e4nv))
        else:
            y = _mxfp8_round(y, W // 32)
            tl.store(bf16_ptr + p * N + start + tl.arange(0, W),
                     y.to(tl.bfloat16))

    @triton.jit
    def swiglu_quantize(gate_up_ptr, row_scale_in_ptr, gamma_ptr, expert_ptr,
                        offsets_ptr, sf_offsets_ptr, q_ptr, sf_ptr,
                        row_scale_ptr, bf16_ptr, gate_cap, up_cap,
                        N: tl.constexpr, HEAD: tl.constexpr,
                        TAIL: tl.constexpr, NVFP4: tl.constexpr,
                        HW: tl.constexpr):
        # one pass over a power-of-two head and a power-of-two tail
        p = tl.program_id(0).to(tl.int64)
        outer = 1.0
        sf_row = p
        if NVFP4:
            outer = tl.load(row_scale_in_ptr + p)
            expert = tl.load(expert_ptr + p)
            sf_row = tl.load(sf_offsets_ptr + expert) + p - tl.load(
                offsets_ptr + expert)
        gamma = tl.load(gamma_ptr + p)
        head = _situ(gate_up_ptr, p, 0, outer, gamma, gate_cap, up_cap, N,
                     HEAD, NVFP4)
        amax = tl.max(tl.abs(head), 0)
        tail = head
        if TAIL > 0:
            tail = _situ(gate_up_ptr, p, HEAD, outer, gamma, gate_cap,
                         up_cap, N, TAIL, NVFP4)
            amax = tl.maximum(amax, tl.max(tl.abs(tail), 0))
        if NVFP4:
            tl.store(row_scale_ptr + p, amax * (1.0 / 1536.0))
        _store_quantized(head, amax, p, 0, q_ptr, sf_ptr, sf_row, bf16_ptr,
                         N, HEAD, NVFP4, HW)
        if TAIL > 0:
            _store_quantized(tail, amax, p, HEAD, q_ptr, sf_ptr, sf_row,
                             bf16_ptr, N, TAIL, NVFP4, HW)

    @triton.jit
    def combine(down_ptr, row_scale_ptr, c_map_ptr, out_ptr, K: tl.constexpr,
                KP: tl.constexpr, TOPK: tl.constexpr, SCALED: tl.constexpr):
        token = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, KP)
        mask = offs < K
        acc = tl.zeros((KP,), dtype=tl.float32)
        for j in tl.static_range(TOPK):
            p = tl.load(c_map_ptr + token * TOPK + j).to(tl.int64)
            d = tl.load(down_ptr + p * K + offs, mask=mask,
                        other=0.0).to(tl.float32)
            if SCALED:
                d = d * tl.load(row_scale_ptr + p)
            acc += d
        tl.store(out_ptr + token * K + offs, acc.to(tl.bfloat16), mask=mask)

    @triton.jit
    def gather_state(x_ptr, stride_x, srcs_ptr, slots_ptr, state_ptr,
                     stride_slot, stride_tap, out_ptr, C,
                     BLOCK: tl.constexpr):
        # out[w, j] = the value of tap source srcs[w, j]; a padding write,
        # with a negative slot, reads nothing
        w = tl.program_id(0).to(tl.int64)
        j = tl.program_id(1)
        cols = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
        mask = (cols < C) & (tl.load(slots_ptr + w) >= 0)
        src = tl.load(srcs_ptr + w * 3 + j)
        value = _tap(x_ptr, stride_x, state_ptr, stride_slot, stride_tap, src,
                     cols, mask)
        tl.store(out_ptr + (w * 3 + j) * C + cols, value.to(tl.bfloat16),
                 mask=mask)

    @triton.jit
    def scatter_state(values_ptr, slots_ptr, state_ptr, stride_slot,
                      stride_tap, C, BLOCK: tl.constexpr):
        # state[slots[w], j] = values[w, j]; a negative slot is padding
        w = tl.program_id(0).to(tl.int64)
        j = tl.program_id(1)
        cols = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
        slot = tl.load(slots_ptr + w)
        mask = (cols < C) & (slot >= 0)
        value = tl.load(values_ptr + (w * 3 + j) * C + cols, mask=mask)
        tl.store(state_ptr + slot * stride_slot + j * stride_tap + cols, value,
                 mask=mask)

    @triton.jit
    def bias_tiles(rel_ptr, stride_r0, stride_r1, proj_ptr, out_ptr,
                   tile_seq_ptr, first_tile_ptr, cu_q_ptr, used_ptr,
                   GROUP: tl.constexpr, KV_HEADS: tl.constexpr,
                   R: tl.constexpr, EXTENT: tl.constexpr, BM: tl.constexpr,
                   BN: tl.constexpr, SPAN: tl.constexpr, WCH: tl.constexpr):
        # one FA4 query tile of one KV head: its rows' bias over the key
        # tiles from its first near tile, column c at key
        # near_start + c, the layout the patched FA4 kernel reads
        tile = tl.program_id(0)
        kv_head = tl.program_id(1)
        seq = tl.load(tile_seq_ptr + tile)
        first_row = (tile - tl.load(first_tile_ptr + seq)) * BM
        q_start = tl.load(cu_q_ptr + seq)
        q_len = tl.load(cu_q_ptr + seq + 1) - q_start
        shift = tl.load(used_ptr + seq) - q_len
        offs_m = tl.arange(0, BM)
        row = first_row + offs_m
        token = row // GROUP
        ok = token < q_len
        head = kv_head * GROUP + row % GROUP
        qpos = token + shift
        qmin = first_row // GROUP + shift
        qmax = tl.minimum((first_row + BM - 1) // GROUP, q_len - 1) + shift
        near_start = tl.maximum(qmin - (EXTENT - 1), 0) // BN * BN
        offs_r = tl.arange(0, R)
        rel = tl.load(rel_ptr + (q_start + token).to(tl.int64)[:, None]
                      * stride_r0 + head[:, None] * stride_r1
                      + offs_r[None, :], mask=ok[:, None], other=0.0)
        out_row = out_ptr + ((tile * KV_HEADS + kv_head).to(tl.int64) * BM
                             + offs_m) * SPAN
        offs_c = tl.arange(0, WCH)
        for c0 in range(0, qmax - near_start + 1, WCH):
            d = c0 + offs_c
            part = tl.load(proj_ptr + offs_r[:, None] * EXTENT + d[None, :],
                           mask=(d < EXTENT)[None, :], other=0.0)
            values = tl.dot(rel, part).to(tl.bfloat16)
            cols = qpos[:, None] - d[None, :] - near_start
            tl.store(out_row[:, None] + cols, values,
                     mask=(cols >= 0) & (cols < SPAN) & ok[:, None])

    # lse's stride is the chunk's row count
    @triton.jit(do_not_specialize=["stride_lse"])
    def sink_gate(o_ptr, lse_ptr, stride_lse, sink_ptr, stride_s0, out_ptr,
                  stride_o0, stride_o1, HEADS: tl.constexpr,
                  HP: tl.constexpr, D: tl.constexpr):
        # the sink's share of each head's softmax mass: o * sigmoid(lse - sink)
        row = tl.program_id(0).to(tl.int64)
        heads = tl.arange(0, HP)
        ok = heads < HEADS
        lse = tl.load(lse_ptr + heads * stride_lse + row, mask=ok, other=0.0)
        sink = tl.load(sink_ptr + row * stride_s0 + heads, mask=ok,
                       other=0.0).to(tl.float32)
        keep = tl.sigmoid(lse - sink)
        offs_d = tl.arange(0, D)
        o = tl.load(o_ptr + (row * HEADS + heads)[:, None] * D
                    + offs_d[None, :], mask=ok[:, None], other=0.0)
        tl.store(out_ptr + row * stride_o0 + heads[:, None] * stride_o1
                 + offs_d[None, :],
                 (o.to(tl.float32) * keep[:, None]).to(tl.bfloat16),
                 mask=ok[:, None])

    _KERNELS.update(
        bias_tiles=bias_tiles, sink_gate=sink_gate,
        scatter_state=scatter_state,
        mix=mix, gate_out=gate_out,
        rms_norm=rms_norm, short_conv=short_conv, kv_conv=kv_conv,
        clip_heads=clip_heads, gate_attention=gate_attention, route=route,
        rel_attention=rel_attention, merge_splits=merge_splits,
        decode_attention=decode_attention, merge_tokens=merge_tokens,
        quantize_tokens=quantize_tokens,
        swiglu_quantize=swiglu_quantize, combine=combine,
        gather_state=gather_state)
    return _KERNELS



# ---- launchers --------------------------------------------------------

def _hp(h):
    return _pow2(h)


def _warps(width):
    return 16 if width >= 4096 else 8 if width >= 1024 else 4


def mix(torch, blocks, num_blocks, partial, query, *, addend=None,
        weight=None, eps=1e-6, squares=False):
    """A block attention residual over each row's states.

    Args:
        torch: The torch module.
        blocks: (rows, max blocks, hidden) bf16; the first num_blocks hold
            completed blocks.
        num_blocks: Completed blocks the mix reads.
        partial: (rows, hidden) partial block, or None.
        query: (queries, hidden) fp32 learned queries, pre-scaled by
            hidden ** -0.5; one output per query.
        addend: Optional (rows, hidden) added to the partial in fp32.
        weight: (queries, hidden) RMS norm weights; None leaves the mix
            unnormed.
        eps: The RMS norms' epsilon.
        squares: With no weight, return the unnormed mix in bf16 and each
            output row's sum of squares, rather than the mix in fp32.

    Returns:
        (queries, rows, hidden): bf16 when normed, else fp32; or, with
        squares, (bf16 mix, (queries, rows) fp32 sums of squares).
    """
    rows, hidden = blocks.shape[0], blocks.shape[2]
    nq = query.shape[0]
    dtype = (torch.bfloat16 if weight is not None or squares
             else torch.float32)
    out = torch.empty((nq, rows, hidden), dtype=dtype, device=blocks.device)
    sums = (torch.empty((nq, rows), dtype=torch.float32, device=blocks.device)
            if squares else out)
    if rows == 0:
        return (out, sums) if squares else out
    has_partial = partial is not None
    partial = partial if has_partial else blocks[:, 0]
    addend_t = addend if addend is not None else partial
    _kernels()["mix"][(rows,)](
        blocks, blocks.stride(1), blocks.stride(0), partial, partial.stride(0),
        addend_t, addend_t.stride(0), query,
        weight if weight is not None else query, out, out.stride(0),
        out.stride(1), sums, sums.stride(0), eps, H=hidden,
        CH=_chunk(hidden, 256), NB=num_blocks,
        NS=num_blocks + int(has_partial), NQ=nq,
        HAS_ADDEND=addend is not None, NORM=weight is not None,
        SQUARES=squares, num_warps=1)
    return (out, sums) if squares else out


def _gate_hidden(hidden):
    """The gate and row strides of a shared or per-gate hidden tensor."""
    if hidden.dim() == 2:
        return 0, hidden.stride(0)
    return hidden.stride(0), hidden.stride(1)


def gate_out(torch, hidden, squares, prev_block, prev_out, ph, pa, pb,
             scores, stats, gains_t, weight, eps=1e-6):
    """Each loop message gate's gated, RMS-normed input in bf16.

    Args:
        torch: The torch module.
        hidden: (rows, width) bf16 shared by every gate, or (gates, rows,
            width).
        squares: (1 or gates, rows) fp32 sums of squares of hidden's rows.
        prev_block: (rows, width) the previous pass's last block, a.
        prev_out: (rows, width) the previous pass's output, b.
        ph: hidden @ each gate's hidden-side down weights, fp32, as
            (rows, gates * rank) for a shared hidden, else
            (gates, rows, rank).
        pa: (rows, >= gates * rank) fp32 view of a @ the gates' message-side
            down weights, gate by gate.
        pb: The same view of b's.
        scores: (rows, >= gates * 2) fp32 view of each gate's message
            query scores (query @ a, query @ b).
        stats: (rows, 3) fp32 (a @ a, b @ b, a @ b).
        gains_t: (gates, rank, 2 * width) bf16 forget and message gains.
        weight: (gates, width) RMS norm weights.
        eps: The norms' epsilon.

    Returns:
        (gates, rows, width) bf16.
    """
    count, width = weight.shape
    rank = gains_t.shape[1]
    rows = prev_block.shape[0]
    out = torch.empty((count, rows, width), dtype=torch.bfloat16,
                      device=prev_block.device)
    if rows == 0:
        return out
    stride_hg, stride_h = _gate_hidden(hidden)
    stride_sg = squares.stride(0) if squares.shape[0] > 1 else 0
    if ph.dim() == 2:
        # a shared hidden's products sit side by side
        stride_pg, stride_p = rank, ph.stride(0)
    else:
        stride_pg, stride_p = ph.stride(0), ph.stride(1)
    block_rows = 64
    _kernels()["gate_out"][(triton_cdiv(rows, block_rows), count)](
        hidden, stride_hg, stride_h, squares, stride_sg, prev_block, prev_out,
        prev_block.stride(0), ph, stride_pg, stride_p, pa, pb, pa.stride(0),
        scores, scores.stride(0), stats, gains_t, weight, out, out.stride(0),
        out.stride(1), rows, eps, H=width, R=rank, BM=block_rows, BN=64,
        num_warps=8)
    return out


def rms_norm(torch, x, weight, eps, out=None):
    rows, width = x.shape
    if out is None:
        out = torch.empty((rows, width), dtype=torch.bfloat16, device=x.device)
    if rows:
        _kernels()["rms_norm"][(rows,)](
            x, x.stride(0), weight, out, out.stride(0), eps, H=width,
            HP=_hp(width), num_warps=_warps(_hp(width)))
    return out


def short_conv(torch, x, taps, state, weight, *, residual=None, addend=None):
    """A static four-tap convolution with residual, scaled by 1/sqrt(5).

    Args:
        torch: The torch module.
        x: (rows, channels) inputs.
        taps: (rows, 3) int32 tap sources (conv_taps).
        state: (slots, 3, channels) saved columns of this convolution.
        weight: (channels, 4) fp32 taps, the last on the current row.
        residual: Optional (rows, channels) residual stream added in fp32
            to the bf16 convolution output.
        addend: Optional term added to the residual first.
    """
    rows, channels = x.shape
    out = torch.empty_like(x)
    if rows == 0:
        return out
    block_t, block_c = 8, 256
    base_t = residual if residual is not None else x
    base2_t = addend if addend is not None else x
    _kernels()["short_conv"][(triton_cdiv(rows, block_t),
                              triton_cdiv(channels, block_c))](
        x, x.stride(0), taps, state, state.stride(0), state.stride(1), weight,
        base_t, base_t.stride(0), base2_t, base2_t.stride(0), out,
        out.stride(0), rows, channels, 1.0 / math.sqrt(5.0), BT=block_t,
        BC=block_c, HAS_BASE=residual is not None,
        HAS_BASE2=addend is not None,
        num_warps=4)
    return out


def kv_conv(torch, x, taps, state, weight, *, heads, head_dim, dynamic=None,
            is_key, clip, eps=1e-6):
    """The key or value convolution, then its clip or norm, (rows, heads, dim).

    Args:
        torch: The torch module.
        x: (rows, heads * dim) view of the projection's columns.
        taps: (rows, 3) int32 tap sources (conv_taps).
        state: (slots, 3, channels) saved columns of this convolution.
        weight: (4, channels) by distance with dynamic taps, else
            (channels, 4) in convolution order.
        heads: KV heads.
        head_dim: Channels per head.
        dynamic: (rows, heads * 4) view of the per-token taps, or None
            for static taps.
        is_key: Whether the output is a key, L2-clipped, rather than a
            value, RMS-normed.
        clip: The key's L2 clip threshold.
        eps: The value norm's epsilon.
    """
    rows = x.shape[0]
    out = torch.empty((rows, heads, head_dim), dtype=torch.bfloat16,
                      device=x.device)
    if rows == 0:
        return out
    scale = 1.0 / math.sqrt(9.0 if dynamic is not None else 5.0)
    dyn = dynamic if dynamic is not None else x
    rows_per = 4
    _kernels()["kv_conv"][(triton_cdiv(rows, rows_per),)](
        x, x.stride(0), dyn, dyn.stride(0), taps, state, state.stride(0),
        state.stride(1), weight, out, out.stride(0), rows, eps, clip, scale,
        HEADS=heads, D=head_dim, BT=rows_per, DYNAMIC=dynamic is not None,
        IS_KEY=is_key, num_warps=8)
    return out


def route(torch, logits, bias, top_k, scale):
    """Top-k experts by sigmoid score plus bias, with normalized scores.

    Args:
        torch: The torch module.
        logits: (rows, experts) fp32 router logits.
        bias: (experts,) fp32 selection bias.
        top_k: Experts per row.
        scale: What each row's weights sum to.

    Returns:
        (ids (rows, top_k) int32, weights (rows, top_k) fp32).
    """
    rows, experts = logits.shape
    ids = torch.empty((rows, top_k), dtype=torch.int32, device=logits.device)
    weights = torch.empty((rows, top_k), dtype=torch.float32,
                          device=logits.device)
    if rows:
        rows_per = 8
        _kernels()["route"][(triton_cdiv(rows, rows_per),)](
            logits, bias, ids, weights, rows, float(scale), E=experts, K=top_k,
            BT=rows_per, num_warps=4)
    return ids, weights


def clip_heads(torch, x, scale, positions, *, heads, head_dim, clip,
               log_scale=None):
    """Scaled L2 clip per head, with log scaling (alpha, floor) if given."""
    rows = x.shape[0]
    out = torch.empty((rows, heads, head_dim), dtype=torch.bfloat16,
                      device=x.device)
    if rows == 0:
        return out
    alpha, floor = log_scale if log_scale is not None else (0.0, 1.0)
    width = _pow2(heads) * head_dim
    _kernels()["clip_heads"][(rows,)](
        x, x.stride(0), scale, positions, out, out.stride(0), out.stride(1),
        clip, alpha, floor, H=heads, HP=_pow2(heads), D=head_dim,
        LOG_SCALE=log_scale is not None, num_warps=_warps(width))
    return out


def gate_attention(torch, attn, gate):
    rows, width = attn.shape
    out = torch.empty_like(attn)
    if rows:
        block = 1024
        _kernels()["gate_attention"][(rows, triton_cdiv(width, block))](
            attn, gate, gate.stride(0), out, width, BLOCK=block, num_warps=4)
    return out



def attention_tiles(cu_q_host, rows_per_tile, keep=None):
    """(sequence, first query) per query tile, over sequences' query counts.

    keep, when given, marks the sequences to tile; the rest get no tiles.
    """
    import numpy as np

    lengths = np.diff(np.asarray(cu_q_host, dtype=np.int64))
    if keep is not None:
        lengths = np.where(keep, lengths, 0)
    counts = -(-lengths // rows_per_tile)
    seqs = np.repeat(np.arange(len(lengths), dtype=np.int32), counts)
    firsts = (np.arange(int(counts.sum()), dtype=np.int64)
              - np.repeat(np.cumsum(counts) - counts, counts)) * rows_per_tile
    return seqs, firsts.astype(np.int32)


def rel_attention(torch, q3, kp, vp, rel, sink, proj, out, tiles, cu_q, used,
                  table, *, window, max_keys, block_rows=64, block_keys=64,
                  num_warps=8, num_stages=3):
    """Causal paged attention with the relative bias and sink logits.

    Args:
        torch: The torch module.
        q3: (rows, heads, dim) queries.
        kp: (pages, page tokens, KV heads, dim) arena key pool.
        vp: The arena value pool, shaped as kp.
        rel: (rows, heads, rank) relative states.
        sink: (rows, heads) view of the sink logits.
        proj: (rank, extent) bf16 bias profiles.
        out: (rows, heads, dim) output.
        tiles: (sequence ids, first queries) on the GPU, from
            attention_tiles with block_rows // group rows per tile.
        cu_q: Each sequence's query bounds over the rows.
        used: Keys per sequence, its queries' included.
        table: Each sequence's arena pages.
        window: Keys each row sees, or 0 for every earlier key.
        max_keys: The most keys any sequence holds.
        block_rows: Rows per query tile, three heads of each token.
        block_keys: Keys per key tile.
        num_warps: Warps per program.
        num_stages: Software pipeline stages of the key loop.
    """
    rows, heads, dim = q3.shape
    kv_heads, page = kp.shape[2], kp.shape[1]
    group = heads // kv_heads
    rank, extent = proj.shape
    if rows == 0 or tiles[0].numel() == 0:
        return out
    num_tiles = tiles[0].numel()
    splits = num_splits(num_tiles * kv_heads, max_keys, window)
    # a program's near keys run from its first near tile to its last
    # query, plus a tile past it; rows start on 16 bytes
    span = -(-(extent + block_rows // group + 2 * block_keys) // 8) * 8
    scratch = torch.empty(num_tiles * splits * kv_heads * block_rows * span,
                          dtype=torch.bfloat16, device=q3.device)
    if splits > 1:
        part_acc = torch.empty((rows * heads * splits, dim),
                               dtype=torch.float32, device=q3.device)
        part_ml = torch.empty((rows * heads * splits, 2), dtype=torch.float32,
                              device=q3.device)
    else:
        part_acc = part_ml = out
    _kernels()["rel_attention"][(num_tiles * splits, kv_heads)](
        q3, kp, vp, rel, proj, sink, out, scratch, part_acc, part_ml,
        tiles[0], tiles[1], cu_q, used, table, q3.stride(0), q3.stride(1),
        rel.stride(0), rel.stride(1), sink.stride(0), out.stride(0),
        out.stride(1), kp.stride(0), kp.stride(1), kp.stride(2), vp.stride(0),
        vp.stride(1), vp.stride(2),
        table.stride(0), dim ** -0.5 * 1.4426950408889634, int(window),
        splits, GROUP=group, HEADS=heads, BM=block_rows, BN=block_keys, D=dim,
        R=rank, EXTENT=extent, PAGE=page, SPAN=span, WCH=32,
        HAS_WINDOW=window > 0, SPLIT=splits > 1,
        num_warps=num_warps, num_stages=num_stages)
    if splits > 1:
        _kernels()["merge_splits"][(rows, heads)](
            part_acc, part_ml, out, out.stride(0), out.stride(1), splits,
            HEADS=heads, D=dim, MAX_SPLITS=MAX_SPLITS, num_warps=4)
    return out


# FA4's query and key tile, and the arena page
FA4_TILE = 128
# key tiles the patched FA4 kernel reads bias for: the bias's 1,024
# distances, a query tile's 43 tokens, and alignment
FA4_BIAS_TILES = 10


def prefill_tiles(cu_q_host, group, keep):
    """FA4 query tiles: each tile's sequence and each sequence's first tile.

    A tile holds FA4_TILE packed rows, three heads of each token; keep
    marks the sequences FA4 runs, and the rest get no tiles.
    """
    import numpy as np

    lengths = np.where(keep, np.diff(np.asarray(cu_q_host, dtype=np.int64)), 0)
    counts = -(-lengths * group // FA4_TILE)
    first = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int32)
    seqs = np.repeat(np.arange(len(lengths), dtype=np.int32), counts)
    return seqs, first


def _bias_marker(kv_heads):
    """The score_mod that turns on the patched FA4 kernel's bias tiles."""
    marker = _MARKERS.get(kv_heads)
    if marker is None:
        import cutlass.cute as cute

        @cute.jit
        def marker(scores, _batch, _head, _q_index, _kv_index, _seqlen,
                   _aux):
            return scores

        marker.inkling_relative_bias = True
        marker.inkling_kv_heads = kv_heads
        _MARKERS[kv_heads] = marker
    return marker


def prefill_attention(torch, q3, kp, vp, rel, sink, proj, out, tiles, cu_q,
                      used_q, used, table, *, window, max_queries):
    """Causal paged attention of query tiles on FA4, with the relative bias.

    Writes every row of out; rows of sequences FA4 skips hold garbage
    until decode_attention writes them.

    Args:
        torch: The torch module.
        q3: (rows, heads, dim) queries.
        kp: (pages, FA4_TILE, KV heads, dim) arena key pool.
        vp: The arena value pool, shaped as kp.
        rel: (rows, heads, rank) relative states.
        sink: (rows, heads) view of the sink logits.
        proj: (rank, extent) bf16 bias profiles.
        out: (rows, heads, dim) output.
        tiles: (sequence per tile, first tile per sequence) on the GPU,
            from prefill_tiles.
        cu_q: Each sequence's query bounds over the rows.
        used_q: Queries per sequence FA4 runs, 0 for the rest.
        used: Keys per sequence, its queries' included.
        table: Each sequence's arena pages.
        window: Keys each row sees, or 0 for every earlier key.
        max_queries: The most queries any sequence FA4 runs holds.
    """
    from flash_attn.cute.interface import flash_attn_varlen_func

    rows, heads, dim = q3.shape
    kv_heads = kp.shape[2]
    group = heads // kv_heads
    rank, extent = proj.shape
    num_tiles = tiles[0].numel()
    if num_tiles == 0:
        return out
    span = FA4_BIAS_TILES * FA4_TILE
    bias = torch.empty(num_tiles * kv_heads * FA4_TILE, span,
                       dtype=torch.bfloat16, device=q3.device)
    _kernels()["bias_tiles"][(num_tiles, kv_heads)](
        rel, rel.stride(0), rel.stride(1), proj, bias, tiles[0], tiles[1],
        cu_q, used, GROUP=group, KV_HEADS=kv_heads, R=rank, EXTENT=extent,
        BM=FA4_TILE, BN=FA4_TILE, SPAN=span, WCH=32, num_warps=4)
    # FA4 builds the bias's TMA descriptor from these
    bias.__assumed_align__ = 16
    bias.__leading_dim__ = 1
    attended, lse = flash_attn_varlen_func(
        q3, kp, vp, cu_seqlens_q=cu_q, max_seqlen_q=max_queries,
        seqused_q=used_q, seqused_k=used, page_table=table,
        softmax_scale=dim ** -0.5, causal=True,
        window_size=(window - 1 if window else None, 0),
        score_mod=_bias_marker(kv_heads), aux_tensors=[bias, tiles[1]],
        return_lse=True, pack_gqa=True)
    _kernels()["sink_gate"][(rows,)](
        attended, lse, lse.stride(0), sink, sink.stride(0), out,
        out.stride(0), out.stride(1), HEADS=heads, HP=_pow2(heads), D=dim,
        num_warps=4)
    return out


def decode_attention(torch, q3, kp, vp, rel, sink, proj_t, out, tokens,
                     cu_q, used, table, *, window, max_keys, block_keys=64,
                     num_warps=4, num_stages=3, num_real=None, splits=None):
    """Attention of answer-step tokens, one token's heads per program.

    Args:
        torch: The torch module.
        q3: (rows, heads, dim) queries.
        kp: (pages, page tokens, KV heads, dim) arena key pool.
        vp: The arena value pool, shaped as kp.
        rel: (rows, heads, rank) relative states.
        sink: (rows, heads) view of the sink logits.
        proj_t: (extent, rank) bf16 bias profiles, transposed.
        out: (rows, heads, dim) output; only the tokens' rows are written.
        tokens: (sequence ids, query indices, rows) of the tokens, on the
            GPU.
        cu_q: Each sequence's query bounds over the rows.
        used: Keys per sequence, its queries' included.
        table: Each sequence's arena pages.
        window: Keys each row sees, or 0 for every earlier key.
        max_keys: The most keys any sequence holds.
        block_keys: Keys per key tile.
        num_warps: Warps per program.
        num_stages: Software pipeline stages of the key loop.
        num_real: A one-element GPU tensor of how many of the tokens are
            real, or None when all are; a CUDA graph's grid covers its
            bucket's tokens.
        splits: Key splits per token, or None to choose them from
            max_keys.
    """
    count = tokens[0].numel()
    if count == 0:
        return out
    heads, dim = q3.shape[1], q3.shape[2]
    kv_heads, page = kp.shape[2], kp.shape[1]
    group = heads // kv_heads
    extent, rank = proj_t.shape
    if splits is None:
        splits = num_splits(count * kv_heads, max_keys, window)
    limit = num_real is not None
    if not limit:
        num_real = tokens[0]
    if splits > 1:
        part_acc = torch.empty((count * heads * splits, dim),
                               dtype=torch.float32, device=q3.device)
        part_ml = torch.empty((count * heads * splits, 2),
                              dtype=torch.float32, device=q3.device)
    else:
        part_acc = part_ml = out
    _kernels()["decode_attention"][(count * splits, kv_heads)](
        q3, kp, vp, rel, proj_t, sink, out, part_acc, part_ml, tokens[0],
        tokens[1], num_real, cu_q, used, table, q3.stride(0), q3.stride(1),
        rel.stride(0), rel.stride(1), sink.stride(0), out.stride(0),
        out.stride(1), kp.stride(0), kp.stride(1), kp.stride(2), vp.stride(0),
        vp.stride(1), vp.stride(2), table.stride(0),
        dim ** -0.5 * 1.4426950408889634, int(window), splits, GROUP=group,
        HEADS=heads, BM=16, BN=block_keys, D=dim, R=rank, EXTENT=extent,
        PAGE=page, HAS_WINDOW=window > 0, SPLIT=splits > 1, LIMIT=limit,
        num_warps=num_warps, num_stages=num_stages)
    if splits > 1:
        _kernels()["merge_tokens"][(count, heads)](
            part_acc, part_ml, tokens[2], num_real, out, out.stride(0),
            out.stride(1), splits, HEADS=heads, D=dim, MAX_SPLITS=MAX_SPLITS,
            LIMIT=limit, num_warps=4)
    return out


# at most this many programs share one query tile's keys
MAX_SPLITS = 16
# programs that fill a B200 several times over
TARGET_PROGRAMS = 4 * 148
# the fewest keys a split reads
SPLIT_KEYS = 512


def num_splits(programs, max_keys, window) -> int:
    """Key splits per query tile: enough programs to fill the GPU."""
    keys = min(max_keys, window) if window else max_keys
    if programs >= TARGET_PROGRAMS or keys < 2 * SPLIT_KEYS:
        return 1
    return int(max(1, min(MAX_SPLITS, -(-TARGET_PROGRAMS // programs),
                          keys // SPLIT_KEYS)))


def triton_cdiv(a, b):
    return -(-a // b)


def _chunk(width, largest):
    """The widest power-of-two chunk up to largest that divides width."""
    chunk = largest
    while width % chunk:
        chunk //= 2
    return chunk


def _hardware_e2m1(device) -> bool:
    import torch

    return torch.cuda.get_device_capability(device)[0] >= 10


def quantize_tokens(torch, x, c_map, ids, offsets, sf_offsets, *, nvfp4,
                    sf_rows):
    """Quantize each token once into its rows of the permuted expert layout.

    Returns:
        NVFP4: (packed (P, K/2) uint8, swizzled scales (sf_rows, K/16)
        fp8, row scales (P,) fp32). MXFP8: (bf16 (P, K) rounded rows,
        None, None).
    """
    tokens, k = x.shape
    topk = ids.shape[1]
    rows = tokens * topk
    device = x.device
    # the swizzled scales come in whole 128-row tiles
    sf_rows = -(-sf_rows // 128) * 128
    if nvfp4:
        q = torch.empty((rows, k // 2), dtype=torch.uint8, device=device)
        sf = torch.zeros((sf_rows, (k // 16 + 3) // 4 * 4),
                         dtype=torch.float8_e4m3fn, device=device)
        row_scale = torch.empty((rows,), dtype=torch.float32, device=device)
        bf16 = q
    else:
        bf16 = torch.empty((rows, k), dtype=torch.bfloat16, device=device)
        q = sf = row_scale = bf16
    if tokens:
        _kernels()["quantize_tokens"][(tokens,)](
            x, x.stride(0), c_map, ids, offsets, sf_offsets, q, sf,
            row_scale, bf16, K=k, CH=_chunk(k, 256), TOPK=topk, NVFP4=nvfp4,
            HW=_hardware_e2m1(device), LINEAR=False, num_warps=4)
    return (q, sf, row_scale) if nvfp4 else (bf16, None, None)


def quantize_dense(torch, x):
    """Quantize each row of x to NVFP4 once, with row-major block scales.

    Returns:
        (packed (rows, K/2) uint8, scales (rows, K/16) uint8 holding fp8,
        row scales (rows,) fp32), as FlashInfer's fused MoE takes them.
    """
    rows, k = x.shape
    device = x.device
    q = torch.empty((rows, k // 2), dtype=torch.uint8, device=device)
    sf = torch.empty((rows, k // 16), dtype=torch.float8_e4m3fn,
                     device=device)
    row_scale = torch.empty((rows,), dtype=torch.float32, device=device)
    if rows:
        # one expert holding every row in order
        c_map = torch.arange(rows, dtype=torch.int32, device=device)
        zeros = torch.zeros(rows, dtype=torch.int32, device=device)
        _kernels()["quantize_tokens"][(rows,)](
            x, x.stride(0), c_map, zeros, zeros, zeros, q, sf, row_scale, q,
            K=k, CH=_chunk(k, 256), TOPK=1, NVFP4=True,
            HW=_hardware_e2m1(device), LINEAR=True, num_warps=4)
    return q, sf.view(torch.uint8), row_scale


def swiglu_quantize(torch, gate_up, row_scale, gamma, expert, offsets,
                    sf_offsets, *, nvfp4, sf_rows, gate_cap, up_cap):
    """The softcapped SwiGLU of interleaved gate/up rows, quantized."""
    rows, width = gate_up.shape
    n = width // 2
    device = gate_up.device
    sf_rows = -(-sf_rows // 128) * 128
    if nvfp4:
        q = torch.empty((rows, n // 2), dtype=torch.uint8, device=device)
        sf = torch.zeros((sf_rows, (n // 16 + 3) // 4 * 4),
                         dtype=torch.float8_e4m3fn, device=device)
        out_scale = torch.empty((rows,), dtype=torch.float32, device=device)
        bf16 = q
    else:
        bf16 = torch.empty((rows, n), dtype=torch.bfloat16, device=device)
        q = sf = out_scale = bf16
    if rows:
        _kernels()["swiglu_quantize"][(rows,)](
            gate_up, row_scale if row_scale is not None else gamma, gamma,
            expert, offsets, sf_offsets, q, sf, out_scale, bf16, gate_cap,
            up_cap, N=n, HEAD=_pow2(n + 1) // 2, TAIL=n - _pow2(n + 1) // 2,
            NVFP4=nvfp4, HW=_hardware_e2m1(device), num_warps=4)
    return (q, sf, out_scale) if nvfp4 else (bf16, None, None)


def combine(torch, down, row_scale, c_map, tokens, topk):
    """Each token's sum of its experts' rows, in fp32, as bf16."""
    k = down.shape[1]
    out = torch.empty((tokens, k), dtype=torch.bfloat16, device=down.device)
    if tokens:
        _kernels()["combine"][(tokens,)](
            down, row_scale if row_scale is not None else down, c_map, out,
            K=k, KP=_pow2(k), TOPK=topk, SCALED=row_scale is not None,
            num_warps=_warps(_pow2(k)))
    return out


def save_state(torch, x, slots, sources, state):
    """Write each slot's three new convolution columns into its saved state.

    The columns are gathered before any is written, since a slot's new
    columns can come from another written slot's old ones.

    Args:
        torch: The torch module.
        x: (rows, channels) convolution inputs of this chunk.
        slots: (writes,) int64 slots; a negative slot is padding.
        sources: (writes, 3) int32 tap sources of each slot's columns.
        state: (slots, 3, channels) this convolution's saved columns.
    """
    writes = slots.shape[0]
    if writes == 0:
        return
    channels = x.shape[1]
    values = torch.empty((writes, 3, channels), dtype=torch.bfloat16,
                         device=x.device)
    block = 1024
    grid = (writes, 3, triton_cdiv(channels, block))
    _kernels()["gather_state"][grid](
        x, x.stride(0), sources, slots, state, state.stride(0),
        state.stride(1), values, channels, BLOCK=block, num_warps=4)
    _kernels()["scatter_state"][grid](
        values, slots, state, state.stride(0), state.stride(1), channels,
        BLOCK=block, num_warps=4)
