"""Triton kernels for the Inkling 2 Mini forward pass.

Each kernel rounds to bf16 where the transformers reference rounds, so
the pass computes what the reference computes up to accumulation order.

- mix: a block attention residual. Per row, a softmax over the
  completed blocks and the running partial block, scored by a learned
  query against each state's RMS-normed values, then the weighted sum
  of the normed states, optionally RMS-normed with a weight.
- gate_inputs and gate_apply: a loop message gate. The message is a
  softmax mix of the previous pass's last block and output; the gate
  mixes it into the hidden state with tanh gains from a rank-64
  projection of both, then RMS-norms the result.
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
- gate_attention: the sigmoid output gate of every head.
- quantize_tokens, swiglu_quantize, combine: the NVFP4 expert path's
  activation quantization (four-over-six scales), its softcapped
  SwiGLU, and the weighted sum of each token's experts.
"""

import math

# a tap source in a conv tap table: a chunk row (>= 0), zero (-1), or a
# saved state column of slot s, tap j (-(2 + 3 * s + j))
ZERO_TAP = -1

_KERNELS = {}


def _pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def _kernels() -> dict:
    """Compile the kernels on first use; Triton needs a GPU process."""
    if _KERNELS:
        return _KERNELS
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

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
            stride_or, eps, H: tl.constexpr, CH: tl.constexpr,
            NB: tl.constexpr, NS: tl.constexpr, NQ: tl.constexpr,
            HAS_ADDEND: tl.constexpr, NORM: tl.constexpr):
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
            if NORM:
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
            if NORM:
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
            if NORM:
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
    def gate_inputs(hidden_ptr, stride_hg, stride_h, prev_block_ptr,
                    prev_out_ptr, stride_prev, query_ptr, out_ptr, stride_og,
                    stride_o, eps, H: tl.constexpr, CH: tl.constexpr):
        # [norm(hidden) | norm(message)] in bf16 for gate program_id(1); the
        # first pass sums products in registers, the second writes
        row = tl.program_id(0).to(tl.int64)
        gate = tl.program_id(1)
        offs = tl.arange(0, CH)
        hidden = hidden_ptr + gate * stride_hg + row * stride_h
        block = prev_block_ptr + row * stride_prev
        prev = prev_out_ptr + row * stride_prev
        query = query_ptr + gate * H
        hh = tl.zeros((CH,), dtype=tl.float32)
        aa = tl.zeros((CH,), dtype=tl.float32)
        bb = tl.zeros((CH,), dtype=tl.float32)
        ab = tl.zeros((CH,), dtype=tl.float32)
        qa = tl.zeros((CH,), dtype=tl.float32)
        qb = tl.zeros((CH,), dtype=tl.float32)
        for c in range(0, H, CH):
            cols = c + offs
            h = tl.load(hidden + cols).to(tl.float32)
            a = tl.load(block + cols).to(tl.float32)
            b = tl.load(prev + cols).to(tl.float32)
            q = tl.load(query + cols)
            hh += h * h
            aa += a * a
            bb += b * b
            ab += a * b
            qa += q * a
            qb += q * b
        alpha, beta, mm = _blend(tl.sum(aa, 0), tl.sum(bb, 0), tl.sum(ab, 0),
                                 tl.sum(qa, 0), tl.sum(qb, 0), eps, H)
        inv_h = tl.math.rsqrt(tl.sum(hh, 0) / H + eps)
        inv_m = tl.math.rsqrt(mm / H + eps)
        alpha *= inv_m
        beta *= inv_m
        out = out_ptr + gate * stride_og + row * stride_o
        for c in range(0, H, CH):
            cols = c + offs
            h = tl.load(hidden + cols).to(tl.float32)
            a = tl.load(block + cols).to(tl.float32)
            b = tl.load(prev + cols).to(tl.float32)
            tl.store(out + cols, (h * inv_h).to(tl.bfloat16))
            tl.store(out + H + cols, (alpha * a + beta * b).to(tl.bfloat16))

    @triton.jit
    def _gate_part(hidden, block, prev, query, gains, start, W: tl.constexpr,
                   H: tl.constexpr):
        # columns [start, start + W) of one gate's row: u = (1 - tanh(f)) h,
        # v = tanh(g), and the previous pass's block a and output b
        cols = start + tl.arange(0, W)
        h = tl.load(hidden + cols).to(tl.float32)
        a = tl.load(block + cols).to(tl.float32)
        b = tl.load(prev + cols).to(tl.float32)
        q = tl.load(query + cols)
        u = (1.0 - libdevice.tanh(tl.load(gains + cols))) * h
        v = libdevice.tanh(tl.load(gains + H + cols))
        return u, v, a, b, q

    @triton.jit
    def gate_apply(hidden_ptr, stride_hg, stride_h, prev_block_ptr,
                   prev_out_ptr, stride_prev, query_ptr, gains_ptr,
                   stride_gg, stride_g, weight_ptr, out_ptr, stride_og,
                   stride_o, eps, H: tl.constexpr, HEAD: tl.constexpr,
                   TAIL: tl.constexpr):
        # norm((1 - tanh(f)) h + tanh(g) m) * weight in one pass, the row
        # held as a power-of-two head and a power-of-two tail
        row = tl.program_id(0).to(tl.int64)
        gate = tl.program_id(1)
        hidden = hidden_ptr + gate * stride_hg + row * stride_h
        block = prev_block_ptr + row * stride_prev
        prev = prev_out_ptr + row * stride_prev
        query = query_ptr + gate * H
        gains = gains_ptr + gate * stride_gg + row * stride_g
        u, v, a, b, q = _gate_part(hidden, block, prev, query, gains, 0, HEAD,
                                   H)
        aa = tl.sum(a * a, 0)
        bb = tl.sum(b * b, 0)
        ab = tl.sum(a * b, 0)
        qa = tl.sum(q * a, 0)
        qb = tl.sum(q * b, 0)
        if TAIL > 0:
            u2, v2, a2, b2, q2 = _gate_part(hidden, block, prev, query, gains,
                                            HEAD, TAIL, H)
            aa += tl.sum(a2 * a2, 0)
            bb += tl.sum(b2 * b2, 0)
            ab += tl.sum(a2 * b2, 0)
            qa += tl.sum(q2 * a2, 0)
            qb += tl.sum(q2 * b2, 0)
        alpha, beta, mm = _blend(aa, bb, ab, qa, qb, eps, H)
        y = u + v * (alpha * a + beta * b)
        total = tl.sum(y * y, 0)
        if TAIL > 0:
            y2 = u2 + v2 * (alpha * a2 + beta * b2)
            total += tl.sum(y2 * y2, 0)
        scale = tl.math.rsqrt(total / H + eps)
        out = out_ptr + gate * stride_og + row * stride_o
        cols = tl.arange(0, HEAD)
        w = tl.load(weight_ptr + gate * H + cols).to(tl.float32)
        tl.store(out + cols, (y * scale * w).to(tl.bfloat16))
        if TAIL > 0:
            cols2 = HEAD + tl.arange(0, TAIL)
            w2 = tl.load(weight_ptr + gate * H + cols2).to(tl.float32)
            tl.store(out + cols2, (y2 * scale * w2).to(tl.bfloat16))

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
    def kv_conv(x_ptr, stride_x, dyn_ptr, stride_dyn, taps_ptr, state_ptr,
                stride_slot, stride_tap, w_ptr, out_ptr, stride_o0, stride_o1,
                eps, clip, inv_scale, D: tl.constexpr, DYNAMIC: tl.constexpr,
                IS_KEY: tl.constexpr):
        # static: w is (C, 4), w[:, 3] current; dynamic: w is (4, C) by
        # backward distance, plus the row's (heads, 4) taps at dyn_ptr
        row = tl.program_id(0).to(tl.int64)
        head = tl.program_id(1)
        cols = head * D + tl.arange(0, D)
        mask = cols < (head + 1) * D
        C = tl.num_programs(1) * D
        x = tl.load(x_ptr + row * stride_x + cols).to(tl.float32)
        srcs0 = tl.load(taps_ptr + row * 3)
        srcs1 = tl.load(taps_ptr + row * 3 + 1)
        srcs2 = tl.load(taps_ptr + row * 3 + 2)
        t3 = _tap(x_ptr, stride_x, state_ptr, stride_slot, stride_tap, srcs0,
                  cols, mask)
        t2 = _tap(x_ptr, stride_x, state_ptr, stride_slot, stride_tap, srcs1,
                  cols, mask)
        t1 = _tap(x_ptr, stride_x, state_ptr, stride_slot, stride_tap, srcs2,
                  cols, mask)
        if DYNAMIC:
            d0 = tl.load(dyn_ptr + row * stride_dyn + head * 4).to(tl.float32)
            d1 = tl.load(dyn_ptr + row * stride_dyn + head * 4 + 1).to(tl.float32)
            d2 = tl.load(dyn_ptr + row * stride_dyn + head * 4 + 2).to(tl.float32)
            d3 = tl.load(dyn_ptr + row * stride_dyn + head * 4 + 3).to(tl.float32)
            w0 = tl.load(w_ptr + cols).to(tl.float32) + d0
            w1 = tl.load(w_ptr + C + cols).to(tl.float32) + d1
            w2 = tl.load(w_ptr + 2 * C + cols).to(tl.float32) + d2
            w3 = tl.load(w_ptr + 3 * C + cols).to(tl.float32) + d3
            acc = w0 * x + w1 * t1 + w2 * t2 + w3 * t3
        else:
            acc = (tl.load(w_ptr + cols * 4 + 3) * x
                   + tl.load(w_ptr + cols * 4 + 2) * t1
                   + tl.load(w_ptr + cols * 4 + 1) * t2
                   + tl.load(w_ptr + cols * 4) * t3)
        y = ((acc + x) * inv_scale).to(tl.bfloat16).to(tl.float32)
        if IS_KEY:
            norm = tl.sqrt(tl.sum(y * y, axis=0))
            y = y * tl.minimum(clip / (norm + 1e-6), 1.0)
        else:
            y = y * tl.math.rsqrt(tl.sum(y * y, axis=0) / D + eps)
        tl.store(out_ptr + row * stride_o0 + head * stride_o1
                 + tl.arange(0, D), y.to(tl.bfloat16))

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
        groups = tl.load(gate_up_ptr + p * 2 * N + 2 * start
                         + tl.arange(0, 2 * W)).to(tl.float32)
        groups = tl.permute(tl.reshape(groups, (W // 64, 2, 64)), (0, 2, 1))
        up, gate = tl.split(groups)
        up = tl.reshape(up, (W,))
        gate = tl.reshape(gate, (W,))
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

    _KERNELS.update(
        scatter_state=scatter_state,
        mix=mix, gate_inputs=gate_inputs, gate_apply=gate_apply,
        rms_norm=rms_norm, short_conv=short_conv, kv_conv=kv_conv,
        clip_heads=clip_heads, gate_attention=gate_attention,
        rel_attention=rel_attention, merge_splits=merge_splits,
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
        weight=None, eps=1e-6):
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
            unnormed in fp32.
        eps: The RMS norms' epsilon.

    Returns:
        (queries, rows, hidden): bf16 when normed, else fp32.
    """
    rows, hidden = blocks.shape[0], blocks.shape[2]
    nq = query.shape[0]
    dtype = torch.bfloat16 if weight is not None else torch.float32
    out = torch.empty((nq, rows, hidden), dtype=dtype, device=blocks.device)
    if rows == 0:
        return out
    has_partial = partial is not None
    partial = partial if has_partial else blocks[:, 0]
    addend_t = addend if addend is not None else partial
    _kernels()["mix"][(rows,)](
        blocks, blocks.stride(1), blocks.stride(0), partial, partial.stride(0),
        addend_t, addend_t.stride(0), query,
        weight if weight is not None else query, out, out.stride(0),
        out.stride(1), eps, H=hidden, CH=_chunk(hidden, 256), NB=num_blocks,
        NS=num_blocks + int(has_partial), NQ=nq,
        HAS_ADDEND=addend is not None, NORM=weight is not None, num_warps=1)
    return out


def _gate_hidden(hidden):
    """The gate and row strides of a shared or per-gate hidden tensor."""
    if hidden.dim() == 2:
        return 0, hidden.stride(0)
    return hidden.stride(0), hidden.stride(1)


def gate_inputs(torch, hidden, prev_block, prev_out, query, eps=1e-6):
    """Each gate's [norm(hidden) | norm(message)] in bf16, (gates, rows, 2w).

    Args:
        torch: The torch module.
        hidden: (rows, width) shared by every gate, or (gates, rows, width).
        prev_block: (rows, width) the previous pass's last block.
        prev_out: (rows, width) the previous pass's output.
        query: (gates, width) fp32 message queries, pre-scaled.
        eps: The norms' epsilon.
    """
    gates, width = query.shape
    rows = prev_block.shape[0]
    out = torch.empty((gates, rows, 2 * width), dtype=torch.bfloat16,
                      device=prev_block.device)
    if rows:
        stride_hg, stride_h = _gate_hidden(hidden)
        _kernels()["gate_inputs"][(rows, gates)](
            hidden, stride_hg, stride_h, prev_block, prev_out,
            prev_block.stride(0), query, out, out.stride(0), out.stride(1),
            eps, H=width, CH=_chunk(width, 256), num_warps=1)
    return out


def gate_apply(torch, hidden, prev_block, prev_out, query, gains, weight,
               eps=1e-6):
    """Each gate's gated, RMS-normed input in bf16, (gates, rows, width).

    Args:
        torch: The torch module.
        hidden: (rows, width) shared by every gate, or (gates, rows, width).
        prev_block: (rows, width) the previous pass's last block.
        prev_out: (rows, width) the previous pass's output.
        query: (gates, width) fp32 message queries, pre-scaled.
        gains: (gates, rows, 2 * width) fp32 forget and message gains
            before their tanh.
        weight: (gates, width) RMS norm weights.
        eps: The norms' epsilon.
    """
    gates, width = query.shape
    rows = prev_block.shape[0]
    out = torch.empty((gates, rows, width), dtype=torch.bfloat16,
                      device=prev_block.device)
    if rows:
        stride_hg, stride_h = _gate_hidden(hidden)
        _kernels()["gate_apply"][(rows, gates)](
            hidden, stride_hg, stride_h, prev_block, prev_out,
            prev_block.stride(0), query, gains, gains.stride(0),
            gains.stride(1), weight, out, out.stride(0), out.stride(1), eps,
            H=width, HEAD=_pow2(width + 1) // 2,
            TAIL=width - _pow2(width + 1) // 2, num_warps=8)
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
    _kernels()["kv_conv"][(rows, heads)](
        x, x.stride(0), dyn, dyn.stride(0), taps, state, state.stride(0),
        state.stride(1), weight, out, out.stride(0), out.stride(1), eps, clip,
        scale, D=head_dim, DYNAMIC=dynamic is not None, IS_KEY=is_key,
        num_warps=4)
    return out


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


def attention_tiles(cu_q_host, rows_per_tile):
    """(sequence, first query) per query tile, over sequences' query counts."""
    import numpy as np

    lengths = np.diff(np.asarray(cu_q_host, dtype=np.int64))
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
