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
    def mix(blocks_ptr, stride_bb, stride_br, partial_ptr, stride_p,
            addend_ptr, stride_a, query_ptr, weight_ptr, out_ptr, stride_oq,
            stride_or, eps, H: tl.constexpr, HP: tl.constexpr,
            NB: tl.constexpr, NS: tl.constexpr, NQ: tl.constexpr,
            HAS_ADDEND: tl.constexpr, NORM: tl.constexpr):
        # states: NB completed blocks, then the partial (+ addend) if any
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, HP)
        mask = offs < H
        logits = tl.zeros((4, 8), dtype=tl.float32) - float("inf")
        rows4 = tl.arange(0, 4)
        cols8 = tl.arange(0, 8)
        queries = tl.load(query_ptr + rows4[:, None] * H + offs[None, :],
                          mask=(rows4[:, None] < NQ) & mask[None, :],
                          other=0.0)
        for j in tl.static_range(NS):
            if j < NB:
                x = tl.load(blocks_ptr + j * stride_bb + row * stride_br
                            + offs, mask=mask, other=0.0).to(tl.float32)
            else:
                x = tl.load(partial_ptr + row * stride_p + offs, mask=mask,
                            other=0.0).to(tl.float32)
                if HAS_ADDEND:
                    x += tl.load(addend_ptr + row * stride_a + offs,
                                 mask=mask, other=0.0).to(tl.float32)
            inv = tl.math.rsqrt(tl.sum(x * x, axis=0) / H + eps)
            score = tl.sum(queries * (x * inv)[None, :], axis=1)
            logits = tl.where(cols8[None, :] == j, score[:, None], logits)
        top = tl.max(logits, axis=1)
        p = tl.exp(logits - top[:, None])
        p = p / tl.sum(p, axis=1)[:, None]
        acc = tl.zeros((4, HP), dtype=tl.float32)
        for j in tl.static_range(NS):
            if j < NB:
                x = tl.load(blocks_ptr + j * stride_bb + row * stride_br
                            + offs, mask=mask, other=0.0).to(tl.float32)
            else:
                x = tl.load(partial_ptr + row * stride_p + offs, mask=mask,
                            other=0.0).to(tl.float32)
                if HAS_ADDEND:
                    x += tl.load(addend_ptr + row * stride_a + offs,
                                 mask=mask, other=0.0).to(tl.float32)
            inv = tl.math.rsqrt(tl.sum(x * x, axis=0) / H + eps)
            weight_j = tl.sum(tl.where(cols8[None, :] == j, p, 0.0), axis=1)
            acc += weight_j[:, None] * (x * inv)[None, :]
        for q in tl.static_range(NQ):
            out = tl.sum(tl.where(rows4[:, None] == q, acc, 0.0), axis=0)
            if NORM:
                w = tl.load(weight_ptr + q * H + offs, mask=mask,
                            other=0.0).to(tl.float32)
                out = out * tl.math.rsqrt(tl.sum(out * out, axis=0) / H + eps)
                out = out * w
            tl.store(out_ptr + q * stride_oq + row * stride_or + offs,
                     out.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _message(prev_block_ptr, prev_out_ptr, stride_prev, query_ptr, row,
                 offs, mask, eps, H: tl.constexpr):
        a = tl.load(prev_block_ptr + row * stride_prev + offs, mask=mask,
                    other=0.0).to(tl.float32)
        b = tl.load(prev_out_ptr + row * stride_prev + offs, mask=mask,
                    other=0.0).to(tl.float32)
        q = tl.load(query_ptr + offs, mask=mask, other=0.0)
        a = a * tl.math.rsqrt(tl.sum(a * a, axis=0) / H + eps)
        b = b * tl.math.rsqrt(tl.sum(b * b, axis=0) / H + eps)
        sa = tl.sum(q * a, axis=0)
        sb = tl.sum(q * b, axis=0)
        top = tl.maximum(sa, sb)
        ea = tl.exp(sa - top)
        eb = tl.exp(sb - top)
        return (ea * a + eb * b) / (ea + eb)

    @triton.jit
    def gate_inputs(hidden_ptr, stride_h, prev_block_ptr, prev_out_ptr,
                    stride_prev, query_ptr, out_ptr, stride_o, eps,
                    H: tl.constexpr, HP: tl.constexpr):
        # [norm(hidden) | norm(message)] in bf16, the gate's down input
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, HP)
        mask = offs < H
        h = tl.load(hidden_ptr + row * stride_h + offs, mask=mask,
                    other=0.0).to(tl.float32)
        m = _message(prev_block_ptr, prev_out_ptr, stride_prev, query_ptr,
                     row, offs, mask, eps, H)
        h = h * tl.math.rsqrt(tl.sum(h * h, axis=0) / H + eps)
        m = m * tl.math.rsqrt(tl.sum(m * m, axis=0) / H + eps)
        tl.store(out_ptr + row * stride_o + offs, h.to(tl.bfloat16), mask=mask)
        tl.store(out_ptr + row * stride_o + H + offs, m.to(tl.bfloat16),
                 mask=mask)

    @triton.jit
    def gate_apply(hidden_ptr, stride_h, prev_block_ptr, prev_out_ptr,
                   stride_prev, query_ptr, gains_ptr, stride_g, weight_ptr,
                   out_ptr, stride_o, eps, H: tl.constexpr, HP: tl.constexpr):
        # norm((1 - tanh(f)) * hidden + tanh(g) * message) * weight
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, HP)
        mask = offs < H
        h = tl.load(hidden_ptr + row * stride_h + offs, mask=mask,
                    other=0.0).to(tl.float32)
        m = _message(prev_block_ptr, prev_out_ptr, stride_prev, query_ptr,
                     row, offs, mask, eps, H)
        forget = libdevice.tanh(tl.load(gains_ptr + row * stride_g + offs,
                                        mask=mask, other=0.0))
        gain = libdevice.tanh(tl.load(gains_ptr + row * stride_g + H + offs,
                                      mask=mask, other=0.0))
        out = (1.0 - forget) * h + gain * m
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        out = out * tl.math.rsqrt(tl.sum(out * out, axis=0) / H + eps) * w
        tl.store(out_ptr + row * stride_o + offs, out.to(tl.bfloat16),
                 mask=mask)

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
                     used, window, profile_row, m_i, l_i, acc, scale_log2,
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
            near = valid & (dist < EXTENT)
            s += tl.load(profile_row[:, None] + (EXTENT - 1 - dist),
                         mask=near, other=0.0).to(tl.float32) * LOG2E
            s = tl.where(valid, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        return m_new, l_i, acc

    @triton.jit(do_not_specialize=["window"])
    def rel_attention(q_ptr, k_ptr, v_ptr, profile_ptr, sink_ptr, out_ptr,
                      tile_seq_ptr, tile_q0_ptr, cu_q_ptr, used_ptr,
                      table_ptr, stride_q0, stride_q1, stride_s0, stride_o0,
                      stride_o1, stride_k0, stride_k1, stride_k2, stride_v0,
                      stride_v1, stride_v2, stride_bt, scale_log2, window,
                      GROUP: tl.constexpr, HEADS: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
                      EXTENT: tl.constexpr, PAGE: tl.constexpr,
                      HAS_WINDOW: tl.constexpr):
        # profile holds each row and head's bias by distance, longest
        # first, (rows, heads, EXTENT): a row's bias over a key tile is one
        # ascending run, so its loads coalesce
        LOG2E: tl.constexpr = 1.4426950408889634
        TQ: tl.constexpr = BM // GROUP
        tile = tl.program_id(0)
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
        profile_row = profile_ptr + (q_row * HEADS + head) * EXTENT
        m_i = sink * LOG2E
        l_i = tl.zeros((BM,), dtype=tl.float32) + 1.0
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
        bt_row = table_ptr + seq.to(tl.int64) * stride_bt
        for n0 in range(start, near, BN):
            m_i, l_i, acc = _attend_tile(
                q, k_ptr, v_ptr, bt_row, n0, kv_head, qpos, row_ok, used,
                window, profile_row, m_i, l_i, acc, scale_log2, stride_k0,
                stride_k1, stride_k2, stride_v0, stride_v1, stride_v2, BN, D,
                EXTENT, PAGE, HAS_WINDOW, False)
        for n0 in range(near, last_q + 1, BN):
            m_i, l_i, acc = _attend_tile(
                q, k_ptr, v_ptr, bt_row, n0, kv_head, qpos, row_ok, used,
                window, profile_row, m_i, l_i, acc, scale_log2, stride_k0,
                stride_k1, stride_k2, stride_v0, stride_v1, stride_v2, BN, D,
                EXTENT, PAGE, HAS_WINDOW, True)
        out = acc / l_i[:, None]
        tl.store(out_ptr + q_row[:, None] * stride_o0
                 + head[:, None] * stride_o1 + offs_d[None, :],
                 out.to(tl.bfloat16), mask=row_ok[:, None])

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
                        HW: tl.constexpr):
        # one token, quantized once and stored to each of its TOPK
        # permuted rows; NVFP4 writes packed values, swizzled scales, and
        # the row scale, else the MXFP8-rounded row in bf16
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
                tl.store(sf_ptr + _scale_offset(
                    sf_rows[:, None], block[None, :], (K // 16 + 3) // 4),
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
        # the softcapped SwiGLU of columns [start, start + W), from
        # (gate, up) pairs interleaved over 2N columns; the result is
        # rounded to bf16, so approximate tanh and sigmoid suffice
        pairs = tl.load(gate_up_ptr + p * 2 * N + 2 * start
                        + tl.arange(0, 2 * W)).to(tl.float32)
        gate, up = tl.split(tl.reshape(pairs, (W, 2)))
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
    def gather_state(x_ptr, stride_x, srcs_ptr, state_ptr, stride_slot,
                     stride_tap, out_ptr, C, BLOCK: tl.constexpr):
        # out[w, j] = the value of tap source srcs[w, j]
        w = tl.program_id(0).to(tl.int64)
        j = tl.program_id(1)
        cols = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < C
        src = tl.load(srcs_ptr + w * 3 + j)
        value = _tap(x_ptr, stride_x, state_ptr, stride_slot, stride_tap, src,
                     cols, mask)
        tl.store(out_ptr + (w * 3 + j) * C + cols, value.to(tl.bfloat16),
                 mask=mask)

    _KERNELS.update(
        mix=mix, gate_inputs=gate_inputs, gate_apply=gate_apply,
        rms_norm=rms_norm, short_conv=short_conv, kv_conv=kv_conv,
        clip_heads=clip_heads, gate_attention=gate_attention,
        rel_attention=rel_attention, quantize_tokens=quantize_tokens,
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
        blocks: (max blocks, rows, hidden) bf16; the first num_blocks hold
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
    rows, hidden = blocks.shape[1], blocks.shape[2]
    nq = query.shape[0]
    dtype = torch.bfloat16 if weight is not None else torch.float32
    out = torch.empty((nq, rows, hidden), dtype=dtype, device=blocks.device)
    if rows == 0:
        return out
    has_partial = partial is not None
    partial = partial if has_partial else blocks[0]
    addend_t = addend if addend is not None else partial
    _kernels()["mix"][(rows,)](
        blocks, blocks.stride(0), blocks.stride(1), partial, partial.stride(0),
        addend_t, addend_t.stride(0), query,
        weight if weight is not None else query, out, out.stride(0),
        out.stride(1), eps, H=hidden, HP=_hp(hidden), NB=num_blocks,
        NS=num_blocks + int(has_partial), NQ=nq,
        HAS_ADDEND=addend is not None,
        NORM=weight is not None, num_warps=16)
    return out


def gate_inputs(torch, hidden, prev_block, prev_out, query, eps=1e-6):
    """[norm(hidden) | norm(message)] in bf16, (rows, 2 * width)."""
    rows, width = hidden.shape
    out = torch.empty((rows, 2 * width), dtype=torch.bfloat16,
                      device=hidden.device)
    if rows:
        _kernels()["gate_inputs"][(rows,)](
            hidden, hidden.stride(0), prev_block, prev_out,
            prev_block.stride(0), query, out, out.stride(0), eps, H=width,
            HP=_hp(width), num_warps=16)
    return out


def gate_apply(torch, hidden, prev_block, prev_out, query, gains, weight,
               eps=1e-6, out=None):
    """The gated, RMS-normed input in bf16, (rows, width)."""
    rows, width = hidden.shape
    if out is None:
        out = torch.empty((rows, width), dtype=torch.bfloat16,
                          device=hidden.device)
    if rows:
        _kernels()["gate_apply"][(rows,)](
            hidden, hidden.stride(0), prev_block, prev_out,
            prev_block.stride(0), query, gains, gains.stride(0), weight, out,
            out.stride(0), eps, H=width, HP=_hp(width), num_warps=16)
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
                  table, *, window, block_rows=128, block_keys=32, num_warps=8,
                  num_stages=3):
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
    # each row and head's bias by distance, longest first, rounded to
    # bf16 as the reference's bias is
    profile = torch.mm(rel.reshape(rows * heads, rank), proj.flip(1))
    _kernels()["rel_attention"][(tiles[0].numel(), kv_heads)](
        q3, kp, vp, profile, sink, out, tiles[0], tiles[1], cu_q, used,
        table, q3.stride(0), q3.stride(1), sink.stride(0), out.stride(0),
        out.stride(1), kp.stride(0), kp.stride(1), kp.stride(2),
        vp.stride(0), vp.stride(1), vp.stride(2), table.stride(0),
        dim ** -0.5 * 1.4426950408889634, int(window), GROUP=group,
        HEADS=heads, BM=block_rows, BN=block_keys, D=dim, EXTENT=extent, PAGE=page,
        HAS_WINDOW=window > 0, num_warps=num_warps, num_stages=num_stages)
    return out


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
            HW=_hardware_e2m1(device), num_warps=4)
    return (q, sf, row_scale) if nvfp4 else (bf16, None, None)


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


def gather_state(torch, x, srcs, state):
    """(writes, 3, channels) bf16 values of each write's three sources."""
    writes = srcs.shape[0]
    channels = x.shape[1]
    out = torch.empty((writes, 3, channels), dtype=torch.bfloat16,
                      device=x.device)
    if writes:
        block = 1024
        _kernels()["gather_state"][(writes, 3, triton_cdiv(channels, block))](
            x, x.stride(0), srcs, state, state.stride(0), state.stride(1), out,
            channels, BLOCK=block, num_warps=4)
    return out
