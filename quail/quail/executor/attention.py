"""The packed forward pass: DeepGEMM matmuls, three Triton fused
kernels, FlashAttention-3 varlen self-attention, paged cross-attention
against the arena, and the softmax-state (LSE) merge.

A chunk is [group_1 | group_2 | ...] where each group is
[prefix? | suffix_1 .. suffix_k]. Per layer, attention is two calls
merged by softmax state: call A is causal over the segment boundaries
(each prefix over itself, each suffix over itself), call B is
non-causal - every suffix token against its group's kept context,
which lives in the arena's pages (fresh prefixes are scattered into
their pages in the same layer, before call B reads them). Suffix
positions start at the kept-context length, identical to standalone
requests, so answers are comparable to per-pair prompts.

Ported from the exploration's join executor; the one structural change
is that call B reads paged KV through a block table instead of
chunk-slice/kept-tensor concatenation. meta["paged"]=False switches to
the gather fallback (contiguous copies) if the paged kernel ever fails
a parity gate.

Everything here imports torch lazily: the module only runs inside the
Modal image.
"""

GROUP = 128            # fp8 quant group size, matches the engine


class Pipeline:
    """Packed forward passes with shared-prefix attention over the
    paged arena."""

    def __init__(self, model, arena):
        import torch
        from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

        self.torch = torch
        self.model = model
        self.arena = arena
        self.layers = model.model.layers
        self.embed = model.model.embed_tokens
        self.final_norm = model.model.norm
        self.rotary = self.layers[0].self_attn.rotary_emb
        attn = self.layers[0].self_attn
        self.num_q_heads = attn.num_heads
        self.num_kv_heads = attn.num_kv_heads
        self.head_dim = attn.head_dim
        self.use_ue8m0 = bool(is_deep_gemm_e8m0_used())
        self.fp8 = torch.float8_e4m3fn
        # The fused kernels compute element offsets in 32-bit ints,
        # so a chunk needs rows x widest_row < 2^31. Derived from the
        # loaded weights so it holds for any checkpoint; the budget
        # arithmetic derives the same cap from ffn_width in the spec.
        widest = max(max(layer.self_attn.qkv_proj.weight.shape[0],
                         layer.mlp.gate_up_proj.weight.shape[0])
                     for layer in self.layers)
        self.max_chunk_tokens = (2**31 - 1) // widest

    # ---- weights and quant ------------------------------------------

    @staticmethod
    def weight_scale(linear):
        for name in ("weight_scale", "weight_scale_inv"):
            scale = getattr(linear, name, None)
            if scale is not None:
                return scale
        raise AttributeError(f"no weight scale on {type(linear).__name__}")

    def gemm(self, q_input, input_scale, linear):
        from vllm.utils.deep_gemm import fp8_gemm_nt
        out = self.torch.empty(
            (q_input.shape[0], linear.weight.shape[0]),
            dtype=self.torch.bfloat16, device=q_input.device)
        fp8_gemm_nt((q_input, input_scale),
                    (linear.weight, self.weight_scale(linear)),
                    out, is_deep_gemm_e8m0_used=self.use_ue8m0)
        return out

    def quant(self, x):
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
        )
        return per_token_group_quant_fp8(
            x, group_size=GROUP, column_major_scales=True,
            use_ue8m0=self.use_ue8m0)

    def _col_major_scales(self, n_tokens, width):
        return self.torch.empty(
            (width // GROUP, n_tokens),
            dtype=self.torch.float32, device="cuda").permute(-1, -2)

    def rms_norm(self, x, norm):
        from vllm import _custom_ops as ops
        out = self.torch.empty_like(x)
        ops.rms_norm(out, x, norm.weight, norm.variance_epsilon)
        return out

    def fused_add_rms_norm(self, hidden, residual, norm):
        from vllm import _custom_ops as ops
        ops.fused_add_rms_norm(hidden, residual, norm.weight,
                               norm.variance_epsilon)
        return hidden, residual

    # ---- the three Triton kernels (exploration round 3/4, verbatim) --

    def _triton_kernels(self):
        if hasattr(self, "_kernels"):
            return self._kernels
        import triton
        import triton.language as tl

        @triton.jit
        def silu_mul_quant(gu_ptr, q_ptr, s_ptr, stride_gu, stride_q,
                           s_stride_g, s_stride_t,
                           HALF: tl.constexpr, GROUP_C: tl.constexpr,
                           GPB: tl.constexpr, UE8M0: tl.constexpr):
            t = tl.program_id(0)
            block = tl.program_id(1)
            offs = block * GROUP_C * GPB + tl.arange(0, GROUP_C * GPB)
            gate = tl.load(gu_ptr + t * stride_gu + offs).to(tl.float32)
            up = tl.load(gu_ptr + t * stride_gu + HALF + offs).to(tl.float32)
            y = gate * tl.sigmoid(gate) * up
            y2 = tl.reshape(y, (GPB, GROUP_C))
            amax = tl.max(tl.abs(y2), axis=1)
            scale = tl.maximum(amax, 1e-10) / 448.0
            if UE8M0:
                scale = tl.math.exp2(tl.ceil(tl.math.log2(scale)))
            q = y2 / scale[:, None]
            q = tl.minimum(tl.maximum(q, -448.0), 448.0)
            tl.store(q_ptr + t * stride_q + offs,
                     tl.reshape(q, (GROUP_C * GPB,)).to(
                         q_ptr.dtype.element_ty))
            g_idx = block * GPB + tl.arange(0, GPB)
            tl.store(s_ptr + g_idx * s_stride_g + t * s_stride_t, scale)

        @triton.jit
        def add_rms_norm_quant(x_ptr, res_ptr, w_ptr, q_ptr, s_ptr,
                               stride_x, stride_res, stride_q,
                               s_stride_g, s_stride_t, eps,
                               H: tl.constexpr, BLOCK: tl.constexpr,
                               GROUP_C: tl.constexpr, NG: tl.constexpr,
                               UE8M0: tl.constexpr):
            t = tl.program_id(0)
            offs = tl.arange(0, BLOCK)
            mask = offs < H
            x = tl.load(x_ptr + t * stride_x + offs, mask=mask,
                        other=0.0).to(tl.float32)
            r = tl.load(res_ptr + t * stride_res + offs, mask=mask,
                        other=0.0).to(tl.float32)
            x = x + r
            tl.store(res_ptr + t * stride_res + offs,
                     x.to(res_ptr.dtype.element_ty), mask=mask)
            ms = tl.sum(x * x, axis=0) / H
            rstd = 1.0 / tl.sqrt(ms + eps)
            w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            y = x * rstd * w
            y2 = tl.reshape(y, (BLOCK // GROUP_C, GROUP_C))
            amax = tl.max(tl.abs(y2), axis=1)
            scale = tl.maximum(amax, 1e-10) / 448.0
            if UE8M0:
                scale = tl.math.exp2(tl.ceil(tl.math.log2(scale)))
            q = y2 / scale[:, None]
            q = tl.minimum(tl.maximum(q, -448.0), 448.0)
            tl.store(q_ptr + t * stride_q + offs,
                     tl.reshape(q, (BLOCK,)).to(q_ptr.dtype.element_ty),
                     mask=mask)
            g_idx = tl.arange(0, BLOCK // GROUP_C)
            tl.store(s_ptr + g_idx * s_stride_g + t * s_stride_t, scale,
                     mask=g_idx < NG)

        @triton.jit
        def qk_norm_rope(qkv_ptr, q_out_ptr, k_out_ptr, cs_ptr, pos_ptr,
                         qw_ptr, kw_ptr, stride_qkv, stride_qo, stride_ko,
                         eps, QH: tl.constexpr, KH: tl.constexpr,
                         HD: tl.constexpr, HALF: tl.constexpr):
            t = tl.program_id(0)
            pos = tl.load(pos_ptr + t)
            half_offs = tl.arange(0, HALF)
            cos = tl.load(cs_ptr + pos * HD + half_offs).to(tl.float32)
            sin = tl.load(cs_ptr + pos * HD + HALF + half_offs).to(tl.float32)
            qw_a = tl.load(qw_ptr + half_offs).to(tl.float32)
            qw_b = tl.load(qw_ptr + HALF + half_offs).to(tl.float32)
            kw_a = tl.load(kw_ptr + half_offs).to(tl.float32)
            kw_b = tl.load(kw_ptr + HALF + half_offs).to(tl.float32)

            q_heads = tl.arange(0, QH)
            qa_offs = q_heads[:, None] * HD + half_offs[None, :]
            qb_offs = qa_offs + HALF
            qa = tl.load(qkv_ptr + t * stride_qkv + qa_offs).to(tl.float32)
            qb = tl.load(qkv_ptr + t * stride_qkv + qb_offs).to(tl.float32)
            ms = (tl.sum(qa * qa, axis=1) + tl.sum(qb * qb, axis=1)) / HD
            rstd = 1.0 / tl.sqrt(ms + eps)
            qa = qa * rstd[:, None] * qw_a[None, :]
            qb = qb * rstd[:, None] * qw_b[None, :]
            out_a = qa * cos[None, :] - qb * sin[None, :]
            out_b = qb * cos[None, :] + qa * sin[None, :]
            tl.store(q_out_ptr + t * stride_qo + qa_offs,
                     out_a.to(q_out_ptr.dtype.element_ty))
            tl.store(q_out_ptr + t * stride_qo + qb_offs,
                     out_b.to(q_out_ptr.dtype.element_ty))

            k_heads = tl.arange(0, KH)
            ka_offs = k_heads[:, None] * HD + half_offs[None, :]
            kb_offs = ka_offs + HALF
            base = qkv_ptr + t * stride_qkv + QH * HD
            ka = tl.load(base + ka_offs).to(tl.float32)
            kb = tl.load(base + kb_offs).to(tl.float32)
            ms = (tl.sum(ka * ka, axis=1) + tl.sum(kb * kb, axis=1)) / HD
            rstd = 1.0 / tl.sqrt(ms + eps)
            ka = ka * rstd[:, None] * kw_a[None, :]
            kb = kb * rstd[:, None] * kw_b[None, :]
            out_a = ka * cos[None, :] - kb * sin[None, :]
            out_b = kb * cos[None, :] + ka * sin[None, :]
            tl.store(k_out_ptr + t * stride_ko + ka_offs,
                     out_a.to(k_out_ptr.dtype.element_ty))
            tl.store(k_out_ptr + t * stride_ko + kb_offs,
                     out_b.to(k_out_ptr.dtype.element_ty))

        @triton.jit
        def lse_merge(rows_ptr, la_ptr, lb_ptr, a_ptr, b_ptr,
                      stride_lah, stride_lat, stride_lbh, stride_lbt,
                      H: tl.constexpr, D: tl.constexpr):
            # one program per (suffix row, head): out_a[src] =
            # lerp(out_a[src], out_b[r], sigmoid(lse_b - lse_a[src])).
            # Matches the unfused chain's rounding: sigmoid in fp32,
            # the weight rounded to bf16, the lerp in fp32 (torch's
            # opmath for bf16 lerp), stored as bf16.
            r = tl.program_id(0)
            h = tl.program_id(1)
            src = tl.load(rows_ptr + r)
            la = tl.load(la_ptr + h * stride_lah + src * stride_lat)
            lb = tl.load(lb_ptr + h * stride_lbh + r * stride_lbt)
            w = tl.sigmoid(lb - la).to(tl.bfloat16).to(tl.float32)
            offs = tl.arange(0, D)
            pa = a_ptr + src * H * D + h * D + offs
            pb = b_ptr + r * H * D + h * D + offs
            a = tl.load(pa).to(tl.float32)
            b = tl.load(pb).to(tl.float32)
            tl.store(pa, (a + w * (b - a)).to(a_ptr.dtype.element_ty))

        self._kernels = {"silu": silu_mul_quant,
                         "norm": add_rms_norm_quant,
                         "qk": qk_norm_rope,
                         "merge": lse_merge}
        return self._kernels

    def custom_silu_quant(self, gate_up):
        n, doubled = gate_up.shape
        half = doubled // 2
        gpb = 4
        q = self.torch.empty((n, half), dtype=self.fp8, device="cuda")
        scales = self._col_major_scales(n, half)
        self._triton_kernels()["silu"][(n, half // (GROUP * gpb))](
            gate_up, q, scales, gate_up.stride(0), q.stride(0),
            scales.stride(1), scales.stride(0),
            HALF=half, GROUP_C=GROUP, GPB=gpb, UE8M0=self.use_ue8m0)
        return q, scales

    def custom_norm_quant(self, hidden, norm, residual):
        n, h = hidden.shape
        block = 1 << (h - 1).bit_length()
        q = self.torch.empty((n, h), dtype=self.fp8, device="cuda")
        scales = self._col_major_scales(n, h)
        self._triton_kernels()["norm"][(n,)](
            hidden, residual, norm.weight, q, scales,
            hidden.stride(0), residual.stride(0), q.stride(0),
            scales.stride(1), scales.stride(0), norm.variance_epsilon,
            H=h, BLOCK=block, GROUP_C=GROUP, NG=h // GROUP,
            UE8M0=self.use_ue8m0)
        return q, scales

    def custom_qk_norm_rope(self, qkv, positions, attn):
        n = qkv.shape[0]
        qw = self.num_q_heads * self.head_dim
        kw = self.num_kv_heads * self.head_dim
        q = self.torch.empty((n, qw), dtype=self.torch.bfloat16,
                             device="cuda")
        k = self.torch.empty((n, kw), dtype=self.torch.bfloat16,
                             device="cuda")
        self._triton_kernels()["qk"][(n,)](
            qkv, q, k, self.rotary.cos_sin_cache, positions,
            attn.q_norm.weight, attn.k_norm.weight,
            qkv.stride(0), q.stride(0), k.stride(0),
            attn.q_norm.variance_epsilon,
            QH=self.num_q_heads, KH=self.num_kv_heads,
            HD=self.head_dim, HALF=self.head_dim // 2)
        return q, k

    # ---- attention: two calls, one merge ----------------------------

    def _fa(self, q, k, v, cu_q, cu_k, max_q, max_k, causal,
            block_table=None, seqused_k=None):
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        return flash_attn_varlen_func(
            q, k, v, max_seqlen_q=max_q, cu_seqlens_q=cu_q,
            max_seqlen_k=max_k, cu_seqlens_k=cu_k,
            block_table=block_table, seqused_k=seqused_k,
            causal=causal, fa_version=3, return_softmax_lse=True)

    def _merge_lse(self, out_a, lse_a, out_b, lse_b, rows, n):
        """The softmax-state merge of the two attention calls, fused
        into one kernel: out_a[rows] = lerp(out_a[rows], out_b,
        sigmoid(lse_b - lse_a[rows])).

        Replaces ~9 launches per layer (two LSE transposes, three
        index_selects, sub, sigmoid, cast, lerp, index_copy_) and two
        extra passes over the merged rows; the LSE tensors are read in
        their native layout through strides, so the transposing copies
        are gone too."""
        n_suf = rows.shape[0]
        H, D = self.num_q_heads, self.head_dim

        def ht_strides(lse, n_tok):
            # the varlen wrapper returns (heads, tokens); accept
            # (tokens, heads) without a transposing copy
            if lse.shape[0] == n_tok:
                return lse.stride(1), lse.stride(0)
            return lse.stride(0), lse.stride(1)

        sah, sat = ht_strides(lse_a, n)
        sbh, sbt = ht_strides(lse_b, n_suf)
        self._triton_kernels()["merge"][(n_suf, H)](
            rows, lse_a, lse_b, out_a, out_b, sah, sat, sbh, sbt,
            H=H, D=D)
        return out_a

    def attention(self, q, k, v, meta):
        """meta carries the chunk layout; see pack_chunk in loop.py."""
        torch = self.torch
        n = q.shape[0]
        H, KH, D = self.num_q_heads, self.num_kv_heads, self.head_dim
        q3 = q.view(n, H, D)
        k3 = k.view(n, KH, D)
        v3 = v.contiguous().view(n, KH, D)
        layer = meta["layer"]

        # fresh KV into the arena's pages, before call B reads them:
        # one gather + one scatter per layer for the whole chunk
        if meta["kv_src"] is not None:
            src, dst = meta["kv_src"], meta["kv_dst"]
            self.arena.k[layer].index_copy_(0, dst,
                                            k3.index_select(0, src))
            self.arena.v[layer].index_copy_(0, dst,
                                            v3.index_select(0, src))

        out_a, lse_a = self._fa(
            q3, k3, v3, meta["cu_a"], meta["cu_a"],
            meta["max_a"], meta["max_a"], causal=True)

        cross = meta["cross"]
        if cross is None:
            # no kept-context reads in this chunk: whole-pair segments
            # (probe reference) or a cache-only pass
            meta["layer"] += 1
            return out_a.view(n, H * D)

        rows = cross["rows"]
        q_suf = q3.index_select(0, rows)
        if meta.get("paged", True):
            kp, vp = self.arena.paged_kv(layer)
            out_b, lse_b = self._fa(
                q_suf, kp, vp, cross["cu_q"], None,
                cross["max_q"], cross["max_used"], causal=False,
                block_table=cross["table"], seqused_k=cross["used"])
        else:
            # gather fallback: contiguous copies of each group's kept
            # context, concatenated in group order
            ks, vs = [], []
            for key in cross["keys"]:
                kg, vg = self.arena.gather(layer, key)
                ks.append(kg)
                vs.append(vg)
            kx = ks[0] if len(ks) == 1 else torch.cat(ks)
            vx = vs[0] if len(vs) == 1 else torch.cat(vs)
            out_b, lse_b = self._fa(
                q_suf, kx, vx, cross["cu_q"], cross["cu_k"],
                cross["max_q"], cross["max_used"], causal=False)

        # (wa*A + wb*B)/(wa+wb) == A + (B-A)*sigmoid(lse_b - lse_a):
        # one fused kernel instead of the elementwise chain
        out = self._merge_lse(out_a, lse_a, out_b, lse_b, rows, n)
        meta["layer"] += 1
        return out.view(n, H * D)

    # ---- the forward loop -------------------------------------------

    def forward_chunk(self, chunk):
        meta = chunk["meta"]
        meta["layer"] = 0
        input_ids, positions = chunk["input_ids"], chunk["positions"]
        hidden = self.embed(input_ids)
        residual = None
        for layer in self.layers:
            attn = layer.self_attn
            if residual is None:
                residual = hidden
                q_in, q_scale = self.quant(
                    self.rms_norm(hidden, layer.input_layernorm))
            else:
                q_in, q_scale = self.custom_norm_quant(
                    hidden, layer.input_layernorm, residual)
            qkv = self.gemm(q_in, q_scale, attn.qkv_proj)
            q, k = self.custom_qk_norm_rope(qkv, positions, attn)
            v = qkv[:, (self.num_q_heads + self.num_kv_heads)
                    * self.head_dim:]
            attn_out = self.attention(q, k, v, meta)
            o_in, o_scale = self.quant(attn_out)
            hidden = self.gemm(o_in, o_scale, attn.o_proj)
            g_in, g_scale = self.custom_norm_quant(
                hidden, layer.post_attention_layernorm, residual)
            gate_up = self.gemm(g_in, g_scale, layer.mlp.gate_up_proj)
            d_in, d_scale = self.custom_silu_quant(gate_up)
            hidden = self.gemm(d_in, d_scale, layer.mlp.down_proj)
        final = chunk["final_indices"]
        last_hidden = hidden.index_select(0, final)
        last_residual = residual.index_select(0, final)
        normed, _ = self.fused_add_rms_norm(
            last_hidden, last_residual, self.final_norm)
        return normed
