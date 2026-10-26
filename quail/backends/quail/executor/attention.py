"""The forward-pass engine: kernels, quantization, and attention over the arena.

DeepGEMM matmuls, fused Triton kernels, FlashAttention varlen
self-attention, paged cross-attention against the arena, and
softmax-state merge.

Two attention paths: tree (two attention calls merged with a fused
kernel) and unified (one causal paged attention call after scattering
current KV into the arena). A chunk with no arena pages skips the
arena and runs one causal varlen call per group.

Engine(kernels="vllm") swaps the fused Triton kernels for vLLM's
unfused equivalents. torch is imported lazily.

A forward loop, in models/<arch>.py, calls four primitives: norm_quant,
qk_norm_rope, attention, and activation_quant. Each picks the kernel
set and the precision path itself; the attention path comes with the
Chunk that pack_chunk built. A loop whose layers differ in head
geometry calls attention_unified directly with (rows, heads, dim)
tensors, its own softmax scale, and its sliding window.

The fused qk_norm_rope kernel assumes per-head Q and K RMSNorm before
rotary, as Qwen3 and Gemma 4 have; qkv_norm_rope_heads is its form
for any head geometry, with Gemma's weightless per-head V norm.
"""

GROUP = 128            # fp8 quant group size, matches the engine
# FlashAttention 3's widest head; wider heads run FlashAttention 4,
# whose Hopper build takes heads up to 512, over arena pages.
FA_MAX_HEAD_DIM = 256

# The two attention paths a chunk can run. "unified" is one causal
# paged call in which every row reads the KV its own document needs.
# "tree" is two calls: a causal one over the chunk's rows and one in
# which the readers of one resident KV (a join anchor, a shared
# prefix) read it stacked, merged by the fused merge+quant kernel.
# Only pipelines that set tree_attention run the tree path.
ATTENTION_PATHS = ("unified", "tree")


def flash_attention_version(capability: tuple[int, int]) -> int:
    """Select the attention implementation for a supported CUDA architecture.

    FlashAttention 3 is Hopper 9.x only. FlashAttention 4 covers 9.x,
    10.x, and 11.x. FlashAttention 2 covers 8.0 and up. L40S (8.9)
    uses 2; B200 (10.x) uses 4; H100 stays on 3; RTX PRO 6000 (12.0)
    stays on 2.
    """
    major, _minor = capability
    if capability == (8, 9):
        return 2
    if capability == (9, 0):
        return 3
    if major == 10:
        return 4
    if capability == (12, 0):
        return 2
    raise ValueError(f"Quail does not support CUDA capability {capability}")


def deep_gemm_supported(capability: tuple[int, int]) -> bool:
    """Whether vLLM 0.29.0 enables DeepGEMM on this CUDA capability.

    Matches CudaPlatform.support_deep_gemm: 9.0, 10.x, and 12.x.
    """
    major, _minor = capability
    return capability == (9, 0) or major in (10, 12)


def merge_partial(out, lse, index, out_c, lse_c) -> None:
    """Merge attention over two disjoint key sets into the selected rows.

    Updates out and lse in place using log-sum-exp weights.

    Args:
        out: Existing output tensor with shape (rows, heads, dimension).
        lse: Existing log-sum-exp tensor with shape (heads, rows).
        index: Indices of the output rows to update.
        out_c: Partial output for the same queries over another key set.
        lse_c: Log-sum-exp values corresponding to out_c.
    """
    lse_b = lse[:, index]
    top = lse_b.maximum(lse_c)
    w_b = (lse_b - top).exp()
    w_c = (lse_c - top).exp()
    total = w_b + w_c
    weight_b = (w_b / total).transpose(0, 1).unsqueeze(-1)
    weight_c = (w_c / total).transpose(0, 1).unsqueeze(-1)
    merged = (out.index_select(0, index).float() * weight_b
              + out_c.float() * weight_c)
    out.index_copy_(0, index, merged.to(out.dtype))
    lse[:, index] = top + total.log()


class Engine:
    """Kernels, quantization, and shared-prefix attention over the paged arena.

    Args:
        arena: KVArena the attention paths read and write.
        num_q_heads: Query heads.
        num_kv_heads: KV heads.
        head_dim: Head dimension.
        rotary: The loaded model's rotary embedding module.
        fp8: Whether the linear weights are fp8 block-quantized.
        kernels: "quail" for the fused Triton kernels, "vllm" for
            vLLM's unfused equivalents.
    """

    def __init__(self, arena, *, num_q_heads, num_kv_heads, head_dim, rotary, fp8,
                 kernels="quail"):
        import torch
        from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

        if kernels not in ("quail", "vllm"):
            raise ValueError(f"kernels must be 'quail' or 'vllm', "
                             f"got {kernels!r}")
        self.kernels = kernels

        capability = torch.cuda.get_device_capability()
        self.fa_version = flash_attention_version(capability)
        self.use_deep_gemm = deep_gemm_supported(capability)
        self.torch = torch
        self.arena = arena
        self.rotary = rotary
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_ue8m0 = bool(is_deep_gemm_e8m0_used())
        self.fp8 = torch.float8_e4m3fn
        self.is_fp8 = bool(fp8)

    # ---- weights and quant ------------------------------------------

    @staticmethod
    def weight_scale(linear):
        for name in ("weight_scale", "weight_scale_inv"):
            scale = getattr(linear, name, None)
            if scale is not None:
                return scale
        raise AttributeError(f"no weight scale on {type(linear).__name__}")

    def gemm(self, q_input, input_scale, linear):
        if not self.is_fp8:
            return self.torch.nn.functional.linear(q_input, linear.weight)
        weight_scale = self.weight_scale(linear)
        if self.use_deep_gemm:
            from vllm.utils.deep_gemm import fp8_gemm_nt
            out = self.torch.empty(
                (q_input.shape[0], linear.weight.shape[0]),
                dtype=self.torch.bfloat16, device=q_input.device)
            fp8_gemm_nt((q_input, input_scale),
                        (linear.weight, weight_scale),
                        out, is_deep_gemm_e8m0_used=self.use_ue8m0)
            return out
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            w8a8_triton_block_scaled_mm,
        )
        return w8a8_triton_block_scaled_mm(
            q_input, linear.weight,
            self._triton_activation_scales(q_input, input_scale),
            weight_scale, [GROUP, GROUP],
            output_dtype=self.torch.bfloat16)

    def _triton_activation_scales(self, q_input, input_scale):
        """Return activation scales in the (tokens, groups) layout Triton wants.

        Engine.quant writes column-major scales whose view is already
        (tokens, K/128). A (groups, tokens) tensor is transposed.
        """
        groups = q_input.shape[-1] // GROUP
        tokens = q_input.shape[0]
        if input_scale.shape == (tokens, groups):
            return input_scale
        if input_scale.shape == (groups, tokens):
            return input_scale.transpose(0, 1)
        raise ValueError(
            f"activation scales {tuple(input_scale.shape)} do not match "
            f"tokens={tokens} groups={groups}")

    def quant(self, x):
        if not self.is_fp8:
            return x, None
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
        )
        return per_token_group_quant_fp8(
            x, group_size=GROUP, column_major_scales=True,
            use_ue8m0=self.use_ue8m0)

    def _col_major_scales(self, num_tokens, width):
        return self.torch.empty(
            (width // GROUP, num_tokens),
            dtype=self.torch.float32, device="cuda").permute(-1, -2)

    def rms_norm(self, x, norm):
        return self.norm_rows(x, norm.weight, norm.variance_epsilon)

    def norm_rows(self, x, weight, eps):
        """RMS-normalize the last dimension of x with vLLM's CUDA kernel.

        x may have any rank; weight is a vector over the last
        dimension. Bypasses the module's own dispatch, which runs the
        unfused PyTorch path on this build.
        """
        from vllm import _custom_ops as ops
        width = x.shape[-1]
        rows = x.reshape(-1, width)
        if not rows.is_contiguous():
            rows = rows.contiguous()
        out = self.torch.empty_like(rows)
        ops.rms_norm(out, rows, weight, eps)
        return out.view(x.shape)

    def fused_add_rms_norm(self, hidden, residual, norm):
        from vllm import _custom_ops as ops
        ops.fused_add_rms_norm(hidden, residual, norm.weight,
                               norm.variance_epsilon)
        return hidden, residual

    def norm_quant_rows(self, x, weight, eps):
        """RMS-normalize rows and quantize them to fp8 with per-row scales.

        One vLLM kernel. Returns the fp8 rows and their float32 scales,
        shaped (rows, 1).
        """
        from vllm import _custom_ops as ops
        return ops.rms_norm_dynamic_per_token_quant(
            x, weight, eps, self.torch.float8_e4m3fn)

    def fp8_linear(self, module, x_q, x_s):
        """One of vLLM's fp8 linears on rows already quantized per row.

        module is a vLLM linear whose weight is fp8 with per-output-
        channel scales; its own input quantization is skipped.
        """
        from vllm import _custom_ops as ops
        return ops.cutlass_scaled_mm(
            x_q, module.weight, scale_a=x_s, scale_b=module.weight_scale,
            out_dtype=self.torch.bfloat16, bias=getattr(module, "bias", None))

    def qkv_norm_rope_heads(
        self,
        qkv,
        positions,
        *,
        num_q_heads,
        num_kv_heads,
        head_dim,
        q_weight,
        k_weight,
        eps,
        cos_sin_cache,
    ):
        """The fused per-head q and k RMS norm plus neox rotary, any geometry.

        custom_qk_norm_rope with the layer's own head counts, head
        width, norm weights, and rotary cache instead of the engine's,
        plus the weightless per-head norm of v. Returns contiguous q
        (rows, num_q_heads * head_dim), k and v (rows, num_kv_heads * head_dim), so
        the v slice is never copied on its own.
        """
        n = qkv.shape[0]
        torch = self.torch
        q = torch.empty((n, num_q_heads * head_dim), dtype=torch.bfloat16,
                        device=qkv.device)
        k = torch.empty((n, num_kv_heads * head_dim), dtype=torch.bfloat16,
                        device=qkv.device)
        v = torch.empty((n, num_kv_heads * head_dim), dtype=torch.bfloat16,
                        device=qkv.device)
        self._triton_kernels()["qkv"][(n,)](
            qkv, q, k, v, cos_sin_cache, positions, q_weight, k_weight,
            qkv.stride(0), q.stride(0), k.stride(0), v.stride(0), eps,
            QH=num_q_heads, KH=num_kv_heads, HD=head_dim, HALF=head_dim // 2,
            num_warps=8 if head_dim > 256 else 4)
        return q, k, v

    def gelu_mul_quant(self, gate_up):
        """gelu_tanh(gate) * up, quantized to fp8 with per-row scales."""
        return self._gelu_mul_quant(gate_up, match_vllm_rounding=False)

    def gelu_mul_quant_vllm(self, gate_up):
        """Quantize GELU output after matching vLLM's BF16 rounding."""
        return self._gelu_mul_quant(gate_up, match_vllm_rounding=True)

    def _gelu_mul_quant(self, gate_up, *, match_vllm_rounding):
        n, doubled = gate_up.shape
        half = doubled // 2
        q = self.torch.empty((n, half), dtype=self.torch.float8_e4m3fn,
                             device=gate_up.device)
        scales = self.torch.empty((n, 1), dtype=self.torch.float32,
                                  device=gate_up.device)
        self._triton_kernels()["gelu_quant"][(n,)](
            gate_up, q, scales, gate_up.stride(0), q.stride(0),
            HALF=half, BLOCK=1 << (half - 1).bit_length(),
            match_vllm_rounding=match_vllm_rounding, num_warps=8)
        return q, scales

    def scale_add_norm_quant(self, x, residual, scale, weight, eps):
        """Residual = residual * scale + x in place; the row norm, quantized.

        norm_quant_rows with the residual scaled first. Returns the
        fp8 rows and their float32 scales, shaped (rows, 1).
        """
        n, width = x.shape
        q = self.torch.empty((n, width), dtype=self.torch.float8_e4m3fn,
                             device=x.device)
        scales = self.torch.empty((n, 1), dtype=self.torch.float32,
                                  device=x.device)
        self._triton_kernels()["scale_norm_quant"][(n,)](
            x, residual, weight, q, scales, x.stride(0), residual.stride(0),
            q.stride(0), float(scale), eps, H=width,
            BLOCK=1 << (width - 1).bit_length(), num_warps=8)
        return q, scales

    def norm_rows2(self, x, weight1, weight2, eps):
        """Two RMS norms of the same rows from one read."""
        first, _, second = self._norm_rows2(x, weight1, weight2, eps, False)
        return first, second

    def norm_router_quant(self, x, weight, router_weight, eps):
        """Return FP8 expert input, per-row scales, and BF16 router input."""
        return self._norm_rows2(x, weight, router_weight, eps, True)

    def _norm_rows2(self, x, weight1, weight2, eps, output_fp8):
        n, width = x.shape
        dtype = self.torch.float8_e4m3fn if output_fp8 else x.dtype
        first = self.torch.empty_like(x, dtype=dtype)
        second = self.torch.empty_like(x)
        scales = (self.torch.empty((n, 1), dtype=self.torch.float32,
                                   device=x.device) if output_fp8 else None)
        self._triton_kernels()["norm2"][(n,)](
            x, weight1, weight2, first, second, scales, x.stride(0),
            first.stride(0), second.stride(0), eps, H=width,
            BLOCK=1 << (width - 1).bit_length(), output_fp8=output_fp8,
            num_warps=8)
        return first, scales, second

    def gemma_add_rms_norm(self, x, residual, weight, eps):
        """Add the residual, then RMS-normalize with a (1 + weight) scale.

        Computes in float32 as vLLM's Gemma-style norm does: the sum of
        x and residual is kept as the new residual, rounded to the
        input dtype, and the normalized rows come from the float32 sum.
        With no residual, x is the residual.

        Args:
            x: (rows, width) block output with unit inner stride.
            residual: (rows, width) running residual, or None.
            weight: (width,) raw norm weight; the scale is 1 + weight.
            eps: Variance epsilon.

        Returns:
            (normed, residual): new tensors, or x itself as the
            residual when there was none.
        """
        n, width = x.shape
        has_residual = residual is not None
        out = self.torch.empty_like(x)
        new_residual = self.torch.empty_like(x) if has_residual else x
        self._gemma_norm_kernel()[(n,)](
            x, residual if has_residual else x, weight, out, new_residual,
            x.stride(0), residual.stride(0) if has_residual else 0,
            out.stride(0), new_residual.stride(0), eps, H=width,
            BLOCK=1 << (width - 1).bit_length(),
            HAS_RESIDUAL=has_residual, num_warps=8)
        return out, new_residual

    def _gemma_norm_kernel(self):
        if hasattr(self, "_gemma_kernel"):
            return self._gemma_kernel
        import triton
        import triton.language as tl

        @triton.jit
        def gemma_add_rms_norm(x_ptr, res_ptr, w_ptr, out_ptr, res_out_ptr,
                               stride_x, stride_res, stride_out,
                               stride_res_out, eps, H: tl.constexpr,
                               BLOCK: tl.constexpr,
                               HAS_RESIDUAL: tl.constexpr):
            row = tl.program_id(0).to(tl.int64)
            offs = tl.arange(0, BLOCK)
            mask = offs < H
            total = tl.load(x_ptr + row * stride_x + offs, mask=mask,
                            other=0.0).to(tl.float32)
            if HAS_RESIDUAL:
                total += tl.load(res_ptr + row * stride_res + offs,
                                 mask=mask, other=0.0).to(tl.float32)
                tl.store(res_out_ptr + row * stride_res_out + offs,
                         total.to(res_out_ptr.dtype.element_ty), mask=mask)
            variance = tl.sum(total * total, axis=0) / H
            scale = 1.0 + tl.load(w_ptr + offs, mask=mask,
                                  other=0.0).to(tl.float32)
            normed = total * tl.math.rsqrt(variance + eps) * scale
            tl.store(out_ptr + row * stride_out + offs,
                     normed.to(out_ptr.dtype.element_ty), mask=mask)

        self._gemma_kernel = gemma_add_rms_norm
        return self._gemma_kernel

    # ---- the Triton fused kernels -----------------------------------
    # Each fuses a sequence of vLLM ops into one kernel launch.

    def _triton_kernels(self):
        if hasattr(self, "_kernels"):
            return self._kernels
        import triton
        import triton.language as tl
        from triton.language.extra.cuda import libdevice

        # fuses vLLM's silu_and_mul + per_token_group_quant_fp8
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
            group_index = block * GPB + tl.arange(0, GPB)
            tl.store(s_ptr + group_index * s_stride_g + t * s_stride_t, scale)

        # fuses vLLM's fused_add_rms_norm + per_token_group_quant_fp8
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
            group_index = tl.arange(0, BLOCK // GROUP_C)
            tl.store(s_ptr + group_index * s_stride_g + t * s_stride_t, scale,
                     mask=group_index < NG)

        # fuses vLLM's two per-head rms_norm calls + rotary_emb
        # (five launches) into one
        @triton.jit
        def qk_norm_rope(qkv_ptr, q_out_ptr, k_out_ptr, cs_ptr, position_ptr,
                         qw_ptr, kw_ptr, stride_qkv, stride_qo, stride_ko,
                         eps, QH: tl.constexpr, KH: tl.constexpr,
                         HD: tl.constexpr, HALF: tl.constexpr):
            t = tl.program_id(0)
            position = tl.load(position_ptr + t)
            half_offs = tl.arange(0, HALF)
            cos = tl.load(cs_ptr + position * HD + half_offs).to(tl.float32)
            sin = tl.load(cs_ptr + position * HD + HALF + half_offs).to(tl.float32)
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

        # the paged-KV append every paged engine has (vLLM:
        # reshape_and_cache_flash; FlashInfer: append_paged_kv_cache),
        # with a fused source-side gather: only selected rows of the
        # packed chunk are written, from arbitrary positions
        @triton.jit
        def kv_row_scatter(k_src_ptr, v_src_ptr, k_dst_ptr, v_dst_ptr,
                           src_rows_ptr, dst_rows_ptr,
                           ROW: tl.constexpr, ROW_POW2: tl.constexpr):
            i = tl.program_id(0)
            s = tl.load(src_rows_ptr + i)
            d = tl.load(dst_rows_ptr + i)
            offs = tl.arange(0, ROW_POW2)
            mask = (offs < ROW) & (d >= 0)
            k = tl.load(k_src_ptr + s * ROW + offs, mask=mask)
            tl.store(k_dst_ptr + d * ROW + offs, k, mask=mask)
            v = tl.load(v_src_ptr + s * ROW + offs, mask=mask)
            tl.store(v_dst_ptr + d * ROW + offs, v, mask=mask)

        # the online-softmax LSE merge (Milakov & Gimelshein 2018)
        # fused with vLLM's per-group fp8 quantize pattern; with QUANT
        # off it stores the bf16 merge and no scales
        @triton.jit
        def merge_attn_quant(a_ptr, b_ptr, la_ptr, lb_ptr, source_ptr,
                             q_ptr, s_ptr,
                             stride_a_t, stride_a_h, stride_b_t, stride_b_h,
                             stride_la_t, stride_la_h,
                             stride_lb_t, stride_lb_h,
                             stride_q_t, s_stride_g, s_stride_t,
                             D: tl.constexpr, GPB: tl.constexpr,
                             UE8M0: tl.constexpr, QUANT: tl.constexpr):
            t = tl.program_id(0)
            block = tl.program_id(1)
            offs = tl.arange(0, D * GPB)
            h0 = block * GPB
            a = tl.load(a_ptr + t * stride_a_t
                        + h0 * stride_a_h + offs)
            source = tl.load(source_ptr + t)
            has_b = source >= 0
            source_safe = tl.maximum(source, 0)
            b = tl.load(b_ptr + source_safe * stride_b_t
                        + h0 * stride_b_h + offs,
                        mask=has_b, other=0.0)
            heads = h0 + tl.arange(0, GPB)
            la = tl.load(la_ptr + t * stride_la_t
                         + heads * stride_la_h,
                         mask=has_b, other=0.0)
            lb = tl.load(lb_ptr + source_safe * stride_lb_t
                         + heads * stride_lb_h,
                         mask=has_b, other=0.0)
            w = tl.sigmoid(lb - la).to(tl.bfloat16).to(tl.float32)
            a2 = tl.reshape(a, (GPB, D)).to(tl.float32)
            b2 = tl.reshape(b, (GPB, D)).to(tl.float32)
            merged = a2 + (b2 - a2) * w[:, None]
            merged = merged.to(tl.bfloat16).to(tl.float32)
            y = tl.where(has_b, merged, a2)
            if QUANT:
                amax = tl.max(tl.abs(y), axis=1)
                scale = tl.maximum(amax, 1e-10) / 448.0
                if UE8M0:
                    scale = tl.math.exp2(tl.ceil(tl.math.log2(scale)))
                q = y / scale[:, None]
                q = tl.minimum(tl.maximum(q, -448.0), 448.0)
                tl.store(q_ptr + t * stride_q_t + h0 * D + offs,
                         tl.reshape(q, (D * GPB,)).to(q_ptr.dtype.element_ty))
                tl.store(s_ptr + heads * s_stride_g + t * s_stride_t, scale)
            else:
                tl.store(q_ptr + t * stride_q_t + h0 * D + offs,
                         tl.reshape(y, (D * GPB,)).to(q_ptr.dtype.element_ty))

        # merges attention over a second key set into chosen rows of a
        # first, in place, by log-sum-exp weights: merge_partial in one
        # pass over the rows
        @triton.jit
        def merge_rows(out_ptr, lse_ptr, index_ptr, c_ptr, lc_ptr,
                       stride_o_t, stride_o_h, stride_l_h, stride_l_t,
                       stride_c_t, stride_c_h, stride_lc_h, stride_lc_t,
                       D: tl.constexpr, GPB: tl.constexpr):
            i = tl.program_id(0)
            block = tl.program_id(1)
            offs = tl.arange(0, D * GPB)
            h0 = block * GPB
            heads = h0 + tl.arange(0, GPB)
            j = tl.load(index_ptr + i)
            row = out_ptr + j * stride_o_t + h0 * stride_o_h + offs
            o = tl.reshape(tl.load(row), (GPB, D)).to(tl.float32)
            c = tl.reshape(tl.load(c_ptr + i * stride_c_t + h0 * stride_c_h
                                   + offs), (GPB, D)).to(tl.float32)
            lo = tl.load(lse_ptr + heads * stride_l_h + j * stride_l_t)
            lc = tl.load(lc_ptr + heads * stride_lc_h + i * stride_lc_t)
            top = tl.maximum(lo, lc)
            wo = tl.exp(lo - top)
            wc = tl.exp(lc - top)
            total = wo + wc
            merged = (o * wo[:, None] + c * wc[:, None]) / total[:, None]
            tl.store(row, tl.reshape(merged, (D * GPB,)).to(
                out_ptr.dtype.element_ty))
            tl.store(lse_ptr + heads * stride_l_h + j * stride_l_t,
                     top + tl.log(total))

        # gelu_tanh_and_mul fused with the per-row fp8 quantization the
        # next linear would run on its output
        @triton.jit
        def gelu_mul_quant(gu_ptr, q_ptr, s_ptr, stride_gu, stride_q,
                           HALF: tl.constexpr, BLOCK: tl.constexpr,
                           match_vllm_rounding: tl.constexpr):
            t = tl.program_id(0)
            offs = tl.arange(0, BLOCK)
            mask = offs < HALF
            gate = tl.load(gu_ptr + t * stride_gu + offs, mask=mask,
                           other=0.0).to(tl.float32)
            up = tl.load(gu_ptr + t * stride_gu + HALF + offs, mask=mask,
                         other=0.0).to(tl.float32)
            if match_vllm_rounding:
                # vLLM rounds GELU to BF16 before the gating multiplication.
                cube = gate * gate * gate
                inner = 0.7978845608028654 * (gate + 0.044715 * cube)
                activated = (0.5 * gate * (1.0 + libdevice.tanh(inner)))
                y = activated.to(tl.bfloat16).to(tl.float32) * up
            else:
                inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
                th = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
                y = 0.5 * gate * (1.0 + th) * up
            # the unfused path rounds the activation to bf16 first
            y = y.to(tl.bfloat16).to(tl.float32)
            amax = tl.max(tl.abs(y), axis=0)
            if match_vllm_rounding:
                scale = tl.maximum(tl.div_rn(amax, 448.0),
                                   1.0 / (448.0 * 512.0))
                q = tl.div_rn(y, scale)
            else:
                scale = tl.maximum(amax / 448.0, 1.0 / (448.0 * 512.0))
                q = y / scale
            q = tl.minimum(tl.maximum(q, -448.0), 448.0)
            tl.store(q_ptr + t * stride_q + offs,
                     q.to(q_ptr.dtype.element_ty), mask=mask)
            tl.store(s_ptr + t, scale)

        # vLLM's rms_norm_dynamic_per_token_quant with a residual,
        # with the residual scaled first: residual = residual * scale
        # + x, then the row norm quantized per row
        @triton.jit
        def scale_add_rms_norm_quant(x_ptr, res_ptr, w_ptr, q_ptr, s_ptr,
                                     stride_x, stride_res, stride_q,
                                     scale, eps, H: tl.constexpr,
                                     BLOCK: tl.constexpr):
            t = tl.program_id(0)
            offs = tl.arange(0, BLOCK)
            mask = offs < H
            x = tl.load(x_ptr + t * stride_x + offs, mask=mask,
                        other=0.0).to(tl.float32)
            r = tl.load(res_ptr + t * stride_res + offs, mask=mask,
                        other=0.0).to(tl.float32)
            r = r * scale + x
            tl.store(res_ptr + t * stride_res + offs,
                     r.to(res_ptr.dtype.element_ty), mask=mask)
            ms = tl.sum(r * r, axis=0) / H
            rstd = 1.0 / tl.sqrt(ms + eps)
            w = tl.load(w_ptr + offs, mask=mask, other=0.0)
            y = ((r * rstd).to(tl.bfloat16) * w.to(tl.bfloat16)).to(tl.float32)
            amax = tl.max(tl.abs(y), axis=0)
            qscale = tl.maximum(amax / 448.0, 1.0 / (448.0 * 512.0))
            q = tl.minimum(tl.maximum(y / qscale, -448.0), 448.0)
            tl.store(q_ptr + t * stride_q + offs,
                     q.to(q_ptr.dtype.element_ty), mask=mask)
            tl.store(s_ptr + t, qscale)

        # one read of a row for two RMS norms with different weights
        @triton.jit
        def rms_norm2(x_ptr, w1_ptr, w2_ptr, o1_ptr, o2_ptr, s_ptr, stride_x,
                      stride_o1, stride_o2, eps, H: tl.constexpr,
                      BLOCK: tl.constexpr, output_fp8: tl.constexpr):
            t = tl.program_id(0)
            # Keep the BF16 reduction order when one output becomes FP8.
            offs = tl.max_contiguous(tl.arange(0, BLOCK), 8)
            mask = offs < H
            x = tl.load(x_ptr + t * stride_x + offs, mask=mask,
                        other=0.0).to(tl.float32)
            ms = tl.sum(x * x, axis=0) / H
            xn = x * (1.0 / tl.sqrt(ms + eps))
            w1 = tl.load(w1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            w2 = tl.load(w2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            first = (xn * w1).to(x_ptr.dtype.element_ty)
            if output_fp8:
                # Preserve the BF16 intermediate's rounding before quantizing.
                first = first.to(tl.float32)
                scale = tl.maximum(tl.div_rn(tl.max(tl.abs(first), axis=0),
                                             448.0),
                                   1.0 / (448.0 * 512.0))
                first = tl.minimum(tl.maximum(tl.div_rn(first, scale), -448.0),
                                   448.0)
                tl.store(s_ptr + t, scale)
            tl.store(o1_ptr + t * stride_o1 + offs,
                     first.to(o1_ptr.dtype.element_ty), mask=mask)
            tl.store(o2_ptr + t * stride_o2 + offs,
                     (xn * w2).to(o2_ptr.dtype.element_ty), mask=mask)

        # qk_norm_rope plus the weightless per-head v norm Gemma
        # applies, written contiguous for the KV scatter
        @triton.jit
        def qkv_norm_rope(qkv_ptr, q_out_ptr, k_out_ptr, v_out_ptr, cs_ptr,
                          position_ptr, qw_ptr, kw_ptr, stride_qkv, stride_qo,
                          stride_ko, stride_vo, eps, QH: tl.constexpr,
                          KH: tl.constexpr, HD: tl.constexpr,
                          HALF: tl.constexpr):
            qk_norm_rope(qkv_ptr, q_out_ptr, k_out_ptr, cs_ptr, position_ptr,
                         qw_ptr, kw_ptr, stride_qkv, stride_qo, stride_ko,
                         eps, QH, KH, HD, HALF)
            t = tl.program_id(0)
            half_offs = tl.arange(0, HALF)
            k_heads = tl.arange(0, KH)
            ka_offs = k_heads[:, None] * HD + half_offs[None, :]
            kb_offs = ka_offs + HALF
            base = qkv_ptr + t * stride_qkv + (QH + KH) * HD
            va = tl.load(base + ka_offs).to(tl.float32)
            vb = tl.load(base + kb_offs).to(tl.float32)
            ms = (tl.sum(va * va, axis=1) + tl.sum(vb * vb, axis=1)) / HD
            rstd = 1.0 / tl.sqrt(ms + eps)
            tl.store(v_out_ptr + t * stride_vo + ka_offs,
                     (va * rstd[:, None]).to(v_out_ptr.dtype.element_ty))
            tl.store(v_out_ptr + t * stride_vo + kb_offs,
                     (vb * rstd[:, None]).to(v_out_ptr.dtype.element_ty))

        self._kernels = {"silu": silu_mul_quant,
                         "norm": add_rms_norm_quant,
                         "qk": qk_norm_rope,
                         "qkv": qkv_norm_rope,
                         "gelu_quant": gelu_mul_quant,
                         "scale_norm_quant": scale_add_rms_norm_quant,
                         "norm2": rms_norm2,
                         "kv_scatter": kv_row_scatter,
                         "merge_quant": merge_attn_quant,
                         "merge_rows": merge_rows}
        return self._kernels

    def custom_silu_quant(self, gate_up):
        if not self.is_fp8:
            return self.vllm_silu_quant(gate_up)
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
        if not self.is_fp8:
            return self.vllm_norm_quant(hidden, norm, residual)
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

    # ---- the vLLM-kernel path (kernels="vllm") ----------------------
    # The same ops the engine's compiled graph runs, called eagerly.

    def vllm_norm_quant(self, hidden, norm, residual):
        """fused-add rms_norm, then a separate quantize.

        Two kernels where custom_norm_quant is one.
        """
        normed, residual = self.fused_add_rms_norm(hidden, residual,
                                                   norm)
        return self.quant(normed)

    def vllm_silu_quant(self, gate_up):
        """silu_and_mul, then a separate quantize."""
        out = self.torch.empty(
            (gate_up.shape[0], gate_up.shape[1] // 2),
            dtype=gate_up.dtype, device=gate_up.device)
        self.torch.ops._C.silu_and_mul(out, gate_up)
        return self.quant(out)

    def vllm_qk_norm_rope(self, qkv, positions, attn):
        """Per-head q/k norms plus rotary, five kernels in total.

        Two contiguous copies, two rms_norm calls, one rotary.
        """
        n = qkv.shape[0]
        qw = self.num_q_heads * self.head_dim
        kw = self.num_kv_heads * self.head_dim
        q, k, _ = qkv.split([qw, kw, kw], dim=-1)
        q = self.rms_norm(q.reshape(-1, self.head_dim).contiguous(),
                          attn.q_norm).reshape(n, qw)
        k = self.rms_norm(k.reshape(-1, self.head_dim).contiguous(),
                          attn.k_norm).reshape(n, kw)
        return self.rotary(positions, q, k)

    def kv_row_scatter(self, k3, v3, src, dst, layer):
        """Scatter fresh KV rows from the packed chunk into arena pages.

        Negative destination rows are padding and perform no reads or writes.
        """
        assert k3.is_contiguous() and v3.is_contiguous()
        n = src.shape[0]
        row = k3.shape[1] * k3.shape[2]
        k_pool, v_pool = self.arena.layer_kv(layer)
        self._triton_kernels()["kv_scatter"][(n,)](
            k3, v3, k_pool, v_pool, src, dst,
            ROW=row, ROW_POW2=1 << (row - 1).bit_length())

    def merge_attn_quant(self, out_a, lse_a, out_b, lse_b, source):
        """Merge cached and fresh attention and write the o_proj input.

        The source vector maps each packed row to its row in out_b, or
        contains -1 when out_a is already the complete answer. An fp8
        model gets FP8 rows and their scales; one 128-value FP8 group
        is one attention head. A bf16 model gets bf16 rows and None.
        """
        n, heads, dim = out_a.shape
        gpb = 4
        if heads % gpb:
            raise ValueError(
                f"the merge kernel takes heads in groups of {gpb}, not {heads}")
        if self.is_fp8:
            assert dim == GROUP
            q = self.torch.empty((n, heads * dim), dtype=self.fp8,
                                 device=out_a.device)
            scales = self._col_major_scales(n, heads * dim)
            scale_strides = scales.stride(1), scales.stride(0)
        else:
            q = self.torch.empty((n, heads * dim), dtype=self.torch.bfloat16,
                                 device=out_a.device)
            scales = None
            scale_strides = 0, 0
        self._triton_kernels()["merge_quant"][(n, heads // gpb)](
            out_a, out_b, lse_a, lse_b, source, q,
            q if scales is None else scales,
            out_a.stride(0), out_a.stride(1),
            out_b.stride(0), out_b.stride(1),
            lse_a.stride(0), lse_a.stride(1),
            lse_b.stride(0), lse_b.stride(1),
            q.stride(0), *scale_strides,
            D=dim, GPB=gpb, UE8M0=self.use_ue8m0, QUANT=self.is_fp8)
        return q, scales

    def merge_document_read(self, out, lse, index, out_d, lse_d):
        """Merge the document read into call B's rows, in place.

        Computes what merge_partial does, in one kernel.

        Args:
            out: Call B's output, (rows, heads, dim), heads contiguous.
            lse: Call B's log-sum-exp, (heads, rows).
            index: Per document-read row, its row in out.
            out_d: The document read's output, (index rows, heads, dim).
            lse_d: The document read's log-sum-exp, (heads, index rows).
        """
        n, heads, dim = out_d.shape
        gpb = 4
        if heads % gpb or out.stride(1) != dim or out_d.stride(1) != dim:
            raise ValueError("the merge takes contiguous heads in groups of 4")
        if not n:
            return
        self._triton_kernels()["merge_rows"][(n, heads // gpb)](
            out, lse, index, out_d, lse_d,
            out.stride(0), out.stride(1), lse.stride(0), lse.stride(1),
            out_d.stride(0), out_d.stride(1), lse_d.stride(0), lse_d.stride(1),
            D=dim, GPB=gpb)

    # ---- the primitives a forward loop calls -------------------------

    def norm_quant(self, hidden, norm, residual=None):
        """RMS-normed, quantized GEMM input; adds the residual first when given."""
        if residual is None:
            return self.quant(self.rms_norm(hidden, norm))
        if self.kernels == "quail":
            return self.custom_norm_quant(hidden, norm, residual)
        return self.vllm_norm_quant(hidden, norm, residual)

    def activation_quant(self, gate_up):
        """SiLU-gated product of the two gate_up halves, quantized."""
        if self.kernels == "quail":
            return self.custom_silu_quant(gate_up)
        return self.vllm_silu_quant(gate_up)

    def qk_norm_rope(self, qkv, positions, attn):
        """Per-head normed and rotated q and k from the fused qkv output."""
        if self.kernels == "quail":
            return self.custom_qk_norm_rope(qkv, positions, attn)
        return self.vllm_qk_norm_rope(qkv, positions, attn)

    def attention(self, q, k, v, chunk):
        """Attention over the chunk and the arena; the quantized o_proj input.

        The chunk carries the path it was packed for. q, k, and v are
        the flat per-row projections at this engine's head geometry.
        """
        n = q.shape[0]
        H, KH, D = self.num_q_heads, self.num_kv_heads, self.head_dim
        q3 = q.view(n, H, D)
        k3 = k.view(n, KH, D)
        v3 = v.contiguous().view(n, KH, D)
        if chunk.attention_mode == "tree":
            return self.attention_tree(q3, k3, v3, chunk.meta)
        return self.quant(self.attention_unified(q3, k3, v3, chunk.meta))

    # ---- attention: the two workload paths --------------------------


    def _fa(self, q, k, v, cu_q, cu_k, max_q, max_k, causal,
            block_table=None, seqused_k=None, softmax_scale=None,
            window=None, version=None):
        # LSE comes back as [heads, total_queries] for the merge kernel.
        # Blackwell's head-256 FA4 kernel rejects the SplitKV that
        # num_splits=0 can select.
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        version = version or self.fa_version
        extra = {}
        if softmax_scale is not None:
            extra["softmax_scale"] = softmax_scale
        if window is not None:
            extra["window_size"] = list(window)
        shape = getattr(k, "shape", None)
        if version == 4 and shape is not None and shape[-1] == 256:
            extra["num_splits"] = 1
            if block_table is not None:
                # FA4 requires the table capacity; seqused_k masks padding.
                max_k = block_table.shape[1] * k.shape[1]
        return flash_attn_varlen_func(
            q, k, v, max_seqlen_q=max_q, cu_seqlens_q=cu_q,
            max_seqlen_k=max_k, cu_seqlens_k=cu_k,
            block_table=block_table, seqused_k=seqused_k,
            causal=causal, fa_version=version,
            return_softmax_lse=True, **extra)

    def _flashinfer_paged(self, q3, kp, vp, metadata, *, causal,
                          softmax_scale, window):
        """Run paged attention with a FlashInfer plan shared across layers."""
        import numpy as np
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper

        if not hasattr(self, "_flashinfer_plans"):
            self._flashinfer_plans = {}
            # Head-512 split-KV plans exceed the usual 128 MiB workspace.
            self._flashinfer_workspace = self.torch.empty(
                256 * 1024 * 1024, dtype=self.torch.uint8, device=q3.device)
        key = (causal, softmax_scale, window, q3.shape[1:], kp.shape[1:],
               q3.dtype, kp.dtype)
        previous, wrapper, uploaded = self._flashinfer_plans.get(
            key, (None, None, None))
        if wrapper is None:
            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self._flashinfer_workspace, kv_layout="NHD", backend="fa2")
            uploaded = self.torch.cuda.Event()
        if metadata is not previous:
            page_tokens = kp.shape[1]
            used = np.asarray(metadata["used_host"], dtype=np.int32)
            counts = -(-used // page_tokens)
            indptr = np.concatenate(([0], counts.cumsum())).astype(np.int32)
            indices = np.concatenate([
                pages[:count] for pages, count in zip(metadata["page_rows"], counts)
            ]).astype(np.int32)
            options = dict(
                causal=causal, sm_scale=softmax_scale,
                window_left=-1 if window is None else window[0],
                q_data_type=q3.dtype, kv_data_type=kp.dtype)
            # FlashInfer rewrites pinned host memory before its asynchronous upload.
            uploaded.synchronize()
            wrapper.plan(
                self.torch.as_tensor(metadata["cu_q_host"], dtype=self.torch.int32),
                self.torch.from_numpy(indptr), self.torch.from_numpy(indices),
                self.torch.from_numpy((used - 1) % page_tokens + 1),
                q3.shape[1], kp.shape[2], q3.shape[2], page_tokens, **options)
            uploaded.record()
            self._flashinfer_plans[key] = metadata, wrapper, uploaded
        return wrapper.run(q3, (kp, vp))

    def _paged(self, q3, kp, vp, cu_q, max_q, seqused_k, max_seqused_k, block_table, *,
               causal, softmax_scale=None, window=None, metadata=None):
        """One paged attention call.

        FlashAttention 3, or FlashAttention 4 when the head is wider
        than it takes. A 512-wide head on a GPU that selects
        FlashAttention 4 runs through FlashInfer CUDA attention.
        """
        if self.fa_version == 4 and q3.shape[-1] == 512:
            return self._flashinfer_paged(
                q3, kp, vp, metadata, causal=causal,
                softmax_scale=softmax_scale, window=window)
        wide = q3.shape[-1] > FA_MAX_HEAD_DIM
        out, _ = self._fa(
            q3, kp, vp, cu_q, None, max_q, max_seqused_k, causal=causal,
            block_table=block_table, seqused_k=seqused_k,
            softmax_scale=softmax_scale, window=window,
            version=4 if wide else None)
        return out

    def attention_tree(self, q3, k3, v3, meta):
        """The tree attention path: two calls, then one fused merge.

        Takes (rows, heads, dim) tensors and returns the input pair
        for o_proj. Canvas rows are not packed for this path.
        """
        n, H, D = q3.shape
        layer = meta["layer"]
        if meta.get("canvas") is not None:
            raise ValueError("canvas rows run the unified attention path")

        if meta["kv_src"] is not None:
            self.kv_row_scatter(k3, v3, meta["kv_src"], meta["kv_dst"],
                                layer)
        if meta.get("rounds") is not None:
            return self._attention_rounds(q3, k3, v3, meta)

        out_a, lse_a = self._fa(
            q3, k3, v3, meta["cu_a"], meta["cu_a"],
            meta["max_a"], meta["max_a"], causal=True)
        reads = meta["reads"]
        if reads is None:
            meta["layer"] += 1
            return self.quant(out_a.view(n, H * D))

        rows = reads["rows"]
        q_readers = q3.index_select(0, rows)
        kp, vp = self.arena.paged_kv(layer)
        out_b, lse_b = self._fa(
            q_readers, kp, vp, reads["cu_q"], None,
            reads["max_q"], reads["max_used"], causal=False,
            block_table=reads["table"], seqused_k=reads["used"])
        nodes = reads.get("nodes")
        if nodes is not None:
            # call C: path rows over their ancestor rows in earlier
            # paths of the chunk, merged into their call B result
            out_c, lse_c = self._fa(
                q3.index_select(0, nodes["rows"]),
                k3.index_select(0, nodes["key_rows"]),
                v3.index_select(0, nodes["key_rows"]),
                nodes["cu_q"], nodes["cu_k"], nodes["max_q"], nodes["max_k"],
                causal=False)
            merge_partial(out_b, lse_b, nodes["b_index"], out_c, lse_c)
        lse_a = lse_a.transpose(0, 1)
        lse_b = lse_b.transpose(0, 1)
        q_out, scales = self.merge_attn_quant(
            out_a, lse_a, out_b, lse_b, reads["source"])
        meta["layer"] += 1
        return q_out, scales

    def _attention_rounds(self, q3, k3, v3, meta):
        """The tree path for a pass holding map rounds (see Stage.document_reads).

        Each round's KV is in its branch's pages already, so one causal
        paged call reads the branch's own pages with the round's rows,
        in place of calls A and B. A second call reads each document's
        borrowed pages once for the rows of all its rounds, merged in
        by log-sum-exp. Other rows of the pass take calls A and B.
        """
        n, H, D = q3.shape
        layer = meta["layer"]
        rounds = meta["rounds"]
        kp, vp = self.arena.paged_kv(layer)
        rows = rounds["rows"]
        q_rounds = q3.index_select(0, rows)
        out_r, lse_r = self._fa(
            q_rounds, kp, vp, rounds["cu_q"], None, rounds["max_q"],
            rounds["max_used"], causal=True, block_table=rounds["table"],
            seqused_k=rounds["used"])
        documents = rounds.get("documents")
        if documents is not None:
            out_d, lse_d = self._fa(
                q_rounds.index_select(0, documents["b_index"]), kp, vp,
                documents["cu_q"], None, documents["max_q"],
                documents["max_used"], causal=False,
                block_table=documents["table"], seqused_k=documents["used"])
            self.merge_document_read(out_r, lse_r, documents["b_index"],
                                     out_d, lse_d)
        a_rows = meta["a_rows"]
        reads = meta["reads"]
        if a_rows is None and reads is None:
            # every row is a round's, packed in order
            meta["layer"] += 1
            return self.quant(out_r.view(n, H * D))
        out_a = out_r.new_empty((n, H, D))
        lse_a = lse_r.new_empty((H, n))
        if a_rows is not None:
            out_s, lse_s = self._fa(
                q3.index_select(0, a_rows), k3.index_select(0, a_rows),
                v3.index_select(0, a_rows), meta["cu_a"], meta["cu_a"],
                meta["max_a"], meta["max_a"], causal=True)
            out_a.index_copy_(0, a_rows, out_s)
            lse_a[:, a_rows] = lse_s
        out_a.index_copy_(0, rows, out_r)
        if reads is None:
            meta["layer"] += 1
            return self.quant(out_a.view(n, H * D))
        # a round's row has no call B source, so the merge keeps out_a
        q_readers = q3.index_select(0, reads["rows"])
        out_b, lse_b = self._fa(
            q_readers, kp, vp, reads["cu_q"], None,
            reads["max_q"], reads["max_used"], causal=False,
            block_table=reads["table"], seqused_k=reads["used"])
        nodes = reads.get("nodes")
        if nodes is not None:
            out_c, lse_c = self._fa(
                q3.index_select(0, nodes["rows"]),
                k3.index_select(0, nodes["key_rows"]),
                v3.index_select(0, nodes["key_rows"]),
                nodes["cu_q"], nodes["cu_k"], nodes["max_q"], nodes["max_k"],
                causal=False)
            merge_partial(out_b, lse_b, nodes["b_index"], out_c, lse_c)
        q_out, scales = self.merge_attn_quant(
            out_a, lse_a.transpose(0, 1), out_b, lse_b.transpose(0, 1),
            reads["source"])
        meta["layer"] += 1
        return q_out, scales

    def _pool(self, layer, tables):
        """The block tables a layer reads: the sliding pool's on a sliding layer."""
        if tables is None or layer not in self.arena.sliding_layers:
            return tables
        return tables.get("sliding") or tables

    def attention_unified(self, q3, k3, v3, meta, *, softmax_scale=None,
                          window=None):
        """Write current KV to its cache slots, then one causal paged attention call.

        The call reads retained and current KV together. Takes
        (rows, heads, dim) tensors and returns (rows, heads * dim).

        When meta["unified"] is None (no arena pages), falls back to
        a plain varlen causal call with no scatter or paged read.

        Canvas rows (meta["canvas"]) get a second, non-causal call
        over the same KV: each canvas row sees its whole prompt and
        every row of its own canvas. Its result replaces the causal
        call's rows. A one-row canvas is the last row of its causal
        segment, which already sees the whole prompt, so it skips the
        second call.

        Args:
            q3: Queries, (rows, heads, dim).
            k3: Keys, (rows, KV heads, dim), contiguous.
            v3: Values, (rows, KV heads, dim), contiguous.
            meta: The chunk's attention bookkeeping from pack_chunk.
            softmax_scale: Attention logit scale; None is 1/sqrt(dim).
            window: Sliding window in tokens, or None for full
                attention. Causal rows see `window` keys behind them;
                canvas rows see `window` keys on either side.
        """
        n = q3.shape[0]
        layer = meta["layer"]
        unified = meta["unified"]
        canvas = meta.get("canvas")
        if canvas is not None and canvas["max_q"] == 1:
            canvas = None
        behind = None if window is None else (window - 1, 0)
        around = None if window is None else (window - 1, window - 1)
        pool = self._pool(layer, unified)
        canvas_pool = self._pool(layer, canvas)
        if unified is None:
            if q3.shape[-1] > FA_MAX_HEAD_DIM:
                raise ValueError(
                    f"a {q3.shape[-1]}-wide head runs paged attention "
                    f"only; pack the chunk with arena pages")
            out, _ = self._fa(
                q3, k3, v3, meta["cu_a"], meta["cu_a"],
                meta["max_a"], meta["max_a"], causal=True,
                softmax_scale=softmax_scale, window=behind)
            if canvas is not None:
                q_c = q3.index_select(0, canvas["rows"])
                out_c, _ = self._fa(
                    q_c, k3, v3, canvas["cu_q"], meta["cu_a"],
                    canvas["max_q"], meta["max_a"], causal=False,
                    softmax_scale=softmax_scale, window=around)
                out.index_copy_(0, canvas["rows"], out_c)
            meta["layer"] += 1
            return out.view(n, -1)
        if pool["src"].numel():
            self.kv_row_scatter(k3, v3, pool["src"], pool["dst"], layer)
        if pool["tail_src"] is not None:
            k_pool, v_pool = self.arena.layer_kv(layer)
            self.kv_row_scatter(k_pool, v_pool,
                                pool["tail_src"], pool["tail_dst"], layer)
        kp, vp = self.arena.paged_kv(layer)
        out = self._paged(
            q3, kp, vp, unified["cu_q"], unified["max_q"], pool["used"],
            pool["max_used"], pool["table"], causal=True,
            softmax_scale=softmax_scale, window=behind, metadata=pool)
        if canvas is not None:
            q_c = q3.index_select(0, canvas["rows"])
            out_c = self._paged(
                q_c, kp, vp, canvas["cu_q"], canvas["max_q"],
                canvas_pool["used"], canvas_pool["max_used"],
                canvas_pool["table"], causal=False,
                softmax_scale=softmax_scale, window=around, metadata=canvas_pool)
            out.index_copy_(0, canvas["rows"], out_c)
        meta["layer"] += 1
        return out.view(n, -1)
