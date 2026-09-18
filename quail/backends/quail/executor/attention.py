"""The forward-pass engine: kernels, quantization, and attention over the arena.

DeepGEMM matmuls, fused Triton kernels, FlashAttention varlen
self-attention, paged cross-attention against the arena, and
softmax-state merge.

Two attention paths: merge_quant (two attention calls merged with a fused
kernel) and unified (one causal paged attention call after scattering
current KV into the arena). A chunk with no arena pages skips the
arena and runs one causal varlen call per group.

Engine(kernels="vllm") swaps the fused Triton kernels for vLLM's
unfused equivalents. torch is imported lazily.

A forward loop, in models/<arch>.py, calls four primitives: norm_quant,
qk_norm_rope, attention, and activation_quant. Each picks the kernel
set and the precision path itself; the attention path comes with the
Chunk that pack_chunk built.

The fused qk_norm_rope kernel assumes per-head Q and K RMSNorm before
rotary, as Qwen3 has.
"""

from dataclasses import dataclass
from typing import Any

GROUP = 128            # fp8 quant group size, matches the engine

# Filters run "unified" (one causal paged attention call); joins run
# "merge_quant" (two-call pattern with fused merge+quant kernel).
FILTER_ATTENTION = "unified"
JOIN_ATTENTION = "merge_quant"


def join_attention_mode(is_fp8: bool) -> str:
    """The path a join's chunks run.

    merge_quant writes fp8 GEMM inputs, so bf16 weights run the
    unified path instead.
    """
    return JOIN_ATTENTION if is_fp8 else FILTER_ATTENTION


@dataclass
class Chunk:
    """One packed forward pass: token rows plus attention bookkeeping.

    Attributes:
        input_ids: Token ids, one per packed row, on the GPU.
        positions: Rotary position of each row.
        final_indices: Rows whose hidden state feeds the answer readout.
        meta: Attention-path bookkeeping built by pack_chunk: the layer
            counter, KV scatter maps, block tables, and sequence bounds.
        attention_mode: "unified" or "merge_quant", the path the
            packer laid the chunk out for.
        tokens: Rows in the chunk.
        layout: (arena key, suffix count) per group in chunk order.
        temporary_keys: Arena keys the loop frees after the forward pass.
    """

    input_ids: Any
    positions: Any
    final_indices: Any
    meta: dict
    attention_mode: str
    tokens: int
    layout: list
    temporary_keys: tuple = ()


def flash_attention_version(capability: tuple[int, int]) -> int:
    """Select the attention implementation for a supported CUDA architecture."""
    if capability == (9, 0):
        return 3
    if capability == (12, 0):
        return 2
    raise ValueError(f"Quail does not support CUDA capability {capability}")


class Engine:
    """Kernels, quantization, and shared-prefix attention over the paged arena.

    Args:
        arena: KVArena the attention paths read and write.
        n_q: Query heads.
        n_kv: KV heads.
        head_dim: Head dimension.
        rotary: The loaded model's rotary embedding module.
        fp8: Whether the linear weights are fp8 block-quantized.
        kernels: "quail" for the fused Triton kernels, "vllm" for
            vLLM's unfused equivalents.
    """

    def __init__(self, arena, *, n_q, n_kv, head_dim, rotary, fp8,
                 kernels="quail"):
        import torch
        from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

        if kernels not in ("quail", "vllm"):
            raise ValueError(f"kernels must be 'quail' or 'vllm', "
                             f"got {kernels!r}")
        self.kernels = kernels

        self.fa_version = flash_attention_version(torch.cuda.get_device_capability())
        self.torch = torch
        self.arena = arena
        self.rotary = rotary
        self.num_q_heads = n_q
        self.num_kv_heads = n_kv
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
        # all linear projections (QKV, O, gate-up, down) run vLLM's
        # DeepGEMM fp8 matmul; weights and scales are vLLM's layout
        from vllm.utils.deep_gemm import fp8_gemm_nt
        out = self.torch.empty(
            (q_input.shape[0], linear.weight.shape[0]),
            dtype=self.torch.bfloat16, device=q_input.device)
        fp8_gemm_nt((q_input, input_scale),
                    (linear.weight, self.weight_scale(linear)),
                    out, is_deep_gemm_e8m0_used=self.use_ue8m0)
        return out

    def quant(self, x):
        if not self.is_fp8:
            return x, None
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

    # ---- the Triton fused kernels -----------------------------------
    # Each fuses a sequence of vLLM ops into one kernel launch.

    def _triton_kernels(self):
        if hasattr(self, "_kernels"):
            return self._kernels
        import triton
        import triton.language as tl

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
            g_idx = block * GPB + tl.arange(0, GPB)
            tl.store(s_ptr + g_idx * s_stride_g + t * s_stride_t, scale)

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
            g_idx = tl.arange(0, BLOCK // GROUP_C)
            tl.store(s_ptr + g_idx * s_stride_g + t * s_stride_t, scale,
                     mask=g_idx < NG)

        # fuses vLLM's two per-head rms_norm calls + rotary_emb
        # (five launches) into one
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
            mask = offs < ROW
            k = tl.load(k_src_ptr + s * ROW + offs, mask=mask)
            tl.store(k_dst_ptr + d * ROW + offs, k, mask=mask)
            v = tl.load(v_src_ptr + s * ROW + offs, mask=mask)
            tl.store(v_dst_ptr + d * ROW + offs, v, mask=mask)

        # the online-softmax LSE merge (Milakov & Gimelshein 2018)
        # fused with vLLM's per-group fp8 quantize pattern
        @triton.jit
        def merge_attn_quant(a_ptr, b_ptr, la_ptr, lb_ptr, source_ptr,
                             q_ptr, s_ptr,
                             stride_a_t, stride_a_h, stride_b_t, stride_b_h,
                             stride_la_t, stride_la_h,
                             stride_lb_t, stride_lb_h,
                             stride_q_t, s_stride_g, s_stride_t,
                             D: tl.constexpr, GPB: tl.constexpr,
                             UE8M0: tl.constexpr):
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
            amax = tl.max(tl.abs(y), axis=1)
            scale = tl.maximum(amax, 1e-10) / 448.0
            if UE8M0:
                scale = tl.math.exp2(tl.ceil(tl.math.log2(scale)))
            q = y / scale[:, None]
            q = tl.minimum(tl.maximum(q, -448.0), 448.0)
            tl.store(q_ptr + t * stride_q_t + h0 * D + offs,
                     tl.reshape(q, (D * GPB,)).to(q_ptr.dtype.element_ty))
            tl.store(s_ptr + heads * s_stride_g + t * s_stride_t, scale)

        self._kernels = {"silu": silu_mul_quant,
                         "norm": add_rms_norm_quant,
                         "qk": qk_norm_rope,
                         "kv_scatter": kv_row_scatter,
                         "merge_quant": merge_attn_quant}
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

        One kernel launch per layer.
        """
        assert k3.is_contiguous() and v3.is_contiguous()
        n = src.shape[0]
        row = self.num_kv_heads * self.head_dim
        k_pool, v_pool = self.arena.layer_kv(layer)
        self._triton_kernels()["kv_scatter"][(n,)](
            k3, v3, k_pool, v_pool, src, dst,
            ROW=row, ROW_POW2=1 << (row - 1).bit_length())

    def merge_attn_quant(self, out_a, lse_a, out_b, lse_b, source):
        """Merge cached and fresh attention and write FP8 GEMM input.

        One 128-value FP8 group is one attention head for Qwen3 4B.
        The source vector maps each packed row to its row in out_b, or
        contains -1 when out_a is already the complete answer.
        """
        n, heads, dim = out_a.shape
        assert dim == GROUP
        q = self.torch.empty((n, heads * dim), dtype=self.fp8,
                             device=out_a.device)
        scales = self._col_major_scales(n, heads * dim)
        gpb = 4
        self._triton_kernels()["merge_quant"][(n, heads // gpb)](
            out_a, out_b, lse_a, lse_b, source, q, scales,
            out_a.stride(0), out_a.stride(1),
            out_b.stride(0), out_b.stride(1),
            lse_a.stride(0), lse_a.stride(1),
            lse_b.stride(0), lse_b.stride(1),
            q.stride(0), scales.stride(1), scales.stride(0),
            D=dim, GPB=gpb, UE8M0=self.use_ue8m0)
        return q, scales

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

        The chunk carries the path it was packed for.
        """
        if chunk.attention_mode == "merge_quant":
            if not self.is_fp8:
                raise ValueError("BF16 forward passes require unified attention")
            return self.attention_merge_quant(q, k, v, chunk.meta)
        return self.quant(self.attention_unified(q, k, v, chunk.meta))

    # ---- attention: the two workload paths --------------------------

    def _fa(self, q, k, v, cu_q, cu_k, max_q, max_k, causal,
            block_table=None, seqused_k=None):
        # vLLM's FA2 and FA3 both accept 16-token pages and return
        # LSE as [heads, total_queries] for the merge kernel.
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        return flash_attn_varlen_func(
            q, k, v, max_seqlen_q=max_q, cu_seqlens_q=cu_q,
            max_seqlen_k=max_k, cu_seqlens_k=cu_k,
            block_table=block_table, seqused_k=seqused_k,
            causal=causal, fa_version=self.fa_version, return_softmax_lse=True)

    def attention_merge_quant(self, q, k, v, meta):
        """The two-call attention path with the fused merge plus FP8 quantize.

        Returns the input pair for o_proj.
        """
        n = q.shape[0]
        H, KH, D = self.num_q_heads, self.num_kv_heads, self.head_dim
        q3 = q.view(n, H, D)
        k3 = k.view(n, KH, D)
        v3 = v.contiguous().view(n, KH, D)
        layer = meta["layer"]

        if meta["kv_src"] is not None:
            self.kv_row_scatter(k3, v3, meta["kv_src"], meta["kv_dst"],
                                layer)

        out_a, lse_a = self._fa(
            q3, k3, v3, meta["cu_a"], meta["cu_a"],
            meta["max_a"], meta["max_a"], causal=True)
        cross = meta["cross"]
        if cross is None:
            meta["layer"] += 1
            return self.quant(out_a.view(n, H * D))

        rows = cross["rows"]
        q_suf = q3.index_select(0, rows)
        kp, vp = self.arena.paged_kv(layer)
        out_b, lse_b = self._fa(
            q_suf, kp, vp, cross["cu_q"], None,
            cross["max_q"], cross["max_used"], causal=False,
            block_table=cross["table"], seqused_k=cross["used"])
        lse_a = lse_a.transpose(0, 1)
        lse_b = lse_b.transpose(0, 1)
        q_out, scales = self.merge_attn_quant(
            out_a, lse_a, out_b, lse_b, cross["source"])
        meta["layer"] += 1
        return q_out, scales

    def attention_unified(self, q, k, v, meta):
        """Write current KV to its cache slots, then one causal paged attention call.

        The call reads retained and current KV together.

        When meta["unified"] is None (no arena pages), falls back to
        a plain varlen causal call with no scatter or paged read.
        """
        n = q.shape[0]
        H, KH, D = self.num_q_heads, self.num_kv_heads, self.head_dim
        q3 = q.view(n, H, D)
        k3 = k.view(n, KH, D)
        v3 = v.contiguous().view(n, KH, D)
        layer = meta["layer"]
        unified = meta["unified"]
        if unified is None:
            out, _ = self._fa(
                q3, k3, v3, meta["cu_a"], meta["cu_a"],
                meta["max_a"], meta["max_a"], causal=True)
            meta["layer"] += 1
            return out.view(n, H * D)
        if unified["src"].numel():
            self.kv_row_scatter(k3, v3, unified["src"], unified["dst"],
                                layer)
        if unified["tail_src"] is not None:
            k_pool, v_pool = self.arena.layer_kv(layer)
            self.kv_row_scatter(k_pool, v_pool,
                                unified["tail_src"], unified["tail_dst"], layer)
        kp, vp = self.arena.paged_kv(layer)
        out, _ = self._fa(
            q3, kp, vp, unified["cu_q"], None,
            unified["max_q"], unified["max_used"], causal=True,
            block_table=unified["table"], seqused_k=unified["used"])
        meta["layer"] += 1
        return out.view(n, H * D)
