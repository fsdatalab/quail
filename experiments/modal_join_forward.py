"""The join prototype: packed forward passes against grouped stock vLLM.

A join here is an AI-if over pairs: one yes/no question about an
anchor document and a partner document, answer constrained to the
YES/NO tokens, zero decode. The anchor sits first in the prompt so
its KV depends on nothing pair-specific; the partner is the per-pair
suffix and is recomputed for every pair in every design.

Three entrypoints:

  probe     the correctness and rate gates, before any timed arm.
            One chunk of [prefix | k suffixes] with shared-prefix
            attention (two FlashAttention-3 calls merged by softmax
            state - the Hydragen / cascade decomposition) against the
            same pairs run one-per-chunk, where no sharing exists and
            the plain causal path is trivially correct. Gate (a):
            identical answers, logits close. Gate (b): chunk
            throughput near the derived effective rate. Plus a
            kept-KV replay: the same suffixes against a stored prefix
            tensor must reproduce the in-chunk answers exactly.

  join2way  the BioDEX sample: 100 reports x all terms. Arms: packed
            at the derived B* (brim-packed chunks from pack_stream,
            several reports per chunk, next chunk built on the CPU
            while the GPU runs the current one), and grouped stock
            vLLM - synchronous, one generate() over the pair list in
            report order, admission via max_num_seqs derived from
            the filter run's token budget. The join has no
            gating, so nothing here needs the async client the
            filter experiment's stock arm used. (A packed reference
            cell at 25,305 ran in earlier revisions and was dropped:
            its prefix recompute changes the prefix/suffix token
            mix, so it was not a single-variable chunk-size
            control.)

  nway3     the planted 3-relation chain on IMDB reviews, no vLLM:
            per B document, stage 1 packs [prefix | A suffixes] and
            keeps the prefix K/V per layer; a B with any YES runs
            stage 2 as [C suffixes] against the kept tensors. The
            loop is software-pipelined: the CPU builds the next
            chunks and reads gate answers (event-synced pinned
            copies) while the GPU runs, so stage walls are per-stage
            CUDA-event sums and total_wall_s is the loop wall. The
            recorded answers replay through a nested-loop reference
            and the triple sets must be identical, and stage-2 pair
            count must equal survivors x |C| exactly.

Predictions, stated before the runs (plans/join_plan.md, derived by
plans/join_estimates.py from committed artifacts; document lengths
were assumed - this file measures them and prints both):
  - packed 2-way sample at B*: 3.3 min at ~110k tok/s effective
    (121,045 measured packed rate minus priced cross-attention).
  - grouped stock: 5.4 min.
  - nway3: about a minute of GPU per stage.

Run:
    modal run experiments/modal_join_forward.py::run_probe
    modal run experiments/modal_join_forward.py::run_join2way
    modal run experiments/modal_join_forward.py::run_nway3
"""

import json
import os

import modal

from workload import IMAGE_BASE, MODEL, hf_cache, results_vol

join_image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub", "pandas", "pyarrow",
                 "numpy", "datasets")
    .env({"VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("workload", "quail")
)

app = modal.App("quail-join-forward")

GROUP = 128            # fp8 quant group size, matches the engine
SLACK = 2              # the budget formula's declared slack
DATA_SEED = 20260817

PREAMBLE = ("You will be shown a patient report and one candidate "
            "medical reaction term. Decide whether the report "
            "describes that reaction as something the patient "
            "experienced.\n\nREPORT:\n")


def pair_suffix_text(term):
    return (f"\n\nCANDIDATE REACTION: {term}\n"
            f"Instruction: answer YES if the report above describes "
            f"this reaction, NO otherwise.\nANSWER=")


# --------------------------------------------------------- chunk budget

def chunk_budget():
    """B* = (M x util - weights - kept) / act, over the declared
    slack; sigma = 0 because suffixes never write KV. Same numbers
    as plans/join_estimates.py."""
    from quail.configs import H100_SXM, QWEN3_4B_FP8
    from quail.plan.cost import ACT_BYTES_PER_HIDDEN, BOOT_POOL_FRACTION
    act = ACT_BYTES_PER_HIDDEN * QWEN3_4B_FP8.h
    free = H100_SXM.M * BOOT_POOL_FRACTION - QWEN3_4B_FP8.W_mem
    return int(free // act) // SLACK


# ------------------------------------------------------------ the model

def _load_vllm_model():
    """The checkpoint as vLLM's processed module - merged qkv and
    gate_up, FP8 weights and block scales laid out for DeepGEMM. No
    engine, no KV pool. Mirrors modal_single_filter_forward."""
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model
    from vllm.utils.network_utils import get_open_port

    config = EngineArgs(model=MODEL, dtype="bfloat16",
                        enforce_eager=True).create_engine_config()
    with set_current_vllm_config(config):
        import torch.distributed as dist
        if not dist.is_initialized():
            init_distributed_environment(
                world_size=1, rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
                local_rank=0, backend="nccl")
            ensure_model_parallel_initialized(1, 1)
        model = get_model(vllm_config=config)
    torch.cuda.synchronize()
    return model


class JoinPipeline:
    """Packed forward passes with shared-prefix attention.

    A chunk is [prefix | suffix_1 .. suffix_k] (or suffixes only,
    against a kept prefix tensor). Per layer, attention is two
    FlashAttention-3 varlen calls merged by softmax state: call A is
    causal over the segment boundaries (prefix over itself, each
    suffix over itself), call B is non-causal - every suffix token
    against the prefix K/V, which is either this chunk's first f rows
    or a tensor kept from an earlier chunk. Suffix positions start at
    f, identical to standalone requests, so answers are comparable to
    per-pair prompts. Everything else is the round-4 custom pipeline:
    DeepGEMM multiplies, our three Triton kernels."""

    def __init__(self, model):
        import torch
        from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

        self.torch = torch
        self.model = model
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
        self.new_kv = None    # anchor -> {layer: (K, V)} written this pass

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

    # ---- the three Triton kernels (round 3/4, verbatim) -------------

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

        self._kernels = {"silu": silu_mul_quant,
                         "norm": add_rms_norm_quant,
                         "qk": qk_norm_rope}
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

    def _fa(self, q, k, v, cu_q, cu_k, max_q, max_k, causal):
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        return flash_attn_varlen_func(
            q, k, v, max_seqlen_q=max_q, cu_seqlens_q=cu_q,
            max_seqlen_k=max_k, cu_seqlens_k=cu_k, causal=causal,
            fa_version=3, return_softmax_lse=True)

    @staticmethod
    def _lse_tokens_first(lse, n_tokens):
        # normalize to (tokens, heads); the wrapper returns
        # (heads, tokens) for varlen
        if lse.shape[0] != n_tokens:
            return lse.transpose(0, 1).contiguous()
        return lse

    def attention(self, q, k, v, meta):
        """meta carries the chunk layout; see pack_join_chunk."""
        torch = self.torch
        n = q.shape[0]
        H, KH, D = self.num_q_heads, self.num_kv_heads, self.head_dim
        q3 = q.view(n, H, D)
        k3 = k.view(n, KH, D)
        v3 = v.contiguous().view(n, KH, D)
        layer = meta["layer"]

        for anchor, r0, r1 in meta["kv_writes"]:
            self.new_kv.setdefault(anchor, {})[layer] = (
                k3[r0:r1].clone(), v3[r0:r1].clone())

        out_a, lse_a = self._fa(
            q3, k3, v3, meta["cu_a"], meta["cu_a"],
            meta["max_a"], meta["max_a"], causal=True)

        if meta["cu_b_q"] is None:
            # no shared-prefix reads in this chunk: whole-pair
            # segments (probe reference) or a cache-only pass
            meta["layer"] += 1
            return out_a.view(n, H * D)

        # call B's keys: each group's prefix rows, fresh from this
        # chunk or from the kept store, concatenated in group order
        ks, vs = [], []
        for src in meta["cross_src"]:
            if src[0] == "chunk":
                _, r0, r1 = src
                ks.append(k3[r0:r1])
                vs.append(v3[r0:r1])
            else:
                kp, vp = meta["kv_cache"][src[1]][layer]
                ks.append(kp)
                vs.append(vp)
        kx = ks[0] if len(ks) == 1 else torch.cat(ks)
        vx = vs[0] if len(vs) == 1 else torch.cat(vs)

        rows = meta["suffix_rows"]
        q_suf = q3.index_select(0, rows)
        out_b, lse_b = self._fa(
            q_suf, kx, vx, meta["cu_b_q"], meta["cu_b_k"],
            meta["max_b_q"], meta["max_b_k"], causal=False)

        la = self._lse_tokens_first(lse_a, n).index_select(0, rows)
        lb = self._lse_tokens_first(lse_b, rows.shape[0])
        # (wa*A + wb*B)/(wa+wb) == A + (B-A)*sigmoid(lse_b - lse_a):
        # same merge, no fp32 copies of the row tensors
        w = torch.sigmoid(lb - la).to(torch.bfloat16)[..., None]
        merged = torch.lerp(out_a.index_select(0, rows), out_b, w)
        out = out_a.index_copy_(0, rows, merged)
        meta["layer"] += 1
        return out.view(n, H * D)

    # ---- the forward loop -------------------------------------------

    def forward_chunk(self, chunk):
        torch = self.torch
        meta = chunk["meta"]
        meta["layer"] = 0
        input_ids, positions = chunk["input_ids"], chunk["positions"]
        self.new_kv = {}
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
        new_kv = self.new_kv
        self.new_kv = None
        return normed, new_kv


def pack_join_chunk(torch, groups, kv_cache=None):
    """Tensors for one chunk, built from groups in chunk order.

    Each group is a dict:
      anchor    id used for KV-cache writes and lookups
      prefix    fresh prefix token list, packed into the chunk - or
                None when the anchor's KV is already in kv_cache
      f         prefix length in tokens; required when prefix is
                None (suffix positions start at f either way, so
                answers match standalone [prefix | suffix] prompts)
      suffixes  list of suffix token lists (may be empty for a
                cache-only group)
      cache_kv  write this group's fresh prefix KV per layer
                into the cache returned by forward_chunk

    kv_cache: anchor -> {layer: (K, V)}, the KV cache read by
    groups whose prefix is not in this chunk.
    """
    dev = "cuda"
    ids, pos, cu_a, finals = [], [], [0], []
    suffix_rows = []
    cross_src, cu_b_q, cu_b_k = [], [0], [0]
    max_b_q = max_b_k = 0
    kv_writes = []
    for g in groups:
        fresh = g.get("prefix") is not None
        f = len(g["prefix"]) if fresh else g["f"]
        row0 = len(ids)
        if fresh:
            ids.extend(g["prefix"])
            pos.extend(range(f))
            cu_a.append(len(ids))
            if g.get("cache_kv"):
                kv_writes.append((g["anchor"], row0, row0 + f))
        s_row0 = len(ids)
        for suf in g["suffixes"]:
            ids.extend(suf)
            pos.extend(range(f, f + len(suf)))
            cu_a.append(len(ids))
            finals.append(len(ids) - 1)
        s_count = len(ids) - s_row0
        if s_count and f:
            suffix_rows.extend(range(s_row0, len(ids)))
            cu_b_q.append(cu_b_q[-1] + s_count)
            cu_b_k.append(cu_b_k[-1] + f)
            max_b_q = max(max_b_q, s_count)
            max_b_k = max(max_b_k, f)
            cross_src.append(("chunk", row0, row0 + f) if fresh
                             else ("kept", g["anchor"]))
    meta = dict(
        layer=0,
        kv_writes=kv_writes,
        kv_cache=kv_cache,
        cu_a=torch.tensor(cu_a, dtype=torch.int32, device=dev),
        max_a=max(cu_a[i + 1] - cu_a[i] for i in range(len(cu_a) - 1)),
        cu_b_q=(torch.tensor(cu_b_q, dtype=torch.int32, device=dev)
                if cross_src else None),
        cu_b_k=(torch.tensor(cu_b_k, dtype=torch.int32, device=dev)
                if cross_src else None),
        max_b_q=max_b_q, max_b_k=max_b_k,
        suffix_rows=(torch.tensor(suffix_rows, dtype=torch.int64,
                                  device=dev) if cross_src else None),
        cross_src=cross_src,
    )
    return dict(
        input_ids=torch.tensor(ids, dtype=torch.int64, device=dev),
        positions=torch.tensor(pos, dtype=torch.int64, device=dev),
        final_indices=torch.tensor(finals, dtype=torch.int64, device=dev),
        meta=meta, tokens=len(ids))


class Answerer:
    """YES/NO from final-position hidden states, scored against only
    the allowed token rows - no full-vocabulary logits."""

    def __init__(self, torch, F, model, tokenizer):
        from workload import yes_no_ids
        yes_ids, no_ids = yes_no_ids(tokenizer)
        self.F = F
        self.allowed = sorted(yes_ids | no_ids)
        self.yes_ids = yes_ids
        sel = torch.tensor(self.allowed, device="cuda")
        self.weights = model.lm_head.weight.index_select(0, sel).to(
            torch.bfloat16)
        self.yes_cols = torch.tensor(
            [i for i, t in enumerate(self.allowed) if t in yes_ids],
            device="cuda")
        self.no_cols = torch.tensor(
            [i for i, t in enumerate(self.allowed) if t in no_ids],
            device="cuda")

    def __call__(self, normed):
        scores = self.F.linear(normed, self.weights)
        yes = scores.index_select(1, self.yes_cols).amax(dim=1)
        no = scores.index_select(1, self.no_cols).amax(dim=1)
        return (yes > no).int().cpu().tolist()

    def margins(self, normed):
        scores = self.F.linear(normed, self.weights)
        yes = scores.index_select(1, self.yes_cols).amax(dim=1)
        no = scores.index_select(1, self.no_cols).amax(dim=1)
        return (yes - no).float().cpu().tolist()


class AsyncAnswers:
    """YES/NO readout that does not stall the stream: submit()
    computes the bits on GPU, enqueues a copy to pinned host memory,
    and records an event; result() waits only for that event - ops
    enqueued after the event (the next chunk's forward) keep the GPU
    busy while the CPU reads the answers."""

    def __init__(self, torch, answerer):
        self.torch = torch
        self.ans = answerer

    def submit(self, normed):
        torch, ans = self.torch, self.ans
        scores = ans.F.linear(normed, ans.weights)
        yes = scores.index_select(1, ans.yes_cols).amax(dim=1)
        no = scores.index_select(1, ans.no_cols).amax(dim=1)
        bits = (yes > no).to(torch.uint8)
        host = torch.empty(bits.shape[0], dtype=torch.uint8,
                           pin_memory=True)
        host.copy_(bits, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        return event, host

    @staticmethod
    def result(handle):
        event, host = handle
        event.synchronize()
        return [int(b) for b in host.tolist()]


def run_join(torch, pipeline, async_ans, anchor_prefixes,
             stage_suffixes, budget, group_size=None):
    """RUN of the plan, for the star shape: every stage streams a
    fresh partner list against the same anchor side. The 2-way is
    the one-stage case; nothing else in this file executes joins.

    anchor_prefixes: anchor id (list index) -> prefix token list.
    stage_suffixes: per stage, the partner suffix token lists.
    budget: the chunk token budget B.
    group_size: anchors gated together between stages; None = all
    anchors in one group (right for one stage, where no gate runs).

    Pipelining, one rule: while the GPU runs a chunk, the CPU builds
    the next buildable one. Within a stage that is the next chunk of
    the plan; at a gate, whose next chunk cannot be built until the
    group's answers arrive, it is the next group's stage-0 chunk
    (which depends on nothing) - launched before the gate resolves,
    so the GPU never drains. Answers travel as event-synced pinned
    copies (AsyncAnswers); an anchor's prefix KV is written to
    the executor's KV cache at most once and freed when its group
    leaves its last stage. Suffix KV is never cached.

    Returns (ans, spans, tokens): ans[j][a] = 0/1 row over stage-j
    partners, present only for anchors that reached stage j; spans =
    (stage, start_event, end_event) per forward for GPU-time sums;
    tokens = fresh tokens packed.
    """
    from quail.joinlogic import pack_stream

    k = len(stage_suffixes)
    n = len(anchor_prefixes)
    group_size = n if group_size is None else group_size
    groups = [list(range(i, min(i + group_size, n)))
              for i in range(0, n, group_size)]
    suffix_lens = [[len(s) for s in sufs] for sufs in stage_suffixes]
    ans = [dict() for _ in range(k)]
    kv_cache, spans = {}, []
    tokens = 0

    def plan_stage(members, j):
        live = [a for a in members
                if j == 0 or any(ans[j - 1].get(a, []))]
        if not live:
            return [], [], set()
        spec = [(len(anchor_prefixes[a]), suffix_lens[j])
                for a in live]
        keep_loc = set(range(len(live))) if j + 1 < k else set()
        already_loc = {i for i, a in enumerate(live)
                       if a in kv_cache}
        plan, to_cache = pack_stream(spec, budget, keep=keep_loc,
                                     already_kept=already_loc)
        return live, plan, to_cache

    def build(j, idx, to_cache, chunk_groups):
        return pack_join_chunk(
            torch,
            [dict(anchor=idx[a],
                  prefix=(anchor_prefixes[idx[a]] if carried
                          else None),
                  f=len(anchor_prefixes[idx[a]]),
                  suffixes=stage_suffixes[j][start:end],
                  cache_kv=(carried and a in to_cache))
             for a, start, end, carried in chunk_groups],
            kv_cache=kv_cache)

    def launch(j, chunk):
        nonlocal tokens
        tokens += chunk["tokens"]
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        normed, new_kv = pipeline.forward_chunk(chunk)
        e1.record()
        spans.append((j, e0, e1))
        kv_cache.update(new_kv)
        return async_ans.submit(normed)

    def scatter(j, idx, chunk_groups, bits):
        pos = 0
        for a, start, end, _ in chunk_groups:
            cnt = end - start
            ans[j].setdefault(idx[a], []).extend(bits[pos:pos + cnt])
            pos += cnt

    prefetch = None     # (idx, plan, to_cache, handle0) of next group's
                        # stage 0, chunk 0 already launched
    for g, members in enumerate(groups):
        for j in range(k):
            if j == 0 and prefetch is not None:
                idx, plan, to_cache, h0 = prefetch
                prefetch = None
            else:
                idx, plan, to_cache = plan_stage(members, j)
                h0 = launch(j, build(j, idx, to_cache, plan[0])) \
                    if plan else None
            if not plan:
                continue
            handles = [(h0, plan[0])]
            if j == 0 and k > 1 and g + 1 < len(groups):
                # the gate below cannot be planned past; keep the
                # GPU fed with the next group's gate-free stage 0
                nidx, nplan, ncache = plan_stage(groups[g + 1], 0)
                if nplan:
                    nh = launch(0, build(0, nidx, ncache, nplan[0]))
                    prefetch = (nidx, nplan, ncache, nh)
            for t in range(1, len(plan)):
                c = build(j, idx, to_cache, plan[t])  # CPU, GPU busy
                h = launch(j, c)
                h_prev, pg = handles.pop(0)
                scatter(j, idx, pg, async_ans.result(h_prev))
                handles.append((h, plan[t]))
            while handles:
                h, pg = handles.pop(0)
                scatter(j, idx, pg, async_ans.result(h))
            # free cached KV that nothing later reads: after the
            # last stage everything in the group is done; between
            # stages, the gate's casualties are done
            if j == k - 1:
                for a in members:
                    kv_cache.pop(a, None)
            else:
                for a in idx:
                    if not any(ans[j][a]):
                        kv_cache.pop(a, None)
    return ans, spans, tokens


# ------------------------------------------------------------- the data

def biodex_sample(tokenizer, n_reports=100, vocab_cap=3718,
                  max_report_tokens=3500, seed=DATA_SEED):
    """100 reports and the reaction-term vocabulary, tokenized.

    Reports and their gold reaction terms come from the BioDEX
    dataset; the vocabulary is the most frequent vocab_cap terms over
    the sampled pool, mirroring the FDJ paper's 3,718-term side.
    Reports are truncated to max_report_tokens (recorded); gold =
    the report's own terms that made the vocabulary, kept only for
    an accuracy-sanity read, never a claim."""
    import numpy as np
    from datasets import load_dataset

    # Streamed: the pool is the first 2,000 usable rows in dataset
    # order (deterministic for a pinned dataset), so the container
    # never downloads the full corpus. The 100 reports are a seeded
    # choice from that pool.
    ds = load_dataset("BioDEX/BioDEX-Reactions", split="train",
                      streaming=True)

    def reactions_of(row):
        return [t.strip() for t in str(row.get("reactions", "")).split(",")
                if t.strip()]

    freq = {}
    rows = []
    for row in ds:
        terms = reactions_of(row)
        text = str(row.get("fulltext_processed") or row.get("abstract"))
        if not terms or len(text) < 200:
            continue
        rows.append((text, terms))
        for t in terms:
            freq[t] = freq.get(t, 0) + 1
        if len(rows) >= 2000:
            break
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    vocab = [t for t, _ in sorted(freq.items(),
                                  key=lambda kv: (-kv[1], kv[0]))]
    vocab = vocab[:vocab_cap]
    vset = set(vocab)
    keep = [(text, [t for t in terms if t in vset])
            for text, terms in rows]
    keep = [r for r in keep if r[1]][:n_reports]
    if len(keep) < n_reports:
        raise RuntimeError(f"only {len(keep)} usable reports")

    pre_ids = tokenizer(PREAMBLE, add_special_tokens=False)["input_ids"]
    prefixes, gold = [], []
    for text, terms in keep:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        prefixes.append(pre_ids + ids[:max_report_tokens])
        gold.append(terms)
    suffixes = [tokenizer(pair_suffix_text(t),
                          add_special_tokens=False)["input_ids"]
                for t in vocab]
    return dict(prefixes=prefixes, suffixes=suffixes, vocab=vocab,
                gold=gold, preamble_tokens=len(pre_ids),
                max_report_tokens=max_report_tokens)


NWAY_PREAMBLE = ("You will be shown a report document and a candidate "
                 "document, each carrying a planted key line. Answer "
                 "from those lines only.\n\nREPORT DOCUMENT:\n")
N_A = 100
N_B = 100
N_C = 100
X_VALUES = 50          # a_j carries X = j % 50
B_GATED = 20           # b_i for i >= 80 gets an X no A document has
Y_VALUES = 25          # c_j carries Y = j % 25


def nway_truth():
    """The planted ground truth, from the key assignment alone."""
    ans1 = {b: [1 if (b < N_B - B_GATED and a % X_VALUES == b % X_VALUES)
                else 0 for a in range(N_A)] for b in range(N_B)}
    ans2_full = {b: [1 if c % Y_VALUES == b % Y_VALUES else 0
                     for c in range(N_C)] for b in range(N_B)}
    return ans1, ans2_full


def nway_corpus(tokenizer):
    """Three planted collections from the IMDB pool. B documents are
    ~4k tokens (12 reviews concatenated) so the keep-KV branch is the
    right choice by the size rule; A and C stay single reviews."""
    from workload import build_pool

    reviews = build_pool(1500)
    a_docs = [f"{reviews[j]}\n\n[KEY] X={j % X_VALUES}"
              for j in range(N_A)]
    c_docs = [f"{reviews[N_A + j]}\n\n[KEY] Y={j % Y_VALUES}"
              for j in range(N_C)]
    b_docs = []
    base = N_A + N_C
    for i in range(N_B):
        body = "\n\n".join(reviews[base + i * 12: base + (i + 1) * 12])
        x = (i % X_VALUES) if i < N_B - B_GATED else 1000 + i
        b_docs.append(f"{body}\n\n[KEYS] X={x} Y={i % Y_VALUES}")

    pre = tokenizer(NWAY_PREAMBLE, add_special_tokens=False)["input_ids"]
    b_prefix = [pre + tokenizer(d, add_special_tokens=False)["input_ids"]
                for d in b_docs]

    def suffix(doc, key):
        return tokenizer(
            f"\n\nCANDIDATE DOCUMENT:\n{doc}\n"
            f"Instruction: answer YES if the [KEY] {key} value in the "
            f"candidate equals the [KEYS] {key} value in the report "
            f"document, NO otherwise.\nANSWER=",
            add_special_tokens=False)["input_ids"]

    a_suffix = [suffix(d, "X") for d in a_docs]
    c_suffix = [suffix(d, "Y") for d in c_docs]
    return b_prefix, a_suffix, c_suffix


# ------------------------------------------------------------ the probe

@app.function(image=join_image, gpu="H100!", timeout=2400, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def probe() -> str:
    """Gates before any timed arm: shared-prefix answers identical to
    the one-pair-per-chunk path, kept-KV replay identical to
    in-chunk, and chunk throughput near the derived effective rate."""
    import time

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from quail.joinlogic import plan_groups

    import math as _math

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = _load_vllm_model()
    pipeline = JoinPipeline(model)
    answerer = Answerer(torch, F, model, tokenizer)
    result = {"b_star": chunk_budget()}

    # attention math in isolation: the two-call merge against a plain
    # fp32 reference on random tensors shaped [prefix | 3 suffixes].
    # If this is not tight, the merge or the FA calls are wrong; if it
    # is tight and pipeline pairs still flip, the flip is knife-edge
    # amplification, not math.
    H, KH, D = pipeline.num_q_heads, pipeline.num_kv_heads, \
        pipeline.head_dim
    f0, sufs0 = 256, [32, 32, 32]
    n0 = f0 + sum(sufs0)
    gen = torch.Generator(device="cuda").manual_seed(7)
    q0 = torch.randn(n0, H, D, device="cuda", dtype=torch.bfloat16,
                     generator=gen)
    k0 = torch.randn(n0, KH, D, device="cuda", dtype=torch.bfloat16,
                     generator=gen)
    v0 = torch.randn(n0, KH, D, device="cuda", dtype=torch.bfloat16,
                     generator=gen)
    cu = [0, f0]
    for s in sufs0:
        cu.append(cu[-1] + s)
    meta0 = dict(layer=0, kv_writes=[], kv_cache=None,
                 cu_a=torch.tensor(cu, dtype=torch.int32, device="cuda"),
                 max_a=f0,
                 cu_b_q=torch.tensor([0, n0 - f0], dtype=torch.int32,
                                     device="cuda"),
                 cu_b_k=torch.tensor([0, f0], dtype=torch.int32,
                                     device="cuda"),
                 max_b_q=n0 - f0, max_b_k=f0,
                 suffix_rows=torch.arange(f0, n0, device="cuda"),
                 cross_src=[("chunk", 0, f0)])
    with torch.inference_mode():
        shared0 = pipeline.attention(
            q0.reshape(n0, H * D), k0.reshape(n0, KH * D),
            v0.reshape(n0, KH * D), meta0).view(n0, H, D)

        def ref_rows(qr, kr, vr):
            kk = kr.repeat_interleave(H // KH, dim=1).float()
            vv = vr.repeat_interleave(H // KH, dim=1).float()
            s = torch.einsum("qhd,khd->hqk", qr.float(), kk)
            s = s / _math.sqrt(D)
            nq = qr.shape[0]
            mask = torch.triu(torch.ones(nq, nq, device="cuda",
                                         dtype=torch.bool), 1)
            s.masked_fill_(mask[None], float("-inf"))
            return torch.einsum("hqk,khd->qhd", s.softmax(-1), vv)

        worst = 0.0
        off = f0
        for s in sufs0:
            idx = torch.tensor(list(range(f0))
                               + list(range(off, off + s)), device="cuda")
            ref = ref_rows(q0.index_select(0, idx),
                           k0.index_select(0, idx),
                           v0.index_select(0, idx))[f0:]
            got = shared0[off:off + s].float()
            worst = max(worst, (ref - got).abs().max().item())
            off += s
    result["attention_math_max_diff"] = round(worst, 4)

    data = biodex_sample(tokenizer, n_reports=4)
    prefix = data["prefixes"][0]
    sufs = data["suffixes"][:64]
    result["lengths"] = dict(
        prefix=len(prefix),
        suffix_mean=round(sum(map(len, sufs)) / len(sufs), 1))

    with torch.inference_mode():
        # shared: one chunk, one prefix, 64 suffixes
        shared_chunk = pack_join_chunk(
            torch, [dict(anchor=0, prefix=prefix, suffixes=sufs)])
        normed_shared, _ = pipeline.forward_chunk(shared_chunk)
        shared_answers = answerer(normed_shared)

        # unshared reference: each pair as its own single-segment
        # chunk - plain causal attention, trivially correct
        unshared_answers, unshared_rows = [], []
        for suf in sufs:
            one = pack_join_chunk(
                torch, [dict(anchor=0, prefix=prefix + suf,
                             suffixes=[])])
            # a chunk with no suffixes has no final rows; treat the
            # whole pair as one segment whose last row answers
            one["final_indices"] = torch.tensor(
                [len(prefix) + len(suf) - 1], device="cuda")
            normed, _ = pipeline.forward_chunk(one)
            unshared_rows.append(normed)
            unshared_answers.extend(answerer(normed))
        gap = (normed_shared.float()
               - torch.cat(unshared_rows).float()).abs().max().item()

        # kept-KV replay: write the prefix KV in one chunk, rerun the
        # same suffixes against the stored tensors
        cache_chunk = pack_join_chunk(
            torch, [dict(anchor=0, prefix=prefix, suffixes=sufs,
                         cache_kv=True)])
        _, kv = pipeline.forward_chunk(cache_chunk)
        kept_chunk = pack_join_chunk(
            torch, [dict(anchor=0, prefix=None, f=len(prefix),
                         suffixes=sufs)],
            kv_cache=kv)
        normed_kept, _ = pipeline.forward_chunk(kept_chunk)
        kept_answers = answerer(normed_kept)

        # multi-group gate: two reports in ONE chunk must answer
        # exactly as each does alone (the brim-packed path)
        p1 = data["prefixes"][1]
        sufs2 = data["suffixes"][64:96]
        alone0, _ = pipeline.forward_chunk(pack_join_chunk(
            torch, [dict(anchor=0, prefix=prefix, suffixes=sufs2)]))
        alone1, _ = pipeline.forward_chunk(pack_join_chunk(
            torch, [dict(anchor=1, prefix=p1, suffixes=sufs2)]))
        both, _ = pipeline.forward_chunk(pack_join_chunk(
            torch, [dict(anchor=0, prefix=prefix, suffixes=sufs2),
                    dict(anchor=1, prefix=p1, suffixes=sufs2)]))
        multi_expect = answerer(alone0) + answerer(alone1)
        multi_got = answerer(both)
        dis_multi = sum(x != y for x, y in zip(multi_got, multi_expect))

        # mixed gate: a fresh group and a kept group in ONE chunk
        mixed, _ = pipeline.forward_chunk(pack_join_chunk(
            torch, [dict(anchor=1, prefix=p1, suffixes=sufs2),
                    dict(anchor=0, prefix=None, f=len(prefix),
                         suffixes=sufs2)],
            kv_cache=kv))
        mixed_expect = answerer(alone1) + answerer(alone0)
        mixed_got = answerer(mixed)
        dis_mixed = sum(x != y for x, y in zip(mixed_got, mixed_expect))

    dis_su = [i for i, (a, b) in enumerate(
        zip(shared_answers, unshared_answers)) if a != b]
    dis_ks = [i for i, (a, b) in enumerate(
        zip(kept_answers, shared_answers)) if a != b]
    m_shared = answerer.margins(normed_shared)
    m_unshared = answerer.margins(torch.cat(unshared_rows))
    row_gap = (normed_shared.float()
               - torch.cat(unshared_rows).float()).abs().amax(dim=1)
    result["gates"] = dict(
        shared_vs_unshared_disagreements=len(dis_su),
        disagreeing_pairs=dis_su[:8],
        disagreeing_margins_shared=[round(m_shared[i], 3)
                                    for i in dis_su[:8]],
        disagreeing_margins_unshared=[round(m_unshared[i], 3)
                                      for i in dis_su[:8]],
        shared_vs_unshared_max_hidden_gap=round(gap, 4),
        rows_with_gap_over_1=int((row_gap > 1.0).sum().item()),
        median_row_gap=round(row_gap.median().item(), 4),
        kept_vs_shared_disagreements=len(dis_ks),
        kept_disagreeing_pairs=dis_ks[:8],
        multi_group_disagreements=int(dis_multi),
        mixed_kept_fresh_disagreements=int(dis_mixed),
        finite=bool(torch.isfinite(normed_shared).all().item()),
    )

    # rate gate: a full report x all terms at B* (one chunk each)
    data_full = biodex_sample(tokenizer, n_reports=2)
    rates = {}
    with torch.inference_mode():
        for name, budget in (("b_star", chunk_budget()),):
            chunks = []
            for r, p in enumerate(data_full["prefixes"]):
                for start, end in plan_groups(
                        len(p), [len(s) for s in data_full["suffixes"]],
                        budget):
                    chunks.append(pack_join_chunk(
                        torch,
                        [dict(anchor=r, prefix=p,
                              suffixes=data_full["suffixes"][start:end])]))
            for c in chunks[:2]:
                pipeline.forward_chunk(c)     # deepgemm + shape warmup
            torch.cuda.synchronize()
            walls = []
            for _ in range(2):
                t0 = time.perf_counter()
                for c in chunks:
                    pipeline.forward_chunk(c)
                torch.cuda.synchronize()
                walls.append(time.perf_counter() - t0)
            wall = min(walls)
            tokens = sum(c["tokens"] for c in chunks)
            rates[name] = dict(chunks=len(chunks), tokens=tokens,
                               wall_s=round(wall, 3),
                               tok_s=round(tokens / wall, 1))
    result["rates"] = rates
    result["peak_gib"] = round(
        torch.cuda.max_memory_allocated() / 2**30, 2)

    print(json.dumps(result, indent=2), flush=True)
    with open("/results/join_probe.json", "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    return json.dumps(result)


# ------------------------------------------------------------ the 2-way

@app.function(image=join_image, gpu="H100!", timeout=10800, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def join2way(n_reports: int = 100, reps_packed: int = 2,
             reps_stock: int = 2) -> str:
    import gc
    import time

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    data = biodex_sample(tokenizer, n_reports=n_reports)
    prefixes, suffixes = data["prefixes"], data["suffixes"]
    n_terms = len(suffixes)
    pairs = n_reports * n_terms
    suffix_lens = [len(s) for s in suffixes]
    fresh_star = sum(len(p) for p in prefixes) + n_reports * sum(suffix_lens)

    report = dict(
        model=MODEL, n_reports=n_reports, n_terms=n_terms, pairs=pairs,
        b_star=chunk_budget(),
        lengths=dict(
            prefix_mean=round(sum(map(len, prefixes)) / n_reports, 1),
            prefix_max=max(map(len, prefixes)),
            suffix_mean=round(sum(suffix_lens) / n_terms, 1),
            preamble=data["preamble_tokens"],
            max_report_tokens=data["max_report_tokens"]),
        prediction=("packed B* ~96 s at measured lengths: same 8.42M "
                    "tokens as the prior 98.7 s run, brim-packed into "
                    "~20 chunks with builds overlapped; peak ~40 GiB; "
                    "stock ~433 s as before"),
        runs=[])
    print(f"[join2way] {pairs:,} pairs; fresh tokens at B* "
          f"{fresh_star / 1e6:.1f}M; lengths {report['lengths']}",
          flush=True)

    # ---- stock arm first (its answers are the cross-implementation
    # reference), then it is torn down before the packed arms
    from vllm import LLM, SamplingParams
    from workload import yes_no_ids

    yes_ids, no_ids = yes_no_ids(tokenizer)
    max_len = max(len(p) for p in prefixes) + max(suffix_lens) + 16
    admission_budget = 749_782      # the filter run's committed budget
    # a stock pair request costs its FULL prompt (prefix + suffix);
    # the packed side's shared accounting must not leak in here
    mean_pair = (sum(len(p) for p in prefixes) * n_terms
                 + n_reports * sum(suffix_lens)) // pairs + 1
    max_seqs = max(64, min(4096, admission_budget // mean_pair))
    # stock's step-token cap: the largest measured point of the filter
    # sweep - the setting the committed stock run used
    stock_batched = 25_305
    llm = LLM(model=MODEL, kv_cache_dtype="auto",
              max_model_len=max_len, max_num_seqs=max_seqs,
              max_num_batched_tokens=stock_batched,
              gpu_memory_utilization=0.88,
              enable_prefix_caching=True, disable_log_stats=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1,
                              allowed_token_ids=sorted(yes_ids | no_ids))
    pair_prompts = [{"prompt_token_ids": p + s}
                    for p in prefixes for s in suffixes]
    report["stock"] = dict(max_num_seqs=max_seqs,
                           kv_cache_dtype="bf16",
                           max_num_batched_tokens=stock_batched,
                           admission_budget=admission_budget)
    llm.generate(pair_prompts[:64], sampling, use_tqdm=False)
    stock_answers = None
    for rep in range(reps_stock):
        llm.reset_prefix_cache()
        t0 = time.perf_counter()
        outputs = llm.generate(pair_prompts, sampling, use_tqdm=False)
        wall = time.perf_counter() - t0
        answers = [1 if int(o.outputs[0].token_ids[0]) in yes_ids else 0
                   for o in outputs]
        cached = sum(getattr(o, "num_cached_tokens", 0) or 0
                     for o in outputs)
        prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
        row = dict(method="stock_grouped", rep=rep, wall=round(wall, 2),
                   fresh_tokens=prompt_tokens - cached,
                   tok_s=round((prompt_tokens - cached) / wall, 1),
                   yes=sum(answers))
        report["runs"].append(row)
        print(f"[join2way] {row}", flush=True)
        stock_answers = answers
    del outputs, llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)

    # ---- packed arm
    model = _load_vllm_model()
    pipeline = JoinPipeline(model)
    answerer = Answerer(torch, F, model, tokenizer)
    async_ans = AsyncAnswers(torch, answerer)

    def run_arm(name, budget, reps):
        packed_answers = None
        for rep in range(reps):
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            with torch.inference_mode():
                ans, spans, tokens = run_join(
                    torch, pipeline, async_ans, prefixes,
                    [suffixes], budget)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            answers = []
            for a in range(n_reports):
                answers.extend(ans[0][a])
            row = dict(method=name, rep=rep, wall=round(wall, 2),
                       chunks=len(spans), fresh_tokens=tokens,
                       tok_s=round(tokens / wall, 1),
                       yes=sum(answers),
                       agrees_with_stock=(
                           None if stock_answers is None else
                           sum(a == b for a, b in
                               zip(answers, stock_answers))),
                       peak_gib=round(
                           torch.cuda.max_memory_allocated() / 2**30, 2))
            report["runs"].append(row)
            print(f"[join2way] {row}", flush=True)
            packed_answers = answers
        return packed_answers

    with torch.inference_mode():
        warm = pack_join_chunk(
            torch, [dict(anchor=0, prefix=prefixes[0],
                         suffixes=suffixes[:64])])
        pipeline.forward_chunk(warm)
    packed_answers = run_arm("packed_bstar_cell", chunk_budget(),
                             reps_packed)

    # accuracy sanity only - never a claim (report's own terms vs
    # packed answers)
    vset = {t: i for i, t in enumerate(data["vocab"])}
    tp = fp = fn = 0
    for r, terms in enumerate(data["gold"]):
        truth = {vset[t] for t in terms}
        for t in range(n_terms):
            a = packed_answers[r * n_terms + t]
            if a and t in truth:
                tp += 1
            elif a:
                fp += 1
            elif t in truth:
                fn += 1
    report["accuracy_sanity"] = dict(tp=tp, fp=fp, fn=fn)

    with open("/results/join2way.json", "w") as f:
        json.dump(report, f, indent=2)
    results_vol.commit()
    print(json.dumps({k: v for k, v in report.items() if k != "runs"},
                     indent=2), flush=True)
    return json.dumps(report)


# ------------------------------------------------------------- the 3-way

@app.function(image=join_image, gpu="H100!", timeout=7200, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def nway3() -> str:
    import time

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from quail.joinlogic import assemble, brute_force_triples, gate

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = _load_vllm_model()
    pipeline = JoinPipeline(model)
    answerer = Answerer(torch, F, model, tokenizer)

    b_prefix, a_suffix, c_suffix = nway_corpus(tokenizer)
    truth1, truth2 = nway_truth()
    report = dict(
        n_a=N_A, n_b=N_B, n_c=N_C,
        lengths=dict(
            b_prefix_mean=round(sum(map(len, b_prefix)) / N_B, 1),
            a_suffix_mean=round(sum(map(len, a_suffix)) / N_A, 1),
            c_suffix_mean=round(sum(map(len, c_suffix)) / N_C, 1)),
        prediction="stage GPU sums near the prior 48.7 + 29.7 walls "
                   "minus the per-B stops; total wall ~70 s; triples "
                   "equal the replay reference; stage-2 pairs = "
                   "survivors x 100")

    async_ans = AsyncAnswers(torch, answerer)
    with torch.inference_mode():
        warm = pack_join_chunk(torch, [dict(
            anchor=-1, prefix=b_prefix[0], suffixes=a_suffix[:8],
            cache_kv=True)])
        pipeline.forward_chunk(warm)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ans, spans, _ = run_join(torch, pipeline, async_ans,
                                 b_prefix, [a_suffix, c_suffix],
                                 chunk_budget(), group_size=1)
        torch.cuda.synchronize()
    total_wall = time.perf_counter() - t0
    ans1, ans2 = ans[0], ans[1]
    stage1_s = sum(e0.elapsed_time(e1) for s, e0, e1 in spans
                   if s == 0) / 1e3
    stage2_s = sum(e0.elapsed_time(e1) for s, e0, e1 in spans
                   if s == 1) / 1e3
    stage2_pairs = sum(len(row) for row in ans2.values())

    survivors = gate(ans1)
    staged = assemble(ans1, ans2)
    reference = brute_force_triples(ans1, ans2)
    model_vs_planted1 = sum(
        ans1[b][a] != truth1[b][a] for b in range(N_B) for a in range(N_A))
    report["result"] = dict(
        stage1_wall_s=round(stage1_s, 1),
        stage2_wall_s=round(stage2_s, 1),
        total_wall_s=round(total_wall, 1),
        survivors=len(survivors),
        stage2_pairs=stage2_pairs,
        stage2_pairs_expected=len(survivors) * N_C,
        triples=len(staged),
        triples_match_replay=staged == reference,
        stage1_model_vs_planted_wrong=model_vs_planted1,
        planted_expected_survivors=N_B - B_GATED,
    )
    print(json.dumps(report, indent=2), flush=True)
    with open("/results/join_nway3.json", "w") as f:
        json.dump(report, f, indent=2)
    results_vol.commit()
    return json.dumps(report)


# ------------------------------------------------------------- entries

def _save(payload, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(json.loads(payload), f, indent=2)
    print(f"saved {out}")


@app.local_entrypoint()
def run_probe(out: str = "results/engine/join_probe.json"):
    _save(probe.remote(), out)


@app.local_entrypoint()
def run_join2way(n_reports: int = 100,
                 out: str = "results/engine/join2way.json"):
    _save(join2way.remote(n_reports), out)


@app.local_entrypoint()
def run_nway3(out: str = "results/engine/join_nway3.json"):
    _save(nway3.remote(), out)
