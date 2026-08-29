"""Kernel-source ablation: our fused Triton kernels against the
kernels stock vLLM's compiled graph runs, on one filter query and one
join query.

What "stock vLLM's kernels" means here was measured, not assumed: the
stock_kernels cell below boots stock vLLM 0.26.0 at its defaults on
the same image and profiles a prefill pass
(/results/ablations/kernel_source_stock.json). At the default -O2
level the config enables the RMSNorm+quant and SiLU+quant fusion
passes (a blocked-fp8 checkpoint forces the quant_fp8 custom op on),
but the profiled graph shows the pattern rewrite does not fire for
this model under the ue8m0 scale mode DeepGEMM uses here: no fused
norm+quant or silu+quant kernel appears. Per layer, stock's compiled
graph actually runs two Inductor-generated add+rms_norm kernels, one
Inductor silu*mul kernel, two Inductor kernels for the q/k head norms
plus rotary (natives traced and fused by Inductor; the qk-norm+rope
fusion pass is off at every -O level), and four standalone CUDA
group-quant launches (one per GEMM input). Attention, the GEMMs, and
the KV-cache write sit outside the compiled graph and are the same
kernels the packed executor calls.

Three rungs, the engine untouched (the non-quail paths live in a
Pipeline subclass below):

  quail          our fused Triton kernels (the shipping executor)
  vllm_ops       vLLM's ops called one by one, unfused (the A2 rung
                 of the 2026-08-19 forward-pass ablation), and on the
                 join path vLLM's merge_attn_states kernel plus a
                 separate group-quant in place of our fused
                 merge+quant kernel
  vllm_compiled  the kernel set stock vLLM's compiled graph runs:
                 torch.compile over the native add+rms_norm, silu*mul
                 and q/k-norm+rope math with vLLM's Inductor
                 settings, the same standalone group-quant per GEMM
                 input, and the same merge_attn_states join merge

Workloads: the committed 10,000-document five-filter query (unified
attention) and the 100-report x 256-term BioDEX join query
(merge_quant attention).

    uv run modal run ablations/vllm_compiled_kernels.py::run_probe
    uv run modal run ablations/vllm_compiled_kernels.py::run_queries
    uv run modal run ablations/vllm_compiled_kernels.py::run_profile
"""

import json
import os

import modal

from quail.executor.attention import GROUP, Pipeline

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub", "pandas", "pyarrow",
                 "numpy", "datasets")
    .env({"VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "TRITON_CACHE_DIR": "/root/.cache/kernels/triton",
          "TORCHINDUCTOR_CACHE_DIR":
              "/root/.cache/kernels/torchinductor"})
    .add_local_python_source("quail", "baselines")
    .add_local_dir("tests/gpu", remote_path="/root/gpu_tests")
)

# House rule: never create new Modal app names - new GPU cells attach
# to an existing app.
app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results",
                                     create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)

GPU_KW = dict(image=image, gpu="H100!", memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})

KERNEL_SOURCES = ("quail", "vllm_ops", "vllm_compiled")

# banked current-executor filter counts (results/attention_paths.json,
# unified path, TRUE/FALSE corpus); the quail rung must reproduce them
BANKED_FILTER = dict(answered=40052, survivors=4645, wrong=0)


class KernelSourcePipeline(Pipeline):
    """Pipeline with a swappable kernel source for the small kernels.

    kernel_source picks who provides the between-GEMM kernels and the
    join merge; the loop, packing, arena, GEMMs, and attention calls
    are the base class's in every mode.
    """

    def __init__(self, model, arena, *, attention_mode,
                 kernel_source="quail"):
        super().__init__(model, arena, kernels="quail",
                         attention_mode=attention_mode)
        self._segments = None
        self.kernel_source = kernel_source

    @property
    def kernel_source(self):
        return self._kernel_source

    @kernel_source.setter
    def kernel_source(self, source):
        if source not in KERNEL_SOURCES:
            raise ValueError(f"kernel_source must be one of "
                             f"{KERNEL_SOURCES}, got {source!r}")
        if (source == "vllm_compiled"
                and self.rotary.rotary_dim != self.head_dim):
            raise RuntimeError(
                "the native rope transcription assumes rotary over "
                "the full head dimension")
        self._kernel_source = source
        self.kernels = "quail" if source == "quail" else "vllm"

    # ---- the segments stock's compiled graph runs -------------------
    # The measured stock inventory (kernel_source_stock.json) shows
    # Inductor-generated kernels for add+rms_norm, silu*mul, and
    # q/k-norm+rope, each followed by the standalone group-quant
    # custom op (the fusion passes do not rewrite this model's graph
    # under ue8m0 scales). The native math is transcribed from
    # vllm/ir/ops/layernorm.py (rms_norm, fused_add_rms_norm) and
    # vllm/model_executor/layers/rotary_embedding/{base,common}.py
    # (forward_static, neox style); combo_kernels matches the
    # inductor_compile_config stock sets for torch >= 2.9. The quant
    # is self.quant, the same vLLM group-quant call the engine and
    # stock both run, so ue8m0 scale rounding stays consistent.

    def _compiled_segments(self):
        if self._segments is not None:
            return self._segments
        torch = self.torch
        H, KH, D = self.num_q_heads, self.num_kv_heads, self.head_dim

        def add_norm_native(hidden, residual, weight, eps):
            # residual accumulates in place, as the engine expects;
            # Inductor folds the write into the generated kernel
            residual += hidden
            xf = residual.to(torch.float32)
            variance = xf.pow(2).mean(dim=-1, keepdim=True)
            xf = xf * torch.rsqrt(variance + eps)
            return xf.to(weight.dtype) * weight

        def silu_mul_native(gate_up):
            half = gate_up.shape[-1] // 2
            gate = gate_up[..., :half]
            up = gate_up[..., half:]
            return torch.nn.functional.silu(gate) * up

        def norm(x, w, eps):
            xf = x.to(torch.float32)
            variance = xf.pow(2).mean(dim=-1, keepdim=True)
            xf = xf * torch.rsqrt(variance + eps)
            return xf.to(w.dtype) * w

        def rope(x, cos, sin):
            x1, x2 = torch.chunk(x, 2, dim=-1)
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            return torch.cat((o1, o2), dim=-1)

        def qk_norm_rope_native(qkv, positions, q_w, k_w, cos_sin,
                                eps):
            n = qkv.shape[0]
            q = qkv[:, :H * D].reshape(n, H, D)
            k = qkv[:, H * D:(H + KH) * D].reshape(n, KH, D)
            q = norm(q, q_w, eps)
            k = norm(k, k_w, eps)
            cos_sin_rows = cos_sin.index_select(0, positions)
            cos, sin = cos_sin_rows.chunk(2, dim=-1)
            cos = cos.unsqueeze(-2).to(q.dtype)
            sin = sin.unsqueeze(-2).to(q.dtype)
            q = rope(q, cos, sin).reshape(n, H * D)
            k = rope(k, cos, sin).reshape(n, KH * D)
            return q.contiguous(), k.contiguous()

        options = {"combo_kernels": True,
                   "benchmark_combo_kernel": True}
        self._segments = dict(
            add_norm=torch.compile(add_norm_native, dynamic=True,
                                   options=options),
            silu_mul=torch.compile(silu_mul_native, dynamic=True,
                                   options=options),
            qk=torch.compile(qk_norm_rope_native, dynamic=True,
                             options=options))
        return self._segments

    def vllm_norm_quant(self, hidden, norm, residual):
        if self.kernel_source != "vllm_compiled":
            return super().vllm_norm_quant(hidden, norm, residual)
        normed = self._compiled_segments()["add_norm"](
            hidden, residual, norm.weight, norm.variance_epsilon)
        return self.quant(normed)

    def vllm_silu_quant(self, gate_up):
        if self.kernel_source != "vllm_compiled":
            return super().vllm_silu_quant(gate_up)
        return self.quant(self._compiled_segments()["silu_mul"](
            gate_up))

    def vllm_qk_norm_rope(self, qkv, positions, attn):
        if self.kernel_source != "vllm_compiled":
            return super().vllm_qk_norm_rope(qkv, positions, attn)
        return self._compiled_segments()["qk"](
            qkv, positions, attn.q_norm.weight, attn.k_norm.weight,
            self.rotary.cos_sin_cache, attn.q_norm.variance_epsilon)

    # ---- the join merge: vLLM's merge_attn_states -------------------
    # Stock never merges in its compiled graph (its cascade merge
    # lives inside the attention backend), so both vLLM rungs use its
    # eager merge kernel plus the standalone group-quant. out_b covers
    # only the rows with cached context, so those rows are gathered,
    # merged, and scattered back; the row list comes from the chunk
    # meta stashed by attention_merge_quant below.

    def attention_merge_quant(self, q, k, v, meta):
        cross = meta.get("cross")
        self._merge_rows = None if cross is None else cross["rows"]
        return super().attention_merge_quant(q, k, v, meta)

    def merge_attn_quant(self, out_a, lse_a, out_b, lse_b, source):
        if self.kernel_source == "quail":
            return super().merge_attn_quant(out_a, lse_a, out_b,
                                            lse_b, source)
        from vllm.v1.attention.ops.merge_attn_states import (
            merge_attn_states)
        torch = self.torch
        n, heads, dim = out_a.shape
        rows = self._merge_rows
        a_sub = out_a.index_select(0, rows)
        merged = torch.empty_like(a_sub)
        # lse arrives as a [tokens, heads] transposed view; the merge
        # kernel wants the original [heads, tokens] layout back
        merge_attn_states(
            merged, out_b, lse_b.transpose(0, 1),
            a_sub, lse_a.transpose(0, 1).index_select(1, rows))
        out_a.index_copy_(0, rows, merged)
        return self.quant(out_a.view(n, heads * dim))


# ------------------------------------------------------------- helpers

def _write(result, name):
    print(json.dumps(result, indent=2), flush=True)
    os.makedirs("/results/ablations", exist_ok=True)
    with open(f"/results/ablations/{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    kernel_cache.commit()
    return json.dumps(result)


def _boot(model):
    """Boot the packed executor with the kernel-source pipeline."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from quail.executor.arena import KVArena
    from quail.executor.attention import FILTER_ATTENTION
    from quail.executor.loop import Answerer, AsyncAnswers
    from quail.executor.model import load_model
    from quail.planner import budgets
    from quail.specs import H100_SXM, MODELS

    spec = MODELS[model]
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
    model_mod = load_model(spec.hf_name)
    chunk = budgets.chunk_budget(spec, H100_SXM)
    arena_tok = budgets.arena_tokens(spec, H100_SXM, chunk)
    arena = KVArena(n_layers=spec.layers,
                    n_pages=arena_tok // budgets.PAGE_TOKENS,
                    page_tokens=budgets.PAGE_TOKENS,
                    n_kv=spec.n_kv, d_head=spec.d_head,
                    dtype=torch.bfloat16)
    pipeline = KernelSourcePipeline(
        model_mod, arena, attention_mode=FILTER_ATTENTION)
    answerer = Answerer(torch, F, model_mod, tokenizer)
    async_ans = AsyncAnswers(torch, answerer)
    budget = min(chunk, pipeline.max_chunk_tokens)
    return spec, tokenizer, arena, pipeline, async_ans, budget, arena_tok


def _dequant(torch, q, scales):
    """FP8 groups back to float32 for cross-kernel comparison."""
    return q.to(torch.float32) * scales.to(
        torch.float32).repeat_interleave(GROUP, dim=1)


def _max_abs(torch, left, right):
    return float((left.to(torch.float32)
                  - right.to(torch.float32)).abs().max().item())


def _wrong_count(answers_by_doc, flags):
    wrong = 0
    for d, row in answers_by_doc.items():
        for j, bit in enumerate(row):
            if bit != int(flags[d][j]):
                wrong += 1
    return wrong


def _disagreements(left, right):
    total = 0
    for d in set(left) | set(right):
        a, b = left.get(d, []), right.get(d, [])
        total += sum(x != y for x, y in zip(a, b))
        total += abs(len(a) - len(b))
    return total


# --------------------------------------------------------------- probe

@app.function(timeout=2400, **GPU_KW)
def probe(model: str = "qwen3-4b-fp8") -> str:
    """Per-kernel parity of the two vLLM rungs against the quail rung,
    plus torch.compile sanity for the q/k segment, on real weights and
    random activations. Runs before the measured cells."""
    import sys
    import time

    sys.path.insert(0, "/root/gpu_tests")

    import torch

    from quail.executor.loop import pack_chunk, run_filter, run_join

    (spec, tokenizer, arena, pipeline, async_ans, budget,
     _) = _boot(model)
    torch.manual_seed(20260829)
    report = dict(cell="kernel_source_probe", model=spec.name,
                  vllm_ops={}, vllm_compiled={}, join_answers={},
                  filter_answers={}, filter_real={},
                  unified_chunk={})

    n = 4096
    h = spec.hidden
    layer = pipeline.layers[0]
    attn = layer.self_attn
    gate_up_w = layer.mlp.gate_up_proj.weight.shape[0]
    qkv_w = attn.qkv_proj.weight.shape[0]

    with torch.inference_mode():
        # norm+quant: same hidden/residual through all three sources
        hidden0 = torch.randn(n, h, device="cuda",
                              dtype=torch.bfloat16)
        residual0 = torch.randn(n, h, device="cuda",
                                dtype=torch.bfloat16)
        outs = {}
        for source in KERNEL_SOURCES:
            pipeline.kernel_source = source
            hidden = hidden0.clone()
            residual = residual0.clone()
            if source == "quail":
                q, s = pipeline.custom_norm_quant(
                    hidden, layer.input_layernorm, residual)
            else:
                q, s = pipeline.vllm_norm_quant(
                    hidden, layer.input_layernorm, residual)
            outs[source] = (_dequant(torch, q, s), residual)
        for source in ("vllm_ops", "vllm_compiled"):
            report[source]["norm_quant_max_abs"] = _max_abs(
                torch, outs[source][0], outs["quail"][0])
            report[source]["norm_residual_max_abs"] = _max_abs(
                torch, outs[source][1], outs["quail"][1])

        # silu+quant
        gate_up = torch.randn(n, gate_up_w, device="cuda",
                              dtype=torch.bfloat16)
        outs = {}
        for source in KERNEL_SOURCES:
            pipeline.kernel_source = source
            if source == "quail":
                q, s = pipeline.custom_silu_quant(gate_up)
            else:
                q, s = pipeline.vllm_silu_quant(gate_up)
            outs[source] = _dequant(torch, q, s)
        for source in ("vllm_ops", "vllm_compiled"):
            report[source]["silu_quant_max_abs"] = _max_abs(
                torch, outs[source], outs["quail"])

        # q/k norm + rope; the compiled source also reports compile
        # time and its kernel-launch count
        qkv = torch.randn(n, qkv_w, device="cuda",
                          dtype=torch.bfloat16)
        positions = torch.randint(0, 4096, (n,), device="cuda",
                                  dtype=torch.int64)
        outs = {}
        for source in KERNEL_SOURCES:
            pipeline.kernel_source = source
            t0 = time.perf_counter()
            if source == "quail":
                qo, ko = pipeline.custom_qk_norm_rope(
                    qkv.clone(), positions, attn)
            else:
                qo, ko = pipeline.vllm_qk_norm_rope(
                    qkv.clone(), positions, attn)
            torch.cuda.synchronize()
            if source == "vllm_compiled":
                report[source]["qk_first_call_s"] = round(
                    time.perf_counter() - t0, 2)
            outs[source] = (qo, ko)
        for source in ("vllm_ops", "vllm_compiled"):
            report[source]["qk_q_max_abs"] = _max_abs(
                torch, outs[source][0], outs["quail"][0])
            report[source]["qk_k_max_abs"] = _max_abs(
                torch, outs[source][1], outs["quail"][1])

        # count launches of each compiled segment at a second token
        # count (also proves dynamic shapes hold - no recompile);
        # add_norm should be one generated kernel with the residual
        # write folded in, not a separate copy
        pipeline.kernel_source = "vllm_compiled"

        def count_kernels(fn):
            fn()
            torch.cuda.synchronize()
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                fn()
                torch.cuda.synchronize()
            return [ev.key for ev in prof.key_averages()
                    if (getattr(ev, "self_device_time_total", 0)
                        or getattr(ev, "self_cuda_time_total", 0))
                    and not ev.key.startswith(("aten::", "_C::"))
                    and "Command Buffer" not in ev.key]

        qkv2 = torch.randn(2048, qkv_w, device="cuda",
                           dtype=torch.bfloat16)
        pos2 = positions[:2048]
        kernels = count_kernels(
            lambda: pipeline.vllm_qk_norm_rope(qkv2, pos2, attn))
        report["vllm_compiled"]["qk_kernel_launches"] = len(kernels)
        report["vllm_compiled"]["qk_kernels"] = kernels[:12]

        hidden2 = hidden0[:2048].clone()
        residual2 = residual0[:2048].clone()
        kernels = count_kernels(
            lambda: pipeline.vllm_norm_quant(
                hidden2, layer.input_layernorm, residual2))
        report["vllm_compiled"]["norm_quant_kernel_launches"] = \
            len(kernels)
        report["vllm_compiled"]["norm_quant_kernels"] = kernels[:12]

        # end to end on synthetic ids: a small filter chain and a
        # small join through every source; answers are the gate that
        # the merge override and the fused ops wire up correctly
        doc = [10 + (i % 500) for i in range(512)]
        docs = [doc] * 64
        # distinct questions per stage: identical questions would be
        # all shared preamble, which run_filter refuses
        questions = [list(range(10 + 20 * j, 26 + 20 * j))
                     for j in range(3)]
        for source in KERNEL_SOURCES:
            pipeline.kernel_source = source
            pipeline.attention_mode = "unified"
            answers, _, _ = run_filter(
                torch, arena, pipeline, async_ans, docs,
                questions, budget, arena_writes=True)
            report["filter_answers"][source] = sum(
                sum(v) for v in answers.values())
            pipeline.attention_mode = "merge_quant"
            answers, _, _ = run_join(
                torch, arena, pipeline, async_ans, docs[:8],
                [[questions[0]] * 24], budget)
            report["join_answers"][source] = sum(
                sum(v) for v in answers[0].values())

        # real text with planted flags: margins are wide, so a broken
        # unified path fails this while kernel rounding drift does not
        from corpus import build_corpus
        body_ids, q_ids, flags = build_corpus(tokenizer, 32)
        for source in KERNEL_SOURCES:
            pipeline.kernel_source = source
            pipeline.attention_mode = "unified"
            answers, _, _ = run_filter(
                torch, arena, pipeline, async_ans, body_ids,
                q_ids[:2], budget, arena_writes=True)
            report["filter_real"][source] = dict(
                answered=sum(len(v) for v in answers.values()),
                wrong=_wrong_count(answers, flags))

        # one paged unified chunk, tensor level: the final-position
        # hidden states each source produces, compared with quail's
        outs = {}
        for source in KERNEL_SOURCES:
            pipeline.kernel_source = source
            pipeline.attention_mode = "unified"
            for i in range(4):
                got = arena.activate(
                    ("probe", i), len(body_ids[i]),
                    capacity_tokens=len(body_ids[i]) + len(q_ids[0]))
                assert got is not None
            chunk = pack_chunk(
                torch, arena,
                [dict(key=("probe", i), prefix=body_ids[i],
                      f=len(body_ids[i]), suffixes=[q_ids[0]])
                 for i in range(4)],
                attention_mode="unified")
            outs[source] = pipeline.forward_chunk(chunk).clone()
            for i in range(4):
                arena.free_key(("probe", i))
        for source in ("vllm_ops", "vllm_compiled"):
            delta = (outs[source].to(torch.float32)
                     - outs["quail"].to(torch.float32))
            report["unified_chunk"][source] = dict(
                max_abs=float(delta.abs().max().item()),
                mean_abs=float(delta.abs().mean().item()),
                nans=int(torch.isnan(outs[source]).sum().item()),
                ref_mean_abs=float(
                    outs["quail"].to(torch.float32)
                    .abs().mean().item()))
    torch.cuda.synchronize()
    return _write(report, "kernel_source_probe")


# ----------------------------------------------- stock kernel inventory

@app.function(timeout=3600, **GPU_KW)
def stock_kernels(n_docs: int = 512) -> str:
    """Boot stock vLLM at its defaults, profile one prefill-heavy
    pass, and record which kernels its compiled graph actually runs
    between the GEMMs, plus the resolved compilation config. This is
    the ground truth the vllm_compiled rung mirrors."""
    import os
    import sys

    sys.path.insert(0, "/root/gpu_tests")

    # the v1 engine runs the model in a child process by default,
    # where this process's profiler cannot see the kernels
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import torch
    from vllm import LLM, SamplingParams

    from corpus import MODEL, build_corpus
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    body_ids, q_ids, _ = build_corpus(tokenizer, n_docs)
    prompts = [dict(prompt_token_ids=b + q_ids[0]) for b in body_ids]

    llm = LLM(model=MODEL, gpu_memory_utilization=0.92,
              enable_prefix_caching=False, disable_log_stats=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=1,
                              min_tokens=1)
    config = llm.llm_engine.vllm_config
    comp = config.compilation_config
    report = dict(
        cell="stock_kernels",
        model=MODEL,
        optimization_level=int(config.optimization_level),
        compilation_mode=str(comp.mode),
        custom_ops=list(comp.custom_ops),
        enabled_custom_ops=dict(comp.enabled_custom_ops),
        disabled_custom_ops=dict(comp.disabled_custom_ops),
        pass_config={
            k: bool(getattr(comp.pass_config, k))
            for k in ("fuse_norm_quant", "fuse_act_quant",
                      "fuse_attn_quant", "enable_qk_norm_rope_fusion")
            if getattr(comp.pass_config, k, None) is not None},
        use_deep_gemm_e8m0=None,
        kernels=[])
    try:
        from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used
        report["use_deep_gemm_e8m0"] = bool(is_deep_gemm_e8m0_used())
    except Exception as exc:
        report["use_deep_gemm_e8m0"] = repr(exc)

    llm.generate(prompts[:32], sampling)   # warm outside the profile
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        llm.generate(prompts, sampling)
    rows = []
    total_us = 0.0
    for ev in prof.key_averages():
        # keep only device-side rows: op wrapper rows (CPU device
        # type) repeat the kernel name with the same device time and
        # would double-count it
        if "CUDA" not in str(getattr(ev, "device_type", "")):
            continue
        cuda_us = getattr(ev, "self_device_time_total", 0) or \
            getattr(ev, "self_cuda_time_total", 0)
        if not cuda_us:
            continue
        total_us += cuda_us
        rows.append((round(cuda_us / 1e3, 2), ev.count, ev.key[:110]))
    rows.sort(reverse=True)
    report["cuda_busy_s"] = round(total_us / 1e6, 2)
    report["kernels"] = [dict(ms=ms, n=n, name=k)
                         for ms, n, k in rows[:60]]
    return _write(report, "kernel_source_stock")


@app.local_entrypoint()
def run_stock_kernels(n_docs: int = 512):
    handle = stock_kernels.spawn(n_docs)
    print(f"stock_kernels fc: {handle.object_id}", flush=True)
    print(handle.get())


# ------------------------------------------------------- measured cell

PREDICTIONS = {
    "filter": {
        "quail": "the reference: ~8.5 us/token, ~35 s "
                 "(results/attention_paths.json, unified)",
        "vllm_ops": "+2.3 to +2.6 us/token over quail (the A2-A3 "
                    "gap of the 2026-08-19 ablation was 2.4), "
                    "about 44-46 s",
        "vllm_compiled": "+1.7 to +2.2 us/token over quail. The "
                         "stock inventory shows the compiled graph "
                         "keeps all four standalone group-quant "
                         "launches per layer, so the quant-fusion "
                         "saving (2.34 us/token GPU in the "
                         "2026-08-19 profile) stays with quail; "
                         "against vllm_ops it recovers only the q/k "
                         "segment (5 launches to ~2, ~0.3-0.5 "
                         "us/token)",
    },
    "join": {
        "quail": "the reference: ~11 us/token "
                 "(results/join_attention_paths_packed 10x256 band)",
        "vllm_ops": "+2.5 to +3.0 us/token over quail: the filter "
                    "gap plus the unfused merge (gather, merge, "
                    "scatter, quant against our one kernel)",
        "vllm_compiled": "+2.0 to +2.7 us/token over quail: same "
                         "unfused quants and unfused merge as "
                         "vllm_ops, minus the q/k segment saving",
    },
}


@app.function(timeout=7200, **GPU_KW)
def queries(model: str = "qwen3-4b-fp8", n_docs: int = 10000,
            n_reports: int = 100, n_terms: int = 256,
            reps: int = 2) -> str:
    """Both queries through all three kernel sources, one container,
    one model load. Writes kernel_source_filter.json and
    kernel_source_join.json."""
    import sys
    import time

    sys.path.insert(0, "/root/gpu_tests")

    import torch

    from corpus import biodex_sample, build_corpus
    from quail.executor.loop import run_filter, run_join, warm_kernels

    (spec, tokenizer, arena, pipeline, async_ans, budget,
     arena_tok) = _boot(model)
    print(f"[kernel_source] predictions: {json.dumps(PREDICTIONS)}",
          flush=True)

    with torch.inference_mode():
        warm = warm_kernels(torch, arena, pipeline, async_ans, budget,
                            model_name=spec.hf_name)
    torch.cuda.synchronize()
    kernel_cache.commit()
    print(f"[kernel_source] warm: {warm}", flush=True)

    # ---- the filter query -------------------------------------------
    body_ids, q_ids, flags = build_corpus(tokenizer, n_docs)
    pipeline.attention_mode = "unified"
    filter_report = dict(
        cell="kernel_source_filter", model=spec.name, n_docs=n_docs,
        reps=reps, budget=budget, arena_tokens=arena_tok,
        attention_mode="unified", warm=warm,
        predictions=PREDICTIONS["filter"], runs={}, comparisons={})
    answers_by_source = {}
    for source in KERNEL_SOURCES:
        pipeline.kernel_source = source
        with torch.inference_mode():
            # per-source warm, unmeasured: first-call op init and,
            # for vllm_compiled, the torch.compile of the q/k segment
            run_filter(torch, arena, pipeline, async_ans,
                       body_ids[:256], q_ids, budget,
                       arena_writes=True)
            torch.cuda.synchronize()
            rows = []
            for rep in range(reps):
                timers = {}
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                answers, spans, tokens = run_filter(
                    torch, arena, pipeline, async_ans, body_ids,
                    q_ids, budget, timing=timers, arena_writes=True)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t0
                answered = sum(len(v) for v in answers.values())
                survivors = sum(len(row) == len(q_ids) and all(row)
                                for row in answers.values())
                row = dict(
                    source=source, rep=rep, wall=round(wall, 3),
                    fresh_tokens=tokens,
                    us_per_token=round(wall * 1e6 / tokens, 3),
                    answered=answered, survivors=survivors,
                    wrong=_wrong_count(answers, flags),
                    chunks=len(spans),
                    gpu_s=round(sum(e0.elapsed_time(e1)
                                    for _, e0, e1 in spans) / 1e3, 3),
                    peak_gib=round(
                        torch.cuda.max_memory_allocated() / 2**30, 2),
                    cpu_phase_s={k: round(v, 3) for k, v
                                 in sorted(timers.items())})
                rows.append(row)
                print(f"[kernel_source_filter] {row}", flush=True)
        filter_report["runs"][source] = rows
        answers_by_source[source] = answers

    reference = answers_by_source["quail"]
    last_quail = filter_report["runs"]["quail"][-1]
    for source in ("vllm_ops", "vllm_compiled"):
        last = filter_report["runs"][source][-1]
        filter_report["comparisons"][source] = dict(
            disagreements=_disagreements(
                reference, answers_by_source[source]),
            wall_delta_s=round(last["wall"] - last_quail["wall"], 3),
            us_per_token_delta=round(
                last["us_per_token"] - last_quail["us_per_token"], 3))
    if n_docs == 10000:
        # the banked counts are for the committed 10k corpus only
        filter_report["gates"] = dict(
            quail_vs_banked=dict(
                measured={k: last_quail[k] for k in BANKED_FILTER},
                banked=BANKED_FILTER),
            passed=all(last_quail[k] == v
                       for k, v in BANKED_FILTER.items()))
    _write(filter_report, "kernel_source_filter")

    # ---- the join query ---------------------------------------------
    data = biodex_sample(tokenizer, n_reports=n_reports)
    prefixes = data["prefixes"]
    suffixes = data["suffixes"][:n_terms]
    pipeline.attention_mode = "merge_quant"
    join_report = dict(
        cell="kernel_source_join", model=spec.name,
        n_reports=n_reports, n_terms=n_terms,
        pairs=n_reports * n_terms, reps=reps, budget=budget,
        attention_mode="merge_quant",
        predictions=PREDICTIONS["join"], runs={}, comparisons={})
    outputs = {}
    for source in KERNEL_SOURCES:
        pipeline.kernel_source = source
        with torch.inference_mode():
            run_join(torch, arena, pipeline, async_ans, prefixes,
                     [suffixes], budget)
            torch.cuda.synchronize()
            rows = []
            for rep in range(reps):
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                answers, spans, tokens = run_join(
                    torch, arena, pipeline, async_ans, prefixes,
                    [suffixes], budget)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t0
                flat = [bit for a in range(n_reports)
                        for bit in answers[0][a]]
                row = dict(
                    source=source, rep=rep, wall=round(wall, 3),
                    fresh_tokens=tokens,
                    us_per_token=round(wall * 1e6 / tokens, 3),
                    chunks=len(spans), yes=sum(flat),
                    peak_gib=round(
                        torch.cuda.max_memory_allocated() / 2**30, 2))
                rows.append(row)
                print(f"[kernel_source_join] {row}", flush=True)
        join_report["runs"][source] = rows
        outputs[source] = flat

    last_quail = join_report["runs"]["quail"][-1]
    for source in ("vllm_ops", "vllm_compiled"):
        last = join_report["runs"][source][-1]
        join_report["comparisons"][source] = dict(
            disagreements=sum(a != b for a, b in
                              zip(outputs["quail"], outputs[source])),
            wall_delta_s=round(last["wall"] - last_quail["wall"], 3),
            us_per_token_delta=round(
                last["us_per_token"] - last_quail["us_per_token"], 3))
    return _write(join_report, "kernel_source_join")


# ------------------------------------------------------------ profile

def _categorize(name):
    """Kernel name -> time bucket. Our Triton kernel names contain
    substrings that also appear in vLLM's op names, so the specific
    names are checked before the generic ones."""
    low = name.lower()
    if "deep_gemm" in low or "sm90_fp8" in low or "gemm" in low:
        return "gemm"
    if "merge_attn_states" in low:
        return "merge"
    if "flash" in low or "attn" in low:
        return "attention"
    if any(k in low for k in ("silu_mul_quant", "add_rms_norm_quant",
                              "qk_norm_rope", "merge_quant")):
        return "quail_fused"
    if any(k in low for k in ("per_block_quant",)):
        return "vllm_fused"
    if low.startswith("triton_"):
        return "inductor"
    if any(k in low for k in ("rms_norm", "rotary", "silu_and_mul")):
        return "vllm_elementwise"
    if "quant" in low:
        return "quant"
    if any(k in low for k in ("memcpy", "copy", "index", "cat",
                              "gather", "scatter", "nonzero")):
        return "copies"
    return "other"


@app.function(timeout=3600, **GPU_KW)
def profile_filter(model: str = "qwen3-4b-fp8",
                   n_docs: int = 3000) -> str:
    """Per-kernel-category GPU time for the three sources on the
    filter query."""
    import sys

    sys.path.insert(0, "/root/gpu_tests")

    import torch

    from corpus import build_corpus
    from quail.executor.loop import run_filter, warm_kernels

    (spec, tokenizer, arena, pipeline, async_ans, budget,
     _) = _boot(model)
    body_ids, q_ids, _ = build_corpus(tokenizer, n_docs)
    pipeline.attention_mode = "unified"
    with torch.inference_mode():
        warm_kernels(torch, arena, pipeline, async_ans, budget,
                     model_name=spec.hf_name)
    torch.cuda.synchronize()
    kernel_cache.commit()

    result = dict(cell="kernel_source_profile", model=spec.name,
                  n_docs=n_docs, sources={})
    for source in KERNEL_SOURCES:
        pipeline.kernel_source = source
        with torch.inference_mode():
            _, _, tokens = run_filter(torch, arena, pipeline,
                                      async_ans, body_ids, q_ids,
                                      budget, arena_writes=True)
            torch.cuda.synchronize()
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                run_filter(torch, arena, pipeline, async_ans,
                           body_ids, q_ids, budget, arena_writes=True)
                torch.cuda.synchronize()
        cats, counts = {}, {}
        rows = []
        for ev in prof.key_averages():
            # keep only device-side rows: op wrapper rows (CPU
            # device type) repeat the kernel name with the same
            # device time and would double-count it
            if "CUDA" not in str(getattr(ev, "device_type", "")):
                continue
            cuda_us = getattr(ev, "self_device_time_total", 0) or \
                getattr(ev, "self_cuda_time_total", 0)
            if not cuda_us:
                continue
            cat = _categorize(ev.key)
            cats[cat] = cats.get(cat, 0.0) + cuda_us
            counts[cat] = counts.get(cat, 0) + ev.count
            rows.append((round(cuda_us / 1e6, 3), ev.count,
                         ev.key[:90]))
        rows.sort(reverse=True)
        busy = sum(cats.values())
        result["sources"][source] = dict(
            fresh_tokens=tokens,
            cuda_busy_s=round(busy / 1e6, 2),
            us_per_token=round(busy / tokens, 2),
            category_us_per_token={k: round(v / tokens, 3)
                                   for k, v in sorted(cats.items())},
            category_launches={k: counts[k] for k in sorted(counts)},
            top_kernels=[dict(s=s, n=n, name=k) for s, n, k
                         in rows[:20]])
        print(f"[kernel_source_profile] {source}: "
              f"{json.dumps(result['sources'][source]['category_us_per_token'])}",
              flush=True)
    return _write(result, "kernel_source_profile")


# ---------------------------------------------------------- entrypoints

@app.local_entrypoint()
def run_probe(model: str = "qwen3-4b-fp8"):
    handle = probe.spawn(model)
    print(f"probe fc: {handle.object_id}", flush=True)
    print(handle.get())


@app.local_entrypoint()
def run_queries(model: str = "qwen3-4b-fp8", n_docs: int = 10000,
                n_reports: int = 100, n_terms: int = 256,
                reps: int = 2):
    handle = queries.spawn(model, n_docs, n_reports, n_terms, reps)
    print(f"queries fc: {handle.object_id}", flush=True)
    print(handle.get())


@app.local_entrypoint()
def run_profile(model: str = "qwen3-4b-fp8", n_docs: int = 3000):
    handle = profile_filter.spawn(model, n_docs)
    print(f"profile_filter fc: {handle.object_id}", flush=True)
    print(handle.get())
