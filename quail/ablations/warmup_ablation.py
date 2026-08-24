"""Warmup ablation: which forward-pass warmup steps are necessary?

After the DeepGEMM linear-layer sweep, warm_kernels runs three kinds
of forward pass. This cell tests which of them actually matter by
dropping each one and measuring whether the first workload chunk
picks up a stall.

Configurations (all include the GEMM sweep):
  A  gemm-only:      no forward-pass warmup at all
  B  filter-both:    run_filter in both attention modes + tiny ladder
  C  filter+join:    B + run_join
  D  filter+fast:    B + fast-path run_filter (arena_writes=False)
  E  full:           B + run_join + fast-path (the current code)

Each config runs in its own container (compiled kernels persist in a
process). Workloads: BIO-1 (filter, arena_writes=False) and a small
join (4 anchor docs, 8 suffixes per anchor).

    uv run modal run ablations/warmup_ablation.py \
        2>&1 | tee results/warmup_ablation.log
"""

import json
import os

import modal

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
          "TRITON_CACHE_DIR": "/root/.cache/kernels/triton"})
    .add_local_python_source("quail", "baselines")
)

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

CONFIGS = {
    "A_gemm_only":    dict(filter_both=False, run_join=False, fast=False),
    "B_filter_both":  dict(filter_both=True,  run_join=False, fast=False),
    "C_filter_join":  dict(filter_both=True,  run_join=True,  fast=False),
    "D_filter_fast":  dict(filter_both=True,  run_join=False, fast=True),
    "E_full":         dict(filter_both=True,  run_join=True,  fast=True),
}


@app.function(**GPU_KW, timeout=1800)
def ablation(model_name: str, config_name: str,
             filter_both: bool, do_run_join: bool, fast: bool) -> str:
    import re
    import time

    from quail.bench.quailb import F7, _biodex_rows
    from quail.executor.attention import (FILTER_ATTENTION,
                                          JOIN_ATTENTION)
    from quail.executor.loop import (TINY_WARM_TOKENS, run_filter,
                                     run_join)
    from quail.logical import SHARED_PRE, ColumnRef, bind_prompt
    from quail.planner.calibrate import _boot, resolve_pair

    spec, device = resolve_pair(model_name, "h100-sxm")
    t0 = time.perf_counter()
    (torch, tokenizer, pipeline, arena, async_ans,
     budget) = _boot(spec, device)

    def tok(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    filler = tok("The document discusses a clinical finding. ")
    doc = (filler * ((512 // len(filler)) + 1))[:512]
    qsuf = tok("\n\nAnswer TRUE or FALSE.\nANSWER:")

    # --- GEMM sweep (always) ---
    layer = pipeline.layers[0]
    linears = (layer.self_attn.qkv_proj, layer.self_attn.o_proj,
               layer.mlp.gate_up_proj, layer.mlp.down_proj)
    coarse = sorted({m for m in range(4096, 32769, 1024)}
                    | {m for m in range(32768, budget + 1, 2048)}
                    | {budget})
    fallback_small = list(range(64, 4097, 256))

    def small_sizes(lin):
        try:
            from vllm.model_executor.warmup.deep_gemm_warmup import (
                _generate_optimal_warmup_m_values)
            return _generate_optimal_warmup_m_values(
                4096, lin.weight.shape[0], torch.device("cuda"))
        except Exception:
            return fallback_small

    work = [(m, lin) for lin in linears
            for m in small_sizes(lin) + coarse]
    with torch.inference_mode():
        cur, buf = None, None
        for m, lin in work:
            if lin is not cur:
                buf = torch.randn(budget, lin.weight.shape[1],
                                  device="cuda", dtype=torch.bfloat16)
                cur = lin
            q, s = pipeline.quant(buf[:m])
            pipeline.gemm(q, s, lin)
        buf = None
        torch.cuda.synchronize()

    # --- build warm docs ---
    q_max = len(qsuf)
    warm_docs, used = [], 0
    pool = [doc] * 64
    for d in pool:
        if used + len(d) + q_max > budget:
            break
        warm_docs.append(d)
        used += len(d) + q_max
    stream = warm_docs[0] if warm_docs else [0]

    # --- forward-pass warmup steps ---
    original_mode = pipeline.attention_mode
    with torch.inference_mode():
        if filter_both:
            for mode in (FILTER_ATTENTION, JOIN_ATTENTION):
                pipeline.attention_mode = mode
                run_filter(torch, arena, pipeline, async_ans, warm_docs,
                           [qsuf], budget, arena_writes=True)
                for t in TINY_WARM_TOKENS:
                    if t >= budget:
                        continue
                    body = (stream * (t // len(stream) + 1)
                            )[:max(8, t - q_max)]
                    run_filter(torch, arena, pipeline, async_ans, [body],
                               [qsuf], budget, arena_writes=True)
        if do_run_join:
            pipeline.attention_mode = JOIN_ATTENTION
            run_join(torch, arena, pipeline, async_ans, warm_docs,
                     [[qsuf] * max(1, 8 // 1)], budget)
        pipeline.attention_mode = original_mode
        if fast:
            run_filter(torch, arena, pipeline, async_ans, warm_docs,
                       [qsuf], budget, arena_writes=False)
    torch.cuda.synchronize()
    warmup_s = round(time.perf_counter() - t0, 2)
    print(f"[{config_name}] warmup_s={warmup_s}", flush=True)

    # --- workload 1: BIO-1 filter ---
    bio = _biodex_rows(200)
    reports = [t for t, _ in bio]
    pre = tok(SHARED_PRE)
    p = bind_prompt(F7, (ColumnRef("r", "r", "report"),), tok)
    qids = tok(re.sub(r"\{\d+\}", "", p.tail))
    bodies = [pre + tok(t) for t in reports]

    trace = []
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.inference_mode():
        pipeline.attention_mode = FILTER_ATTENTION
        _, spans, tokens = run_filter(
            torch, arena, pipeline, async_ans, bodies,
            [qids], budget, trace=trace, arena_writes=False)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t1
    chunk_ms = [round(e0.elapsed_time(e1), 2)
                for _, e0, e1 in spans]
    gpu_s = sum(chunk_ms) / 1e3
    bio1 = dict(
        us_gpu=round(gpu_s / tokens * 1e6, 3),
        us_wall=round(wall / tokens * 1e6, 3),
        chunks=len(chunk_ms),
        chunk_ms=chunk_ms,
        tokens=tokens)
    print(f"[{config_name}] BIO-1: {bio1['us_gpu']} us/tok GPU, "
          f"chunks={chunk_ms}", flush=True)

    # --- workload 2: small join (4 anchors x 8 suffixes) ---
    anchors = [pre + tok(t) for t in reports[:4]]
    suffixes = [tok("\n\nAnswer TRUE or FALSE.\nANSWER:")]
    suf_list = suffixes * 8

    trace2 = []
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    with torch.inference_mode():
        pipeline.attention_mode = JOIN_ATTENTION
        _, spans2, tokens2 = run_join(
            torch, arena, pipeline, async_ans, anchors,
            [suf_list], budget, trace=trace2)
    torch.cuda.synchronize()
    wall2 = time.perf_counter() - t2
    chunk_ms2 = [round(e0.elapsed_time(e1), 2)
                 for _, e0, e1 in spans2]
    gpu_s2 = sum(chunk_ms2) / 1e3
    join = dict(
        us_gpu=round(gpu_s2 / tokens2 * 1e6, 3),
        us_wall=round(wall2 / tokens2 * 1e6, 3),
        chunks=len(chunk_ms2),
        chunk_ms=chunk_ms2,
        tokens=tokens2)
    print(f"[{config_name}] join: {join['us_gpu']} us/tok GPU, "
          f"chunks={chunk_ms2}", flush=True)

    return json.dumps(dict(
        config=config_name, model=model_name,
        warmup_s=warmup_s, bio1=bio1, join=join))


@app.local_entrypoint()
def run(model: str = "qwen3-4b-fp8"):
    calls = []
    for name, cfg in CONFIGS.items():
        c = ablation.spawn(model, name,
                           cfg["filter_both"], cfg["run_join"],
                           cfg["fast"])
        print(f"[ablation] {name} fc={c.object_id}", flush=True)
        calls.append((name, c))
    for name, c in calls:
        print(f"[ablation] {name}: {c.get()}", flush=True)
