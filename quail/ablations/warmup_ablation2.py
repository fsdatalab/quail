"""Warmup ablation round 2: wider workloads.

Round 1 tested BIO-1 (8 large chunks) and a tiny join (1 chunk).
This round adds IMDB-1 (many short docs per chunk) and IMDB-2
(real join: 5000 reviews x 12 aspects, many chunks with many
suffixes).

Only configs A (GEMM only) and B (filter both modes) are tested —
round 1 showed run_join and fast-path add nothing over B for the
narrow workloads. If B still matches full warmup on these wider
workloads, those steps are genuinely redundant.

    uv run modal run ablations/warmup_ablation2.py \
        2>&1 | tee results/warmup_ablation2.log
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
    "A_gemm_only":   dict(filter_both=False),
    "B_filter_both": dict(filter_both=True),
}


@app.function(**GPU_KW, timeout=1800)
def ablation(model_name: str, config_name: str,
             filter_both: bool) -> str:
    import re
    import time

    from quail.bench.quailb import (ASPECTS, DISCUSS_ASPECT, F1, F7,
                                    _biodex_rows, _imdb_pool)
    from quail.executor.attention import (FILTER_ATTENTION,
                                          JOIN_ATTENTION)
    from quail.executor.loop import (TINY_WARM_TOKENS, run_filter,
                                     run_join)
    from quail.logical import (SHARED_PRE, ColumnRef, bind_join_prompt,
                               bind_prompt, join_anchor_note,
                               join_label, render_join_question)
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

    # --- forward-pass warmup ---
    q_max = len(qsuf)
    warm_docs, used = [], 0
    pool = [doc] * 64
    for d in pool:
        if used + len(d) + q_max > budget:
            break
        warm_docs.append(d)
        used += len(d) + q_max
    stream = warm_docs[0] if warm_docs else [0]
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
    pipeline.attention_mode = original_mode
    torch.cuda.synchronize()
    warmup_s = round(time.perf_counter() - t0, 2)
    print(f"[{config_name}] warmup_s={warmup_s}", flush=True)

    pre = tok(SHARED_PRE)
    results = dict(config=config_name, model=model_name,
                   warmup_s=warmup_s, workloads={})

    def run_workload(name, run_fn):
        trace = []
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.inference_mode():
            _, spans, tokens = run_fn(trace)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t1
        chunk_ms = [round(e0.elapsed_time(e1), 2)
                    for _, e0, e1 in spans]
        gpu_s = sum(chunk_ms) / 1e3
        r = dict(us_gpu=round(gpu_s / tokens * 1e6, 3),
                 us_wall=round(wall / tokens * 1e6, 3),
                 chunks=len(chunk_ms),
                 first_chunk_ms=chunk_ms[0] if chunk_ms else 0,
                 chunk_ms=chunk_ms, tokens=tokens)
        results["workloads"][name] = r
        print(f"[{config_name}] {name}: {r['us_gpu']} us/tok GPU, "
              f"first={r['first_chunk_ms']}ms, {len(chunk_ms)} chunks",
              flush=True)

    # --- BIO-1: filter, long docs ---
    bio = _biodex_rows(200)
    reports = [t for t, _ in bio]
    p_bio = bind_prompt(F7, (ColumnRef("r", "r", "report"),), tok)
    qids_bio = tok(re.sub(r"\{\d+\}", "", p_bio.tail))
    bodies_bio = [pre + tok(t) for t in reports]

    def bio1_fn(trace):
        pipeline.attention_mode = FILTER_ATTENTION
        return run_filter(torch, arena, pipeline, async_ans, bodies_bio,
                          [qids_bio], budget, trace=trace,
                          arena_writes=False)
    run_workload("BIO-1", bio1_fn)

    # --- IMDB-1: filter, many short docs ---
    reviews = _imdb_pool()[:5_000]
    p_imdb = bind_prompt(F1, (ColumnRef("r", "r", "body"),), tok)
    qids_imdb = tok(re.sub(r"\{\d+\}", "", p_imdb.tail))
    bodies_imdb = [pre + tok(t) for t in reviews]

    def imdb1_fn(trace):
        pipeline.attention_mode = FILTER_ATTENTION
        return run_filter(torch, arena, pipeline, async_ans, bodies_imdb,
                          [qids_imdb], budget, trace=trace,
                          arena_writes=False)
    run_workload("IMDB-1", imdb1_fn)

    # --- IMDB-2: real join, 5000 reviews x 12 aspects ---
    cols = (("r", "body"), ("a", "aspect"))
    args = tuple(ColumnRef(a, a, c) for a, c in cols)
    bind_join_prompt(DISCUSS_ASPECT, args, tok)
    label = tok(join_label(1))
    tail = tok(render_join_question(DISCUSS_ASPECT))
    frame = tok(join_anchor_note(0))
    prefixes = [pre + tok(t) for t in reviews]
    suffixes = [label + tok(a) + tail for a in ASPECTS]

    def imdb2_fn(trace):
        pipeline.attention_mode = JOIN_ATTENTION
        return run_join(torch, arena, pipeline, async_ans, prefixes,
                        [suffixes], budget, stage_frames=[frame],
                        trace=trace)
    run_workload("IMDB-2", imdb2_fn)

    return json.dumps(results)


@app.local_entrypoint()
def run(model: str = "qwen3-4b-fp8"):
    calls = []
    for name, cfg in CONFIGS.items():
        c = ablation.spawn(model, name, cfg["filter_both"])
        print(f"[ablation2] {name} fc={c.object_id}", flush=True)
        calls.append((name, c))
    for name, c in calls:
        print(f"[ablation2] {name}: {c.get()}", flush=True)
