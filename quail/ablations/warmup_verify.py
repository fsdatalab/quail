"""Verify the run_join warmup fix (issue #25): measure boot time and
BIO-1 us/token with the updated warm_kernels that exercises the join
execution path.

This is a one-shot verification, not a sweep. It boots once on 4B,
runs warm_kernels (which now includes run_join), then runs a single
BIO-1 filter query and prints both numbers for comparison against the
baseline (boot_s=67.62, BIO-1=10.33 us/token from the packing sweep).

    uv run modal run ablations/warmup_verify.py \
        2>&1 | tee results/warmup_verify_4b.log
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


@app.function(**GPU_KW, timeout=1800)
def verify(model_name: str) -> str:
    import socket
    import time

    from quail.executor.attention import FILTER_ATTENTION
    from quail.executor.loop import run_filter, warm_kernels
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
    with torch.inference_mode():
        warm_kernels(torch, arena, pipeline, async_ans, [doc] * 64,
                     [qsuf], budget)
    torch.cuda.synchronize()
    boot_s = round(time.perf_counter() - t0, 2)
    print(f"[verify] boot_s={boot_s} (baseline: 67.62)", flush=True)

    # BIO-1: filter F7 over 200 BioDEX reports
    import re

    from quail.bench.quailb import F7, _biodex_rows
    from quail.logical import SHARED_PRE, ColumnRef, bind_prompt

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

    chunks = [dict(tokens=tr["tokens"],
                   gpu_ms=round(e0.elapsed_time(e1), 3))
              for (_, e0, e1), tr in zip(spans, trace)]
    gpu_s = sum(c["gpu_ms"] for c in chunks) / 1e3
    us_wall = round(wall / tokens * 1e6, 3)
    us_gpu = round(gpu_s / tokens * 1e6, 3)

    result = dict(
        model=model_name, boot_s=boot_s,
        host=socket.gethostname(),
        gpu=torch.cuda.get_device_name(),
        bio1=dict(chunks=len(chunks), fresh_tokens=tokens,
                  wall_s=round(wall, 2), gpu_s=round(gpu_s, 2),
                  us_per_token_wall=us_wall,
                  us_per_token_gpu=us_gpu))

    print(f"[verify] BIO-1: {us_gpu} us/tok GPU, {us_wall} us/tok wall "
          f"(baseline: 10.33 us/tok GPU)", flush=True)

    kernel_cache.commit()
    return json.dumps(result)


@app.local_entrypoint()
def run(model: str = "qwen3-4b-fp8"):
    call = verify.spawn(model)
    print(f"[verify] fc={call.object_id}", flush=True)
    print(call.get(), flush=True)
