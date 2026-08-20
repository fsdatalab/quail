"""Boot-time breakdown: Quail vs stock vLLM, cold and warm.

Quail's worker boot (load_model, arena, pipeline, warm_kernels) and
stock's LLM(...) constructor are timed side-by-side on H100 SXM with
the same vLLM pin (0.26.0). Each side runs `trials` independent
containers (default 3) so cold boots are real process starts; each
container then records one warm reuse. The summary reports mean and
median per phase across trials.

PREDICTION (stated before the run): Quail cold is dominated by
load_model_s when the kernel-cache volume is warm, and by
warm_kernels_s on a cold kernel cache; arena_s and pipeline_s are
small. Stock cold is weight load plus KV-cache profiling inside
LLM(...). Both warm boots are near 0 (Quail hits _BOOTED /
warmed=True; stock keeps the LLM instance).

Run from the quail/ directory (tee per house rule):

    uv run modal run tests/gpu/boot_profile.py \\
        2>&1 | tee results/boot_profile.log
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

from baselines.boot_stats import aggregate as _aggregate
from corpus import MODEL

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub", "pandas", "pyarrow",
                 "numpy", "datasets")
    .env({# vLLM's architecture-inspection subprocess caches under
          # VLLM_CACHE_ROOT/modelinfos; the volume makes it once ever,
          # not once per container (~11-13 s saved per cold boot)
          "VLLM_CACHE_ROOT": "/root/.cache/kernels/vllm",
          "VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "TRITON_CACHE_DIR": "/root/.cache/kernels/triton"})
    .add_local_python_source("quail", "corpus", "baselines")
    .add_local_dir("quail/calibration",
                   remote_path="/root/quail/calibration")
)

# House rule: attach to the existing milestone1 app; never invent a
# new Modal app name.
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

PREDICTION = (
    "Quail cold: load_model_s dominates when kernels are cached; "
    "warm_kernels_s dominates on a cold kernel cache; arena + "
    "pipeline small. Stock cold: LLM(...) = weights + KV profiling. "
    "Both warm ≈ 0."
)


def _round_boot(boot: dict) -> dict:
    out = dict(boot)
    for k, v in list(out.items()):
        if isinstance(v, float):
            out[k] = round(v, 2)
    return out


def _quail_boot_once(docs, warm_q, *, reuse: dict | None) -> tuple[dict, dict]:
    """One Quail boot. reuse=None is cold; reuse=state is warm skip."""
    import torch
    import torch.nn.functional as F

    from quail.executor.arena import KVArena
    from quail.executor.attention import Pipeline
    from quail.executor.loop import Answerer, AsyncAnswers, warm_kernels
    from quail.executor.model import load_model
    from quail.planner import budgets
    from quail.specs import H100_SXM, QWEN3_4B_FP8
    from transformers import AutoTokenizer

    boot = dict(kind="warm", load_model_s=0.0, arena_s=0.0,
                pipeline_s=0.0, warm_kernels_s=0.0)
    t_boot = time.perf_counter()
    spec = QWEN3_4B_FP8
    device = H100_SXM

    if reuse is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        t0 = time.perf_counter()
        model = load_model(MODEL, revision=spec.revision)
        boot["load_model_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        chunk = budgets.chunk_budget(spec, device)
        arena_tok = budgets.arena_tokens(spec, device, chunk)
        arena = KVArena(n_layers=spec.layers,
                        n_pages=arena_tok // budgets.PAGE_TOKENS,
                        page_tokens=budgets.PAGE_TOKENS,
                        n_kv=spec.n_kv, d_head=spec.d_head,
                        dtype=torch.bfloat16)
        boot["arena_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        pipeline = Pipeline(model, arena)
        boot["pipeline_s"] = time.perf_counter() - t0
        answerer = Answerer(torch, F, model, tokenizer)
        async_ans = AsyncAnswers(torch, answerer)
        budget = min(chunk, pipeline.max_chunk_tokens)
        state = dict(torch=torch, model=model, arena=arena,
                     pipeline=pipeline, async_ans=async_ans,
                     budget=budget, warmed=False)
        boot["kind"] = "cold"
    else:
        state = reuse
        torch = state["torch"]
        arena = state["arena"]
        pipeline = state["pipeline"]
        async_ans = state["async_ans"]
        budget = state["budget"]

    if not state["warmed"]:
        t0 = time.perf_counter()
        with torch.inference_mode():
            warm_kernels(torch, arena, pipeline, async_ans,
                         docs, [warm_q], budget)
        torch.cuda.synchronize()
        kernel_cache.commit()
        boot["warm_kernels_s"] = time.perf_counter() - t0
        state["warmed"] = True
        boot["kind"] = "cold"

    boot["boot_s"] = time.perf_counter() - t_boot
    return state, _round_boot(boot)


@app.function(timeout=3600, max_containers=8, **GPU_KW)
def quail_boot_trial(trial: int = 0) -> dict:
    """One container: cold Quail boot, then warm reuse."""
    from corpus import build_corpus
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # Small corpus: boot warmup only needs enough docs to fill one
    # budget-sized chunk; full 10k is wasted work for this cell.
    body_ids, q_ids, _flags = build_corpus(tokenizer, 256)
    warm_q = q_ids[0]

    state, cold = _quail_boot_once(body_ids, warm_q, reuse=None)
    _, warm = _quail_boot_once(body_ids, warm_q, reuse=state)
    row = dict(side="quail", trial=trial, cold=cold, warm=warm)
    print(f"[boot_profile] quail trial {trial}: cold={cold} warm={warm}",
          flush=True)
    return row


@app.function(timeout=3600, max_containers=8, **GPU_KW)
def stock_boot_trial(trial: int = 0) -> dict:
    """One container: cold LLM(...), then warm reuse dict."""
    from baselines.stock_boot import time_llm_boot, warm_boot_dict

    # Same knobs as the committed bf16 stock filter baseline.
    llm, cold = time_llm_boot(
        model=MODEL, max_num_batched_tokens=25_305,
        max_num_seqs=2648, gpu_memory_utilization=0.92,
        enable_prefix_caching=True, disable_log_stats=True)
    # Touch the engine so construction is fully settled before the
    # warm measurement (kept instance, no re-init).
    _ = llm.llm_engine
    warm = warm_boot_dict()
    row = dict(side="stock", trial=trial, cold=cold, warm=warm)
    print(f"[boot_profile] stock trial {trial}: cold={cold} warm={warm}",
          flush=True)
    return row


@app.function(timeout=600, image=image, memory=4096,
              volumes={"/results": results_vol})
def write_boot_report(report: dict) -> str:
    """Persist per-side and merged compare JSON on the results volume."""
    os.makedirs("/results/boot", exist_ok=True)
    for name, key in (("quail", "quail"), ("stock", "stock"),
                      ("compare", None)):
        payload = report if key is None else report[key]
        with open(f"/results/boot/{name}.json", "w") as f:
            json.dump(payload, f, indent=2)
    results_vol.commit()
    return json.dumps(report, indent=2)


@app.local_entrypoint()
def main(trials: int = 3, out: str = "results/boot_profile.json"):
    """Spawn `trials` cold containers per side, aggregate mean/median."""
    print(f"[boot_profile] prediction: {PREDICTION}", flush=True)
    print(f"[boot_profile] trials={trials} (mean + median)", flush=True)

    q_handles = [quail_boot_trial.spawn(i) for i in range(trials)]
    s_handles = [stock_boot_trial.spawn(i) for i in range(trials)]
    quail_trials = [h.get() for h in q_handles]
    stock_trials = [h.get() for h in s_handles]

    report = dict(
        cell="boot_profile",
        model=MODEL,
        gpu="H100!",
        vllm="0.26.0",
        prediction=PREDICTION,
        quail=_aggregate("quail", quail_trials),
        stock=_aggregate("stock", stock_trials),
    )
    payload = write_boot_report.remote(report)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload,
                                                               indent=2)
    Path(out).write_text(text)
    print(f"[boot_profile] saved {out}", flush=True)

    parsed = json.loads(text)
    for side in ("quail", "stock"):
        for phase in ("cold", "warm"):
            boot = parsed[side][phase].get("boot_s", {})
            print(f"[boot_profile] {side} {phase} boot_s "
                  f"mean={boot.get('mean')} median={boot.get('median')} "
                  f"trials={boot.get('trials')}", flush=True)
