"""Boot-time breakdown: Quail vs stock vLLM, cold and warm.

Quail's worker boot (load_model, arena, pipeline) and stock's
LLM(...) constructor are timed side-by-side on H100 SXM with the
same vLLM pin (0.26.0). There is no kernel warmup: the first real
chunk loads cubins from the volume. Each side runs `trials`
independent containers (default 3) so cold boots are real process
starts; each container then records one warm reuse. The summary
reports mean and median per phase across trials.

PREDICTION (stated before the run): Quail cold is dominated by
load_model_s (30-38 s). arena_s and pipeline_s are small. Stock
cold is weight load plus KV-cache profiling inside LLM(...)
(~250 s). Both warm boots are near 0 (Quail hits _BOOTED; stock
keeps the LLM instance).

Run from the quail/ directory (tee per house rule):

    uv run modal run tests/gpu/boot_profile.py \\
        2>&1 | tee results/boot_profile.log
"""

from __future__ import annotations

import json
import os
import time

import modal

from baselines.boot_stats import aggregate as _aggregate
from corpus import MODEL

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
    "Quail cold: load_model_s 30-38 s; arena + pipeline small. "
    "No kernel warmup. Stock cold: LLM(...) = weights + KV "
    "profiling (~250 s). Both warm ≈ 0."
)


def _round_boot(boot: dict) -> dict:
    out = dict(boot)
    for k, v in list(out.items()):
        if isinstance(v, float):
            out[k] = round(v, 2)
    return out


def _quail_boot_once(*, reuse: dict | None) -> tuple[dict, dict]:
    """One Quail boot. reuse=None is cold; reuse=state is warm skip."""
    import torch
    import torch.nn.functional as F

    from quail.executor.arena import KVArena
    from quail.executor.attention import (FILTER_ATTENTION,
                                          Pipeline)
    from quail.executor.loop import Answerer, AsyncAnswers
    from quail.executor.model import load_model
    from quail.planner import budgets
    from quail.specs import H100_SXM, QWEN3_4B_FP8
    from transformers import AutoTokenizer

    boot = dict(kind="warm", load_model_s=0.0, arena_s=0.0,
                pipeline_s=0.0)
    t_boot = time.perf_counter()
    spec = QWEN3_4B_FP8
    device = H100_SXM

    if reuse is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        t0 = time.perf_counter()
        model = load_model(MODEL)
        boot["load_model_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        chunk_tokens = budgets.chunk_budget(spec, device)
        arena_tok = budgets.arena_tokens(spec, device, chunk_tokens)
        arena = KVArena(n_layers=spec.layers,
                        n_pages=arena_tok // budgets.PAGE_TOKENS,
                        page_tokens=budgets.PAGE_TOKENS,
                        n_kv=spec.n_kv, d_head=spec.d_head,
                        dtype=torch.bfloat16)
        boot["arena_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        pipeline = Pipeline(
            model, arena, attention_mode=FILTER_ATTENTION)
        boot["pipeline_s"] = time.perf_counter() - t0
        answerer = Answerer(torch, F, model, tokenizer)
        async_ans = AsyncAnswers(torch, answerer)
        chunk_tokens = min(chunk_tokens, pipeline.max_chunk_tokens)
        state = dict(torch=torch, model=model, arena=arena,
                     pipeline=pipeline, async_ans=async_ans,
                     chunk_tokens=chunk_tokens)
        boot["kind"] = "cold"
    else:
        state = reuse

    boot["boot_s"] = time.perf_counter() - t_boot
    return state, _round_boot(boot)


@app.function(timeout=3600, max_containers=8, **GPU_KW)
def quail_boot_trial(trial: int = 0) -> dict:
    """One container: cold Quail boot, then warm reuse."""
    state, cold = _quail_boot_once(reuse=None)
    _, warm = _quail_boot_once(reuse=state)
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
    tag = "compare"
    with open(f"/results/boot/{tag}.json", "w") as f:
        json.dump(report, f, indent=2)
    with open("/results/boot/quail.json", "w") as f:
        json.dump(report["quail"], f, indent=2)
    if "stock" in report:
        with open("/results/boot/stock.json", "w") as f:
            json.dump(report["stock"], f, indent=2)
    results_vol.commit()
    return json.dumps(report, indent=2)


@app.local_entrypoint()
def main(trials: int = 3, out: str = "", no_stock: bool = False):
    """Spawn `trials` cold containers per side, aggregate mean/median.

    Does not write Modal return values to a local JSON file. Prints
    each fc- id. The volume record is /results/boot/*.json. Pass
    --no-stock to time Quail only."""
    print(f"[boot_profile] prediction: {PREDICTION}", flush=True)
    print(f"[boot_profile] trials={trials} stock={not no_stock}",
          flush=True)

    q_handles = [quail_boot_trial.spawn(i) for i in range(trials)]
    s_handles = ([] if no_stock
                 else [stock_boot_trial.spawn(i) for i in range(trials)])
    for h in q_handles + s_handles:
        print(f"[boot_profile] fc={h.object_id}", flush=True)
    quail_trials = [h.get() for h in q_handles]
    stock_trials = [h.get() for h in s_handles]

    report = dict(
        cell="boot_profile",
        model=MODEL,
        gpu="H100!",
        vllm="0.26.0",
        prediction=PREDICTION,
        quail=_aggregate("quail", quail_trials),
    )
    if stock_trials:
        report["stock"] = _aggregate("stock", stock_trials)
    remote = write_boot_report.remote(report)
    print(f"[boot_profile] volume=/results/boot "
          f"write_fc={getattr(remote, 'object_id', '')}", flush=True)
    text = remote if isinstance(remote, str) else json.dumps(remote,
                                                             indent=2)
    parsed = json.loads(text)
    for side in ("quail", "stock"):
        if side not in parsed:
            continue
        for phase in ("cold", "warm"):
            boot = parsed[side][phase].get("boot_s", {})
            print(f"[boot_profile] {side} {phase} boot_s "
                  f"mean={boot.get('mean')} median={boot.get('median')} "
                  f"trials={boot.get('trials')}", flush=True)
