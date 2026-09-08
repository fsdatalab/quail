"""Cold load_model, phase by phase: current code vs the two fixes.

The two fixes, measured by Charles Frye on the pre-restructure tree
(branch charlesfrye/faster-boot) and ported here:

- VLLM_CACHE_ROOT rides the kernel-cache volume. vLLM resolves the
  model architecture in a fresh Python subprocess at engine-config
  creation and caches the result under VLLM_CACHE_ROOT/modelinfos.
  The default ~/.cache/vllm is ephemeral on Modal, so every container
  paid the subprocess again (~13 s). On the volume it runs once ever.
- The hub revision is pinned to a commit hash (ModelSpec.revision).
  A commit hash resolves from the local HF cache without the API
  round trips a branch name pays, and still downloads on a cold
  cache. Offline mode was rejected because a missing snapshot must
  download, not fail.

Two probe functions on the same image, one cold container per call
(max_inputs=1):

- probe_main: the pre-fix behavior. Pops VLLM_CACHE_ROOT before any
  vllm import (falling back to the ephemeral default) and passes no
  revision.
- probe_pinned: the fixed behavior as shipped in
  quail.runtime.worker. First call seeds the modelinfos JSON on the
  volume (one-time cost); the second call, in a fresh container, is
  the steady-state cold boot every later container sees.

Each probe mirrors quail.executor.model.load_model step by step
(dtype="auto", single-rank group stubs, no NCCL) and adds a raw read
of the cached safetensors files to separate "the volume is slow"
from "vLLM is slow". Records are written to the quail-results
volume under /results/ablations/.

PREDICTION (stated before the run): probe_main pays ~13-19 s in
engine_config (architecture subprocess + hub round trips) and
~14-18 s in get_model (hub round trips + 4.83 GiB of weights at
~2 GiB/s), in-timer total ~35-45 s. probe_pinned's seed call pays
engine_config once at ~13-15 s; its steady-state call drops
engine_config to ~0.1-2 s and get_model to ~10-12 s, in-timer total
~22-27 s. The raw weight read stays ~2 s either way, so the
remaining floor is the torch+vllm import (~12-16 s) plus get_model.

Run from the repository root (tee per house rule):

    uv run modal run -m experiments.cells.load_profile 2>&1 \
        | tee /tmp/load_profile.log
"""

from __future__ import annotations

import json

import modal

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

# The worker's image env, VLLM_CACHE_ROOT included; probe_main pops
# it at runtime to reproduce the pre-fix behavior on the same image.
image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub", "numpy", "pyarrow")
    .env({"VLLM_CACHE_ROOT": "/root/.cache/kernels/vllm",
          "VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "TRITON_CACHE_DIR": "/root/.cache/kernels/triton"})
    .add_local_python_source("quail")
)

# House rule: attach to the existing milestone1 app; never invent a
# new Modal app name.
app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results",
                                     create_if_missing=True)

# single_use_containers: every call gets a cold container, so each
# record is a true cold load, never a warm-process repeat.
GPU_KW = dict(image=image, gpu="H100!", memory=65536, timeout=1800,
              single_use_containers=True,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})

RESULT_DIR = "/results/ablations"


def _profile(model_name: str, revision: str | None) -> dict:
    """Run load_model's exact steps, timing each phase."""
    import time

    phases: dict[str, float] = {}
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    import torch
    phases["import_torch_s"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model
    phases["import_vllm_s"] = round(time.perf_counter() - t0, 2)

    from quail.executor.model import (
        _install_single_rank_groups,
        retain_answer_head,
    )

    t0 = time.perf_counter()
    config = EngineArgs(model=model_name, dtype="auto",
                        revision=revision or None,
                        enforce_eager=True).create_engine_config()
    phases["engine_config_s"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    _install_single_rank_groups(torch)
    with set_current_vllm_config(config):
        model = get_model(vllm_config=config)
    from transformers import AutoTokenizer

    from quail import true_false_ids

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    true_ids, false_ids = true_false_ids(tokenizer)
    retain_answer_head(torch, model, true_ids | false_ids)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    phases["get_model_s"] = round(time.perf_counter() - t0, 2)

    phases["total_s"] = round(time.perf_counter() - t_all, 2)
    del model
    return phases


def _raw_weight_read(model_name: str) -> dict:
    """Read the cached safetensors files; report bytes and throughput."""
    import time
    from pathlib import Path

    root = Path("/root/.cache/huggingface/hub"
                ) / f"models--{model_name.replace('/', '--')}"
    snap = next((root / "snapshots").iterdir())
    files = sorted(snap.glob("*.safetensors"))
    t0 = time.perf_counter()
    total = 0
    for f in files:
        with open(f, "rb") as fh:
            while chunk := fh.read(1 << 26):
                total += len(chunk)
    read_s = time.perf_counter() - t0
    gib = total / 2**30
    return dict(gib=round(gib, 2), read_s=round(read_s, 2),
                gib_per_s=round(gib / read_s, 2))


def _record(variant: str, phases: dict, raw: dict, model: str,
            revision: str | None) -> str:
    import os
    import time

    os.makedirs(RESULT_DIR, exist_ok=True)
    path = f"{RESULT_DIR}/load_model_profile_{variant}.json"
    with open(path, "w") as f:
        json.dump(dict(variant=variant, model=model,
                       revision=revision, phases=phases,
                       raw_weight_read=raw,
                       recorded_at=time.strftime(
                           "%Y-%m-%dT%H:%M:%SZ", time.gmtime())), f,
                  indent=2)
    results_vol.commit()
    print(f"[load_profile] {variant}: {json.dumps(phases)}", flush=True)
    print(f"[load_profile] {variant} raw read: {json.dumps(raw)}",
          flush=True)
    print(f"[load_profile] saved {path}", flush=True)
    return path


@app.function(**GPU_KW)
def probe_main() -> dict:
    """Pre-fix behavior: ephemeral vLLM cache, no pinned revision."""
    import os

    # popped before any vllm import, so the modelinfos cache falls
    # back to the container-local default and misses
    os.environ.pop("VLLM_CACHE_ROOT", None)
    from quail.specs import QWEN3_4B_FP8

    phases = _profile(QWEN3_4B_FP8.hf_name, None)
    raw = _raw_weight_read(QWEN3_4B_FP8.hf_name)
    _record("main", phases, raw, QWEN3_4B_FP8.hf_name, None)
    return phases


@app.function(**GPU_KW)
def probe_pinned(variant: str = "pinned") -> dict:
    """Fixed behavior: modelinfos on the volume, revision pinned."""
    from quail.specs import QWEN3_4B_FP8

    phases = _profile(QWEN3_4B_FP8.hf_name, QWEN3_4B_FP8.revision)
    raw = _raw_weight_read(QWEN3_4B_FP8.hf_name)
    kernel_cache.commit()   # keep the seeded modelinfos JSON
    _record(variant, phases, raw, QWEN3_4B_FP8.hf_name,
            QWEN3_4B_FP8.revision)
    return phases


def _fc_id(call) -> str:
    return (getattr(call, "function_call_id", None)
            or getattr(call, "object_id", None))


@app.local_entrypoint()
def main():
    """Control, seed, then steady state; one cold container each."""
    runs = [
        ("main", probe_main.spawn()),
        # seeds the modelinfos JSON on the kernel-cache volume; its
        # engine_config_s pays the subprocess once
        ("pinned_seed", probe_pinned.spawn("pinned_seed")),
    ]
    for name, call in runs:
        print(f"[load_profile] {name} fc id: {_fc_id(call)}",
              flush=True)
        print(f"[load_profile] {name}: "
              f"{json.dumps(call.get(), indent=2)}", flush=True)
    # after the seed committed the volume: the steady-state cold boot
    call = probe_pinned.spawn("pinned")
    print(f"[load_profile] pinned fc id: {_fc_id(call)}", flush=True)
    print(f"[load_profile] pinned: "
          f"{json.dumps(call.get(), indent=2)}", flush=True)
