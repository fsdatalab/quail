"""Where does load_model's time go? One cold container, each phase timed.

quail.executor.model.load_model measures ~35 s cold (boot_profile mean
35.1 s). This cell reruns its exact steps in the same order in a fresh
container and times each one: the vLLM imports (which live inside the
function, so they ride inside every load_model_s measurement), engine
config creation, distributed init, get_model (weight load + fp8
post-processing), and the closing cuda sync. A raw read of the cached
safetensors files separates "the HF cache volume is slow" from "vLLM
is slow". vLLM logging is raised to INFO before its import so its own
"took N seconds" lines land in the capture.

PREDICTION (stated before the run): the vLLM import dominates -
load_model imports all of vllm lazily inside the timed call, and a
cold ``import vllm`` pulls in triton, the custom-op registry, and the
platform probe. get_model is next at a few seconds (~4.4 GiB of
weights, volume usually streams a shard in ~2 s), then config + NCCL
init at ~1 s each. The raw read finishes in well under 10 s.

Second variant (2026-08-19): the image env sets HF_HUB_OFFLINE=1 so
the hub round-trips go away - everything vLLM needs is already on
the quail-hf-cache volume. Result: get_model_s dropped 14.4 -> 9.9 s,
but engine_config_s stayed ~13.5 s, so its cost is not hub calls.
Third variant: cProfile around create_engine_config and get_model
named the config cost: ~11 s is one subprocess. vLLM resolves the
model architecture by running `python -m
vllm.model_executor.models.registry` in a fresh interpreter (to avoid
initializing CUDA in the main process), and caches the result as JSON
under $VLLM_CACHE_ROOT/modelinfos. VLLM_CACHE_ROOT defaults to
~/.cache/vllm - ephemeral container storage - so every Modal
container pays the full subprocess again.

Fourth variant: VLLM_CACHE_ROOT points at the kernel-cache volume.
One cold call seeds the modelinfo JSON (pays the subprocess once),
then the timed suite reads it back like every later container would.
Result: seed 15.06 s once, timed engine_config_s 0.12 s. Confirmed,
and the JSON persists on the volume.

Fifth variant (this file): HF_HUB_OFFLINE dropped - a missing
snapshot must still download, not fail. Instead the revision is
pinned to the hub commit hash from the spec; a commit hash resolves
from the local HF cache without the API round trips that a branch
name pays. Prediction: get_model_s matches the offline run's ~9.9 s
(round trips gone) while still downloading on a cold cache; with the
modelinfos cache warm, engine_config_s is ~1-2 s (some hub work
remains even pinned); in-timer total ~21-25 s.

Run from the quail/ directory (tee per house rule):

    uv run modal run tests/gpu/load_profile.py \
        2>&1 | tee results/load_profile.log
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

from quail.specs import QWEN3_4B_FP8

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

# Identical image block to tests/gpu/boot_profile.py: the 35 s it
# measured was taken under these exact layers and env.
image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub", "pandas", "pyarrow",
                 "numpy", "datasets")
    .env({# modelinfos JSON rides the kernel-cache volume so the
          # architecture-inspection subprocess runs once ever, not
          # once per container
          "VLLM_CACHE_ROOT": "/root/.cache/kernels/vllm",
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

MODEL = QWEN3_4B_FP8.hf_name
REVISION = QWEN3_4B_FP8.revision
SNAP_ROOT = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B-FP8"

PREDICTION = (
    "revision pinned to a commit hash, hub online: get_model_s "
    "matches the offline run's ~9.9 s (commit hash resolves from the "
    "local HF cache without API calls, branch names cannot); "
    "engine_config_s ~1-2 s with modelinfos warm; in-timer total "
    "~21-25 s versus ~37 s unpinned online. Cold-cache downloads "
    "still work."
)


def _cprofile_top(fn, label: str, n: int = 20) -> tuple:
    """Run fn under cProfile; return (result, top-lines text)."""
    import cProfile
    import io
    import pstats

    prof = cProfile.Profile()
    result = prof.runcall(fn)
    buf = io.StringIO()
    stats = pstats.Stats(prof, stream=buf)
    stats.sort_stats("tottime").print_stats(n)
    # Who calls the one big subprocess wait (engine_config's ~11 s)?
    stats.print_callers("communicate", "poll")
    text = buf.getvalue()
    print(f"[load_profile] cprofile {label}:\n{text}", flush=True)
    return result, text


def _raw_weight_read() -> dict:
    """Read every cached safetensors file; report bytes and throughput."""
    snap = next((Path(SNAP_ROOT) / "snapshots").iterdir())
    files = sorted(snap.glob("*.safetensors"))
    t0 = time.perf_counter()
    total = 0
    for f in files:
        with open(f, "rb") as fh:
            while chunk := fh.read(1 << 26):
                total += len(chunk)
    read_s = time.perf_counter() - t0
    gib = total / 2**30
    return dict(files=[f.name for f in files],
                gib=round(gib, 2), read_s=round(read_s, 2),
                gib_per_s=round(gib / read_s, 2))


class _Capture:
    """Record vLLM's own startup lines with elapsed times."""

    def __init__(self, t0: float):
        self.t0 = t0
        self.lines: list[dict] = []

    def handler(self):
        import logging
        cell = self

        class H(logging.Handler):
            def emit(self, record):
                msg = record.getMessage()
                if any(k in msg.lower() for k in
                       ("took", "loading", "weights", "graph", "compile",
                        "deepgemm", "deep_gemm", "profile")):
                    cell.lines.append(
                        dict(t=round(time.perf_counter() - cell.t0, 2),
                             msg=msg[:200]))

        return H(level=logging.INFO)


@app.function(timeout=1800, gpu="H100!", memory=65536,
              image=image,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache})
def load_phase_profile() -> dict:
    """Time load_model's exact steps, cold, in one container."""
    import logging
    import os

    # Raised before any vllm import; vllm reads the level at import.
    os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
    phases: dict[str, float] = {}

    t_all = time.perf_counter()

    t0 = time.perf_counter()
    import torch
    phases["import_torch_s"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model
    from vllm.utils.network_utils import get_open_port
    phases["import_vllm_s"] = round(time.perf_counter() - t0, 2)

    cap = _Capture(time.perf_counter())
    vllm_log = logging.getLogger("vllm")
    vllm_log.setLevel(logging.INFO)
    vllm_log.addHandler(cap.handler())

    # Seed the modelinfos JSON the way the first container on the
    # volume would (one untimed pass), then measure what every later
    # container sees: a cache hit instead of the ~11 s subprocess.
    t_seed = time.perf_counter()
    EngineArgs(model=MODEL, dtype="bfloat16", revision=REVISION,
               enforce_eager=True).create_engine_config()
    print(f"[load_profile] seed create_engine_config "
          f"{time.perf_counter() - t_seed:.2f}s", flush=True)

    t0 = time.perf_counter()
    config, prof_config = _cprofile_top(
        lambda: EngineArgs(model=MODEL, dtype="bfloat16",
                           revision=REVISION,
                           enforce_eager=True).create_engine_config(),
        "engine_config")
    phases["engine_config_s"] = round(time.perf_counter() - t0, 2)

    with set_current_vllm_config(config):
        import torch.distributed as dist
        t0 = time.perf_counter()
        init_distributed_environment(
            world_size=1, rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            local_rank=0, backend="nccl")
        ensure_model_parallel_initialized(1, 1)
        phases["dist_init_s"] = round(time.perf_counter() - t0, 2)

        t0 = time.perf_counter()
        model, prof_model = _cprofile_top(lambda: get_model(vllm_config=config),
                                          "get_model")
        phases["get_model_s"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    torch.cuda.synchronize()
    phases["cuda_sync_s"] = round(time.perf_counter() - t0, 2)
    del model

    phases["sum_s"] = round(time.perf_counter() - t_all, 2)

    raw = _raw_weight_read()
    out = dict(model=MODEL, prediction=PREDICTION, phases=phases,
               raw_weight_read=raw, vllm_log=cap.lines,
               cprofile_config=prof_config, cprofile_model=prof_model)
    print(f"[load_profile] phases={json.dumps(phases)}", flush=True)
    print(f"[load_profile] raw={json.dumps(raw)}", flush=True)
    for line in cap.lines:
        print(f"[load_profile] vllm@{line['t']:>7.2f}s {line['msg']}",
              flush=True)
    return out


@app.local_entrypoint()
def main(out: str = "results/load_profile.json"):
    print(f"[load_profile] prediction: {PREDICTION}", flush=True)
    report = load_phase_profile.remote()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2))
    print(f"[load_profile] saved {out}", flush=True)
