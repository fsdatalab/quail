"""Boot-phase probe: where do load_model's 28-38 seconds go?

Three single-GPU variants, each in its own Modal function (and so its
own process - distributed state is process-global, so the variants
cannot share a container):

- probe_stock: the old load_model sequence with NCCL
  init_distributed_environment.
- probe_nodist: current load_model - same get_model, single-rank
  group stubs, no NCCL or gloo.
- probe_lean: no vllm.engine.arg_utils import, VllmConfig built
  directly from ModelConfig, gloo instead of NCCL.

PREDICTION (stated before the run): import torch 3-8 s; the vllm
import chain 5-12 s; engine config 1-3 s; NCCL 1-3 s; weight load
8-15 s on a warm HF volume (5.2 GB). nodist drops the NCCL 1-3 s
and leaves the rest. The floor is the torch+vllm import plus the
weight bytes themselves.

Run from the quail/ directory (tee per house rule):

    uv run modal run experiments/cells/boot_probe.py 2>&1 | tee results/boot_probe.log
"""

from __future__ import annotations

import json

import modal

MODEL = "Qwen/Qwen3-4B-FP8"

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub", "numpy")
    .env({"VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "TRITON_CACHE_DIR": "/root/.cache/kernels/triton"})
    .add_local_python_source("quail", "corpus")
)

# House rule: attach to the existing milestone1 app; never invent a
# new Modal app name.
app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)

GPU_KW = dict(image=image, gpu="H100!", memory=65536, timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache})


class _Timer:
    def __init__(self):
        import time
        self.t = time.perf_counter()
        self.phases = {}

    def mark(self, name):
        import time
        now = time.perf_counter()
        self.phases[name] = round(now - self.t, 2)
        self.t = now
        return now



def _stock(model_name: str) -> dict:
    """The old load_model sequence with NCCL, phase by phase."""
    tm = _Timer()
    import torch  # noqa: F401
    tm.mark("import_torch")

    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model
    from vllm.utils.network_utils import get_open_port
    tm.mark("import_vllm")

    config = EngineArgs(model=model_name, dtype="auto",
                        enforce_eager=True).create_engine_config()
    tm.mark("engine_config")

    import torch.distributed as dist
    with set_current_vllm_config(config):
        if not dist.is_initialized():
            init_distributed_environment(
                world_size=1, rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
                local_rank=0, backend="nccl")
            ensure_model_parallel_initialized(1, 1)
        tm.mark("dist_nccl")
        model = get_model(vllm_config=config)
    torch.cuda.synchronize()
    tm.mark("get_model")
    del model
    tm.phases["total"] = round(sum(tm.phases.values()), 2)
    return tm.phases


def _lean(model_name: str, load_format: str | None = None) -> dict:
    """No arg_utils, direct VllmConfig, gloo, optional fastsafetensors."""
    tm = _Timer()
    import torch  # noqa: F401
    tm.mark("import_torch")

    from vllm.config import (LoadConfig, ModelConfig, VllmConfig,
                             set_current_vllm_config)
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.model_executor.model_loader import get_model
    from vllm.utils.network_utils import get_open_port
    tm.mark("import_vllm")

    kwargs = {}
    if load_format is not None:
        kwargs["load_config"] = LoadConfig(load_format=load_format)
    config = VllmConfig(
        model_config=ModelConfig(model=model_name, dtype="auto",
                                 enforce_eager=True),
        **kwargs)
    tm.mark("direct_config")

    import torch.distributed as dist
    with set_current_vllm_config(config):
        if not dist.is_initialized():
            init_distributed_environment(
                world_size=1, rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
                local_rank=0, backend="gloo")
            ensure_model_parallel_initialized(1, 1)
        tm.mark("dist_gloo")
        model = get_model(vllm_config=config)
    torch.cuda.synchronize()
    tm.mark("get_model")
    del model
    tm.phases["total"] = round(sum(tm.phases.values()), 2)
    return tm.phases


@app.function(**GPU_KW)
def probe_stock() -> dict:
    out = _stock(MODEL)
    print(f"[boot_probe] stock: {json.dumps(out)}", flush=True)
    return out


@app.function(**GPU_KW)
def probe_phases() -> dict:
    """The current load_model body, timed per step."""
    tm = _Timer()
    import torch
    tm.mark("import_torch")

    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model
    tm.mark("import_vllm")

    from quail.executor.model import _install_single_rank_groups
    config = EngineArgs(model=MODEL, dtype="auto",
                        enforce_eager=True).create_engine_config()
    tm.mark("engine_config")

    _install_single_rank_groups(torch)
    tm.mark("stubs")

    with set_current_vllm_config(config):
        model = get_model(vllm_config=config)
    torch.cuda.synchronize()
    tm.mark("get_model")
    del model
    tm.phases["total"] = round(sum(tm.phases.values()), 2)
    print(f"[boot_probe] phases: {json.dumps(tm.phases)}", flush=True)
    return tm.phases


@app.function(**GPU_KW)
def probe_nodist() -> dict:
    tm = _Timer()
    import torch  # noqa: F401
    tm.mark("import_torch")
    from quail.executor.model import load_model
    tm.mark("import_load_model")
    model = load_model(MODEL)
    tm.mark("load_model")
    del model
    tm.phases["total"] = round(sum(tm.phases.values()), 2)
    print(f"[boot_probe] nodist: {json.dumps(tm.phases)}", flush=True)
    return tm.phases


@app.function(**GPU_KW)
def probe_lean() -> dict:
    out = _lean(MODEL)
    print(f"[boot_probe] lean: {json.dumps(out)}", flush=True)
    return out


@app.local_entrypoint()
def run_phases():
    h = probe_phases.spawn()
    fc_id = getattr(h, "function_call_id", None) or getattr(
        h, "object_id", None)
    print(f"[boot_probe] spawned phases: {fc_id}", flush=True)
    out = h.get()
    print(json.dumps(out, indent=2), flush=True)


@app.local_entrypoint()
def run_nodist():
    h = probe_nodist.spawn()
    fc_id = getattr(h, "function_call_id", None) or getattr(
        h, "object_id", None)
    print(f"[boot_probe] spawned nodist: {fc_id}", flush=True)
    out = h.get()
    print(json.dumps(out, indent=2), flush=True)


@app.local_entrypoint()
def main():
    handles = {
        "stock": probe_stock.spawn(),
        "nodist": probe_nodist.spawn(),
        "lean": probe_lean.spawn(),
    }
    for name, h in handles.items():
        fc_id = getattr(h, "function_call_id", None) or getattr(
            h, "object_id", None)
        print(f"[boot_probe] spawned {name}: {fc_id}", flush=True)
    report = {name: h.get() for name, h in handles.items()}
    print(json.dumps(report, indent=2), flush=True)
