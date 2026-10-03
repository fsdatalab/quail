"""Measure one Modal GPU: CUDA name, exposed memory, and vLLM kernel flags.

Prediction (2026-10-03): L40S is Ada 8.9, about 48 GB, no DeepGEMM, no
UE8M0. B200 is Blackwell 10.0, about 180 GB usable, DeepGEMM on, UE8M0
on. Both raise in today's flash_attention_version. Modal GPU-only
prices: L40S $1.9512/h, B200 $6.2496/h
(https://modal.com/pricing, $0.000542/s and $0.001736/s).

    mkdir -p results
    log="results/$(date -u +%Y%m%dT%H%M%SZ)-device-probe.log"
    uv run modal run --detach experiments/cells/device_probe.py \
      --gpu-types L40S,B200 \
      2>&1 | tee "$log"
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import modal

from quail.bench.images import gpu_image

PREDICTION_TEXT = (
    "L40S: capability (8, 9), mem_bytes near 48e9, DeepGEMM off, "
    "UE8M0 off, flash_attention_version raises. "
    "B200: capability (10, 0), mem_bytes near 180e9, DeepGEMM on, "
    "UE8M0 on, flash_attention_version raises. "
    "Modal prices 2026-10-03: L40S $1.9512/h, B200 $6.2496/h."
)

# GPU-only rates from https://modal.com/pricing, checked 2026-10-03.
MODAL_USD_PER_HOUR = {
    "L40S": 0.000542 * 3600,
    "B200": 0.001736 * 3600,
    "H100!": 0.001097 * 3600,
}

app = modal.App("quail-milestone1")
image = gpu_image()
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
volumes = {"/results": results_vol}


def _deep_gemm_supported() -> bool | str:
    """Return vLLM's DeepGEMM allow-list result, or the import error."""
    try:
        from vllm.platforms.cuda import CudaPlatform
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    flag = getattr(CudaPlatform, "support_deep_gemm", None)
    if flag is None:
        return "missing"
    if callable(flag):
        return bool(flag())
    return bool(flag)


def _ue8m0_used() -> bool | str:
    """Return whether vLLM writes UE8M0 scales, or the import error."""
    try:
        from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return bool(is_deep_gemm_e8m0_used())


def _triton_block_fp8_name() -> str | None:
    """The Ada block-FP8 matmul symbol, if vLLM exports it."""
    from vllm.model_executor.layers.quantization.utils import fp8_utils

    name = "w8a8_triton_block_scaled_mm"
    return name if hasattr(fp8_utils, name) else None


def _flash_attention_version(capability: tuple[int, int]) -> dict:
    """Record today's dispatcher result for this capability."""
    from quail.backends.quail.executor.attention import flash_attention_version

    try:
        return {"version": flash_attention_version(capability), "error": None}
    except ValueError as error:
        return {"version": None, "error": str(error)}


@app.function(image=image, gpu="L40S", memory=8192, timeout=1200,
              volumes=volumes)
def probe(gpu_type: str, run_dir: str) -> str:
    """Write one GPU's CUDA name, memory, and vLLM flags to the volume."""
    import torch

    props = torch.cuda.get_device_properties(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    record = {
        "gpu_type": gpu_type,
        "cuda_name": torch.cuda.get_device_name(0),
        "capability": list(capability),
        "mem_bytes": int(props.total_memory),
        "major": props.major,
        "minor": props.minor,
        "support_deep_gemm": _deep_gemm_supported(),
        "is_deep_gemm_e8m0_used": _ue8m0_used(),
        "triton_block_fp8": _triton_block_fp8_name(),
        "flash_attention_version": _flash_attention_version(capability),
        "usd_per_hour": MODAL_USD_PER_HOUR.get(gpu_type),
        "price_source": "https://modal.com/pricing",
        "price_checked": "2026-10-03",
        "prediction": PREDICTION_TEXT,
    }
    path = Path(run_dir) / f"{gpu_type.replace('!', '')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    results_vol.commit()
    print(json.dumps(record, indent=2), flush=True)
    return str(path)


@app.local_entrypoint()
def main(gpu_types: str = "L40S,B200"):
    """Submit one probe per GPU type and print the function call ids."""
    selected = [item.strip() for item in gpu_types.split(",") if item.strip()]
    unknown = [item for item in selected if item not in MODAL_USD_PER_HOUR]
    if unknown:
        raise ValueError(f"unknown gpu types: {unknown}")
    print(f"PREDICTION: {PREDICTION_TEXT}", flush=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = f"/results/ablations/device_probe/{run_id}"
    print(f"run directory: {run_dir}", flush=True)
    calls = {
        gpu_type: probe.with_options(gpu=gpu_type).spawn(gpu_type, run_dir)
        for gpu_type in selected
    }
    for gpu_type, call in calls.items():
        print(f"function call id: {call.object_id} ({gpu_type})", flush=True)
    for gpu_type, call in calls.items():
        print(f"{gpu_type}: {call.get()}", flush=True)
