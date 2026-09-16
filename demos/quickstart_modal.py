"""Run the shared quickstart on a Modal H100.

uv run --no-sync --with 'modal[api-proxy-support]==1.5.4' \
  modal run demos/quickstart_modal.py 2>&1 | tee results/quickstart_modal.log
"""

import json
import uuid
from pathlib import Path

import modal

from demos import quickstart

app = modal.App("quail-engine")
volume = modal.Volume.from_name("quail-results", create_if_missing=True)
RESULTS_DIR = Path("/results/quickstart")
IMAGE_REQUIREMENTS = (
    "sqlglot==30.17.0",
    "transformers==5.15.0",
    "huggingface-hub==1.27.0",
    "pyarrow==25.0.1",
    "numpy==2.3.5",
    "gigatoken==0.10.0",
    "datasets==5.0.1",
    "vllm==0.26.0",
)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .pip_install(*IMAGE_REQUIREMENTS)
    .env({
        "HF_HOME": "/results/models",
        "QUAIL_CACHE_DIR": "/results/kernels",
        "DG_CACHE_DIR": "/results/kernels/deep_gemm",
        "DG_JIT_CACHE_DIR": "/results/kernels/deep_gemm",
        "TRITON_CACHE_DIR": "/results/kernels/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/results/kernels/torchinductor",
        "VLLM_CACHE_ROOT": "/results/kernels/vllm",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    })
    .add_local_python_source("quail", "demos")
)


@app.function(
    image=image, gpu="H100!", memory=98304, timeout=1200,
    volumes={"/results": volume},
)
def run_query():
    """Run the quickstart and save its report at this script's chosen path."""
    try:
        rows, report = quickstart.run_query()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"{uuid.uuid4().hex}.json"
        report["result_volume_path"] = str(path)
        path.write_text(json.dumps(report))
        return rows, report
    finally:
        volume.commit()


@app.local_entrypoint()
def main():
    """Submit the function and print its call id and result."""
    call = run_query.spawn()
    print(f"function call id: {call.object_id}", flush=True)
    rows, report = call.get()
    print(rows.to_pylist())
    print(report)
