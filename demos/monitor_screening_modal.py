"""Run the monitor screening demo on a Modal H100 GPU.

uv run modal run --detach demos/monitor_screening_modal.py \
  2>&1 | tee /tmp/quail-monitor-screening.log
"""

from __future__ import annotations

import uuid
from pathlib import Path

import modal

from quail.deploy.images import gpu_image

app = modal.App("quail-milestone1")
results_volume = modal.Volume.from_name("quail-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache", create_if_missing=True)

image = gpu_image().add_local_python_source("demos")

RESULTS_DIR = Path("/vol/results/demos/monitor-screening")


@app.function(image=image, gpu="H100!", timeout=3_600, memory=65_536,
              volumes={"/vol/results": results_volume,
                       "/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache})
def evaluate(threshold: float, max_rows: int) -> dict:
    """Screen the Betley et al. datasets on a Modal GPU and save the report."""
    from demos.monitor_screening import download_betley, run

    directory = RESULTS_DIR / uuid.uuid4().hex
    paths = download_betley(directory / "inputs")
    report = run(paths, directory, threshold=threshold, max_rows=max_rows)
    results_volume.commit()
    print(f"result volume path: {directory}", flush=True)
    return {**report, "result_volume_path": str(directory)}


@app.local_entrypoint()
def main(threshold: float = 0.5, max_rows: int = 5_000):
    """Screen the insecure and secure code datasets."""
    import json

    call = evaluate.spawn(threshold, max_rows)
    print(f"function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=2))
