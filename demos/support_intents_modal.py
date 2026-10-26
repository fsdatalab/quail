"""Run the support intents demo on a Modal H100 GPU.

uv run modal run --detach demos/support_intents_modal.py \
  2>&1 | tee /tmp/quail-support-intents.log
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

RESULTS_DIR = Path("/vol/results/demos/support-intents")


@app.function(image=image, gpu="H100!", timeout=3_600, memory=65_536,
              volumes={"/vol/results": results_volume,
                       "/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache})
def evaluate(scale_factor: float) -> dict:
    """Read the features on a Modal GPU and save the report."""
    from demos.support_intents import run

    directory = RESULTS_DIR / uuid.uuid4().hex
    report = run(directory, scale_factor)
    results_volume.commit()
    print(f"result volume path: {directory}", flush=True)
    return report


@app.local_entrypoint()
def main(scale_factor: float = 1.0):
    """Find intent features in the support conversations."""
    import json

    call = evaluate.spawn(scale_factor)
    print(f"function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=2))
