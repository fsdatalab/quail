"""Run the Civil Comments Quail backend on Modal.

Run from the repository root:

    uv run modal run --detach demos/civil_comments/quail_modal.py \
      --limit 10000 --gpus 1 \
      2>&1 | tee /tmp/civil-comments-quail.log
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import modal

from demos.civil_comments.quail_backend import evaluate
from quail.bench.images import gpu_image

RESULTS_MOUNT = Path("/results")
RESULTS_DIR = Path("demos/civil-comments/quail")

app = modal.App("quail-milestone1")
results_volume = modal.Volume.from_name("quail-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache", create_if_missing=True)
image = gpu_image(("demos", "/root/demos")).add_local_python_source("demos")


@app.function(
    image=image,
    gpu="H100!",
    timeout=86_400,
    memory=98_304,
    volumes={
        str(RESULTS_MOUNT): results_volume,
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/kernels": kernel_cache,
    },
)
def run(limit: int, gpus: int) -> dict:
    """Run one Quail configuration on Modal."""
    results_volume.reload()
    hf_cache.reload()
    relative_directory = RESULTS_DIR / uuid.uuid4().hex
    directory = RESULTS_MOUNT / relative_directory
    directory.mkdir(parents=True)
    summary = evaluate(directory, None if limit == 0 else limit, gpus)
    summary["result_volume_path"] = str(relative_directory)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    results_volume.commit()
    hf_cache.commit()
    return summary


@app.local_entrypoint()
def main(limit: int = 10_000, gpus: int = 1):
    """Start one Modal run and print the function call id."""
    if limit < 0 or gpus not in (1, 2, 4, 8):
        raise ValueError("limit must be >= 0; gpus must be 1, 2, 4, or 8")
    gpu = "H100!" if gpus == 1 else f"H100!:{gpus}"
    call = run.with_options(gpu=gpu).spawn(limit, gpus)
    print(f"function call id: {call.object_id}", flush=True)
    call.get()
