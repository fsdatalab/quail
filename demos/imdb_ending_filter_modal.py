"""Run the IMDB review filter on Modal H100 GPUs.

uv run modal run demos/imdb_ending_filter_modal.py \
  --model decision-2.0-kai-0.6b-bf16 \
  2>&1 | tee /tmp/quail-imdb-filter.log

The report is saved under /results/demos/imdb-filter/ on the
quail-results volume. Fetch the return value later with
modal.FunctionCall.from_id("<id>").get().
"""

import json
import uuid
from pathlib import Path

import modal

from quail.deploy.images import gpu_image

app = modal.App("quail-milestone1")
results_volume = modal.Volume.from_name("quail-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache", create_if_missing=True)

image = gpu_image().add_local_python_source("demos")

RESULTS_DIR = Path("/results/demos/imdb-filter")


@app.function(image=image, gpu="H100!", timeout=3600, memory=65_536,
              volumes={"/results": results_volume,
                       "/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache})
def run_filter(model: str, gpus: int) -> dict:
    """Run the filter and save its summary to the results volume."""
    from demos.imdb_ending_filter import run

    try:
        summary = run(model=model, gpus=gpus)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"{uuid.uuid4().hex}.json"
        summary["result_volume_path"] = str(path)
        path.write_text(json.dumps(summary, indent=2))
        print(f"result volume path: {path}", flush=True)
        return summary
    finally:
        results_volume.commit()
        hf_cache.commit()


@app.local_entrypoint()
def main(model: str = "decision-2.0-kai-0.6b-bf16", gpus: int = 1):
    """Submit the filter and print its function call id."""
    if gpus not in (1, 2, 4, 8):
        raise ValueError("gpus must be 1, 2, 4, or 8")
    call = run_filter.with_options(gpu=f"H100!:{gpus}").spawn(model, gpus)
    print(f"function call id: {call.object_id}", flush=True)
    call.get()
