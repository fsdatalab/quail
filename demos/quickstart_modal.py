"""Run the shared quickstart on a Modal H100.

uv run --no-sync --with 'modal[api-proxy-support]==1.5.4' \
  modal run demos/quickstart_modal.py 2>&1 | tee results/quickstart_modal.log
"""

import json
import uuid
from pathlib import Path

import modal

from demos import quickstart
from quail.bench.images import gpu_image

app = modal.App("quail-engine")
volume = modal.Volume.from_name("quail-results", create_if_missing=True)
RESULTS_DIR = Path("/results/quickstart")

image = gpu_image(("demos", "/root/demos"))


@app.function(
    image=image, gpu="H100!", memory=98304, timeout=1200,
    volumes={
        "/results": volume,
        "/root/.cache/huggingface": modal.Volume.from_name(
            "quail-hf-cache", create_if_missing=True),
        "/root/.cache/kernels": modal.Volume.from_name(
            "quail-kernel-cache", create_if_missing=True),
    },
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
