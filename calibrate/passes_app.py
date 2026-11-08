"""Measure filter-scan pass times on Modal and save them to quail-results.

Run:
    uv run modal run --detach -m calibrate.passes_app \
        --prediction "<ratio range per pass size>" 2>&1 | tee "$run_log"
Then fit locally:
    uv run --with scikit-learn python -m calibrate.fit_passes \
        measurements.jsonl quail/cost/tables/qwen3-4b-fp8__h100-sxm.json
"""

from datetime import datetime, timezone
from pathlib import Path

import modal

from calibrate import passes
from quail.deploy.images import gpu_image
from quail.planner.plan import EngineConfig

ROOT = Path("/results/calibrate")
app = modal.App("quail-milestone1")
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
image = gpu_image().add_local_python_source("calibrate")


@app.function(image=image, gpu="H100!", timeout=7200,
              volumes={"/results": results_vol})
def measure_passes(model: str, device: str, stamp: str, prediction: str) -> str:
    """Measure every grid point on one GPU and save the records; return the path."""
    import quail

    def make_session():
        return quail.Session(EngineConfig(model=model, device=device,
                                          gpu_timing=True))

    records = []
    for words, num_documents in passes.grid():
        # the first run pays warmup, so it is measured twice and the first dropped
        if not records:
            passes.measure(make_session, words, num_documents)
        records.append(passes.measure(make_session, words, num_documents))
    path = ROOT / device / "passes" / stamp / "measurements.jsonl"
    passes.save(path, [{**r, "prediction": prediction} for r in records])
    results_vol.commit()
    return str(path)


@app.local_entrypoint()
def main(model: str = "qwen3-4b-fp8", device: str = "h100-sxm",
         prediction: str = ""):
    """Measure pass times for one model and device."""
    if not prediction.strip():
        raise ValueError("state --prediction before measuring")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    print(f"prediction: {prediction}", flush=True)
    call = measure_passes.spawn(model, device, stamp, prediction)
    print(f"function call id {call.object_id}", flush=True)
    print(f"saved {call.get()}", flush=True)
