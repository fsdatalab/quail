"""Profile vLLM joins with PyTorch Profiler on Modal.

    uv run modal run --detach experiments/profile_vllm_join.py \
      2>&1 | tee /tmp/quail-vllm-join-profile.log

    uv run modal run --detach experiments/profile_vllm_join.py --query BIO-3 \
      2>&1 | tee /tmp/quail-bio3-vllm-join-profile.log

Scheduler CPU/CUDA traces, driver CPU traces, answers, and process timings
are saved on quail-results under /results/ablations/vllm-join-profile-<UTC>/.
Profiler timings are diagnostic and do not replace benchmark measurements.
"""

import json
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path

from experiments.vllm_join_profile_worker import PREDICTION_TEXTS
from quail.bench.quailb_parallel import VOLUMES, app, image, results_vol


@app.function(
    image=image.add_local_python_source("experiments"),
    gpu="H100!", memory=98304, timeout=3600, volumes=VOLUMES,
)
def profile(query="FEV-9"):
    """Run the profiler in a child process and save its traces."""
    from experiments.vllm_join_profile_worker import profile_worker
    from quail.bench.process_isolation import _stop_process_group

    if query not in PREDICTION_TEXTS:
        raise ValueError(f"Unsupported profiling query: {query}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(f"/results/ablations/vllm-join-profile-{stamp}")
    root.mkdir(parents=True)
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=profile_worker, args=(str(root), sender, query))
    process.start()
    sender.close()
    try:
        receiver.recv()
    finally:
        receiver.close()
        cleanup = _stop_process_group(process)
        results_vol.commit()
    if (root / "error.txt").exists():
        raise RuntimeError((root / "error.txt").read_text())
    result_path = root / "result.json"
    result = json.loads(result_path.read_text())
    result["process_cleanup"] = cleanup
    result["result_volume_path"] = str(result_path)
    result_path.write_text(json.dumps(result, indent=2))
    results_vol.commit()
    return str(result_path)


@app.local_entrypoint()
def profile_joins(query: str = "FEV-9"):
    """Start the join profile and print its saved result path."""
    print(f"query: {query}; prediction: {PREDICTION_TEXTS[query]}", flush=True)
    call = profile.spawn(query)
    print(f"function call id: {call.object_id}", flush=True)
    print(f"result volume path: {call.get()}", flush=True)
