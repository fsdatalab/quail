"""Profile all three SGLang FEV-9 joins with PyTorch Profiler on Modal.

    uv run modal run --detach experiments/profile_sglang_join.py \
      2>&1 | tee /tmp/quail-sglang-join-profile.log

Scheduler CPU/CUDA traces, driver CPU traces, answers, and process timings
are saved on quail-results under /results/ablations/sglang-join-profile-<UTC>/.
Profiler timings are diagnostic and do not replace benchmark measurements.
"""

import json
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path

from experiments.sglang_join_profile_worker import PREDICTION
from quail.bench.quailb_parallel import VOLUMES, app, results_vol, sglang_image


@app.function(
    image=sglang_image.add_local_python_source("experiments"),
    gpu="H100!", memory=98304, timeout=3600, volumes=VOLUMES,
)
def profile():
    """Run the profiler in a child process and save its traces."""
    from experiments.sglang_join_profile_worker import profile_worker
    from quail.bench.process_isolation import _stop_process_group

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(f"/results/ablations/sglang-join-profile-{stamp}")
    root.mkdir(parents=True)
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=profile_worker, args=(str(root), sender))
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
def profile_joins():
    """Start the join profile and print its saved result path."""
    print(f"prediction: {PREDICTION}", flush=True)
    call = profile.spawn()
    print(f"function call id: {call.object_id}", flush=True)
    print(f"result volume path: {call.get()}", flush=True)
