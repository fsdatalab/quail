"""Measure how much stock and pipelined vLLM preempt and recompute on a query.

Each backend runs the query on its own H100 through the QUAIL-B runner,
with the baseline's engine settings. A vLLM plugin
(quail.baselines.vllm_trace) appends one line per scheduler step to a
trace file: tokens scheduled, tokens recomputed after a preemption,
requests preempted, requests running and waiting, and KV in use. Each
backend's run.json, steps.jsonl, and summary.json go to
/results/ablations/vllm_preemption/<run id>/<backend>/.

    uv run modal run --detach experiments/cells/vllm_preemption.py::main \\
        --query IMDB-26 2>&1 | tee "$run_log"
"""

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from bench.quailb_parallel import (
    DATA_DIR,
    VOLUMES,
    app,
    ensure_data,
    results_vol,
)
from quail.baselines.vllm_trace import STEP_TRACE_ENV
from quail.deploy.images import gpu_image

OUTPUT_DIR = "/results/ablations/vllm_preemption"
image = gpu_image()


def summarize(record: dict, steps: list[dict]) -> dict:
    """Combine the query's QUAIL-B metrics with its scheduler steps.

    Recompute seconds are an estimate: each step's duration, the time to
    the next step, split by the share of its tokens that were recomputed.
    """
    query = record["queries"][0]
    metrics = query.get("metrics", {})
    scheduled = sum(step["scheduled"] for step in steps)
    recomputed = sum(step["recomputed"] for step in steps)
    recompute_s = sum(
        (later["t"] - step["t"]) * step["recomputed"] / step["scheduled"]
        for step, later in zip(steps, steps[1:]) if step["scheduled"])
    minimum = metrics.get("minimum_tokens")
    return {
        "query": query["id"], "status": query["status"],
        "runtime_s": query.get("runtime_s"),
        "input_tokens_per_second": metrics.get("input_tokens_per_second"),
        "cost_usd": metrics.get("cost_usd"),
        "input_tokens": metrics.get("input_tokens"),
        "minimum_tokens": minimum,
        "regret_approximate": metrics.get("regret_approximate"),
        "num_steps": len(steps),
        "scheduled_tokens": scheduled,
        "recomputed_tokens": recomputed,
        "num_preemptions": sum(step["preempted"] for step in steps),
        "num_steps_preempting": sum(bool(step["preempted"]) for step in steps),
        "recompute_s_estimate": round(recompute_s, 2),
        "scheduled_over_minimum": (round(scheduled / minimum, 4)
                                   if minimum else None),
        "max_running": max((step["running"] for step in steps), default=0),
        "max_waiting": max((step["waiting"] for step in steps), default=0),
        "max_kv_usage": max((step["kv_usage"] for step in steps), default=0.0),
        "steps_at_full_kv": sum(step["kv_usage"] >= 0.99 for step in steps),
    }


@app.function(image=image, gpu="H100!", memory=98304, timeout=6 * 3600,
              volumes=VOLUMES)
def trace_backend(backend: str, query_id: str, scale_factor: float, model: str,
                  collection_id: str, run_dir: str) -> str:
    """Run one query on one vLLM backend and save its scheduler steps.

    Returns:
        The summary as JSON.
    """
    import quail
    from bench.quailb import run_suite

    directory = Path(run_dir) / backend
    trace = Path("/tmp") / f"steps-{backend}.jsonl"
    trace.unlink(missing_ok=True)
    # vLLM's engine process inherits the variable and loads the plugin
    os.environ[STEP_TRACE_ENV] = str(trace)
    config = quail.EngineConfig(gpus=1, model=model, backend=backend,
                                device="h100-sxm")
    record = run_suite(
        [query_id], scale_factor=scale_factor, config=config,
        data_dir=Path(DATA_DIR) / f"sf{scale_factor}",
        ground_truth_collection=collection_id or None,
        output_dir=str(directory))
    steps = [json.loads(line) for line in trace.read_text().splitlines()]
    shutil.copy(trace, directory / "steps.jsonl")
    summary = {"backend": backend, "model": model, "scale_factor": scale_factor,
               **summarize(record, steps)}
    (directory / "summary.json").write_text(json.dumps(summary, indent=1))
    results_vol.commit()
    return json.dumps(summary, indent=1)


@app.local_entrypoint()
def main(query: str = "IMDB-26", scale_factor: float = 0.1,
         model: str = "qwen3-4b-fp8",
         backends: str = "stock_vllm,pipelined_vllm"):
    collection_id = ensure_data.remote(scale_factor, [query], "")
    run_id = (f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
              f"{uuid.uuid4().hex[:8]}")
    run_dir = f"{OUTPUT_DIR}/{run_id}"
    print(f"run directory: {run_dir}", flush=True)
    calls = []
    for backend in filter(None, backends.split(",")):
        call = trace_backend.spawn(backend, query, scale_factor, model,
                                   collection_id, run_dir)
        print(f"function call id: {call.object_id} ({backend})", flush=True)
        calls.append(call)
    for call in calls:
        print(call.get(), flush=True)
