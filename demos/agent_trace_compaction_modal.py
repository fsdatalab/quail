"""Run the agent trace compaction demo on Modal H100 GPUs.

uv run modal run --detach demos/agent_trace_compaction_modal.py \
  --limit 100 --seed 42 \
  2>&1 | tee /tmp/quail-agent-compaction.log
"""

from __future__ import annotations

import uuid
from pathlib import Path

import modal

from quail.bench.images import cpu_image, gpu_image

app = modal.App("quail-milestone1")
results_volume = modal.Volume.from_name("quail-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache", create_if_missing=True)

preparation_image = cpu_image().add_local_python_source("demos")
inference_image = gpu_image().add_local_python_source("demos")

RESULTS_DIR = Path("/vol/results/demos/agent-compaction")


@app.function(image=preparation_image, timeout=86_400, memory=16_384,
              volumes={"/vol/results": results_volume,
                       "/root/.cache/huggingface": hf_cache})
def prepare(limit: int, seed: int) -> str:
    """Sample trajectories and write the Parquet inputs."""
    from demos.agent_trace_compaction import prepare as run_prepare

    directory = RESULTS_DIR / uuid.uuid4().hex
    run_prepare(directory, limit, seed)
    results_volume.commit()
    hf_cache.commit()
    print(f"result volume path: {directory}", flush=True)
    return str(directory)


@app.function(image=inference_image, gpu="H100!", timeout=86_400, memory=65_536,
              volumes={"/vol/results": results_volume,
                       "/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache})
def evaluate(directory: str, gpus: int) -> dict:
    """Run the Quail join on a Modal GPU."""
    from demos.agent_trace_compaction import evaluate as run_evaluate

    results_volume.reload()
    report = run_evaluate(Path(directory), gpus)
    results_volume.commit()
    return report


@app.function(image=preparation_image, timeout=86_400, memory=16_384,
              volumes={"/vol/results": results_volume,
                       "/root/.cache/huggingface": hf_cache})
def evaluate_remote(directory: str, gpus: int, endpoint: str) -> dict:
    """Submit the join to a deployed query service and wait for it.

    Runs without a GPU: the service owns execution. This function
    uploads the inputs, watches the saved status, and fetches the result.
    """
    from demos.agent_trace_compaction import evaluate as run_evaluate

    results_volume.reload()
    report = run_evaluate(Path(directory), gpus, endpoint)
    results_volume.commit()
    return report


@app.function(image=preparation_image, timeout=86_400, memory=32_768,
              volumes={"/vol/results": results_volume})
def reconstruct(directory: str) -> dict:
    """Apply retention decisions and save compacted conversations."""
    from demos.agent_trace_compaction import reconstruct as run_reconstruct

    results_volume.reload()
    summary = run_reconstruct(Path(directory))
    results_volume.commit()
    return summary


@app.local_entrypoint()
def main(limit: int = 100, seed: int = 42, gpus: int = 1, endpoint: str = ""):
    """Compact complete traces using DiffusionGemma on one or more H100s.

    With ``--endpoint`` the join is submitted to a deployed query
    service instead of a GPU function here; the log shows the query id
    and its status page.
    """
    if limit < 1 or gpus not in (1, 2, 4, 8):
        raise ValueError("limit must be positive; gpus must be 1, 2, 4, or 8")
    call = prepare.spawn(limit, seed)
    print(f"function call id (prepare): {call.object_id}", flush=True)
    directory = call.get()
    if endpoint:
        call = evaluate_remote.spawn(directory, gpus, endpoint)
    else:
        call = evaluate.with_options(gpu=f"H100!:{gpus}").spawn(directory, gpus)
    print(f"function call id (evaluate): {call.object_id}", flush=True)
    call.get()
    call = reconstruct.spawn(directory)
    print(f"function call id (reconstruct): {call.object_id}", flush=True)
    call.get()
