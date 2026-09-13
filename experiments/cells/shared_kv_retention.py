"""Compare shared and first-anchor-only FEV-9 retention on one Modal H100.

Prepare the baseline checkout and tee the invocation:

    git worktree add --detach /tmp/quail-first-anchor-baseline-02bd7a2 02bd7a2
    uv run modal run experiments/cells/shared_kv_retention.py \
      --prediction "State the prediction before running." \
      2>&1 | tee /tmp/quail-shared-kv-retention.log

Set QUAIL_BASELINE_DIR to use another location for the baseline checkout.
Results and answer tables are saved on the quail-results volume.
"""

import inspect
import os
from pathlib import Path

import modal

baseline = Path(os.environ.get(
    "QUAIL_BASELINE_DIR", "/tmp/quail-first-anchor-baseline-02bd7a2"))
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub", "numpy", "pyarrow",
                 "sqlglot>=27.0", "bpe-qwen>=0.1.5", "datasets>=5.0.1")
    .env({
        "QUAIL_CACHE_DIR": "/root/.cache/kernels",
        "VLLM_CACHE_ROOT": "/root/.cache/kernels/vllm",
        "VLLM_LOGGING_LEVEL": "WARNING",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
        "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
        "TRITON_CACHE_DIR": "/root/.cache/kernels/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/kernels/torchinductor",
    })
    .add_local_python_source("quail", "quail_b")
).add_local_dir(
    baseline / "quail", "/opt/quail-baseline/quail", ignore=["__pycache__", "*.pyc"]
)
app = modal.App("quail-milestone1")
results = modal.Volume.from_name("quail-results")
volumes = {
    "/results": results,
    "/root/.cache/huggingface": modal.Volume.from_name("quail-hf-cache"),
    "/root/.cache/kernels": modal.Volume.from_name("quail-kernel-cache"),
}


def _run(label, output_dir):
    """Warm the selected implementation and measure FEV-9."""
    import hashlib
    import json
    from pathlib import Path

    import pyarrow.parquet as pq

    import quail
    from quail.backends.quail import expected_join_nodes
    from quail.bench.quailb import queries, register_sets
    from quail.physical import AiFilter, AiJoin
    from quail.planner.plan import EngineConfig
    from quail.specs import H100_USD_PER_HOUR
    from quail_b.data import build_sets

    output = Path(output_dir) / label
    output.mkdir(parents=True, exist_ok=True)
    source = Path(quail.__file__).parent
    source_hash = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        source_hash.update(str(path.relative_to(source)).encode())
        source_hash.update(path.read_bytes())
    with quail.Session(EngineConfig(model="qwen3-4b-fp8", gpus=1),
                       ) as session:
        register_sets(session, build_sets("/results/quailb_data", 0.1, 1))
        query = queries(session)["FEV-9"][1]()
        plan = query.plan()
        estimates = {
            "seconds": plan.estimated_seconds,
            "settings": plan.settings,
            "filter_order": [node.alias for node in plan.nodes
                             if isinstance(node, AiFilter)],
            "retained_filter_aliases": [node.alias for node in plan.nodes
                                        if isinstance(node, AiFilter)
                                        and node.keep_kv],
            "joins": [{"anchor": node.anchor,
                       "predicates": [stage.written_pos for stage in node.stages]}
                      for node in expected_join_nodes(plan)
                      if isinstance(node, AiJoin)],
        }
        print(f"[{label}] estimates before inference: {json.dumps(estimates)}",
              flush=True)
        warmup = query.run()
        warmup.count()
        result = query.run()
        row_count = result.count()
        for kind, tables in result.answer_tables.items():
            for key, answers in tables.items():
                name = "-".join(map(str, key)) if isinstance(key, tuple) else str(key)
                pq.write_table(answers, output / f"{kind}-{name}.parquet")
        pairs = sum(stage["tuples"] for stage in result.report["stages"]
                    if stage["op"] == "join")
        seconds = result.report["wall_s"]
        record = {
            "label": label, "query": "FEV-9", "source_sha256": source_hash.hexdigest(),
            "sf": 0.1, "lf": 1, "model": "qwen3-4b-fp8", "gpus": 1,
            "estimated_plan": estimates, "report": result.report,
            "warmup_report": warmup.report, "rows": row_count,
            "query_seconds": seconds, "evaluated_document_pairs": pairs,
            "document_pairs_per_second": pairs / seconds,
            "usd_per_query": seconds / 3600 * H100_USD_PER_HOUR,
        }
        (output / "summary.json").write_text(json.dumps(record, indent=2))
        print(f"[{label}] {seconds} seconds, {pairs} pairs, {row_count} rows",
              flush=True)


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600, volumes=volumes)
def compare(prediction: str) -> str:
    import json
    import subprocess
    import sys
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(f"/results/ablations/shared-kv-retention-{stamp}")
    output.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader",
    ], text=True).strip()
    (output / "setup.json").write_text(json.dumps({
        "prediction": prediction, "baseline_commit": "02bd7a2",
        "gpu_uuid": gpu, "order": ["first_anchor", "shared"],
        "warmup": "One unmeasured FEV-9 run per implementation",
    }, indent=2))
    for label, python_path in (("first_anchor", "/opt/quail-baseline:/root"),
                               ("shared", "/root")):
        environment = {**os.environ, "PYTHONPATH": os.pathsep.join(
            [python_path, *(path for path in sys.path if path)]
        )}
        completed = subprocess.run([
            sys.executable, "-c",
            inspect.getsource(_run) + "\nimport sys\n_run(*sys.argv[1:])",
            label, str(output),
        ], env=environment, cwd="/tmp", check=False)
        results.commit()
        if completed.returncode:
            raise RuntimeError(f"{label} exited with status {completed.returncode}")
    return str(output)


@app.local_entrypoint()
def main(prediction: str):
    if not prediction:
        raise ValueError("state the prediction before running")
    call = compare.spawn(prediction)
    print(f"function call id: {call.object_id}", flush=True)
    print(f"result volume path: {call.get()}", flush=True)
