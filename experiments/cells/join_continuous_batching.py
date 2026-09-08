r"""Compare pre-planned and continuous join batching on one Modal H100.

The baseline checkout is main before JoinAdmission (d7a96e0). Both
implementations run IMDB-8, FEV-7, and FEV-9 in the same container on
the same GPU, one warmup and one measured run each.

Prepare the baseline checkout and tee the invocation:

    git worktree add --detach /tmp/quail-preplanned-baseline-d7a96e0 d7a96e0
    uv run modal run experiments/cells/join_continuous_batching.py \
      --prediction "State the prediction before running." \
      2>&1 | tee /tmp/quail-join-continuous-batching.log

Set QUAIL_BASELINE_DIR to use another location for the baseline checkout.
Results and answer tables are saved on the quail-results volume.
"""

import inspect
import os
from pathlib import Path

import modal

from quail.runtime.worker import build_worker_image

QUERIES = ("IMDB-8", "FEV-7", "FEV-9")
BASELINE_COMMIT = "d7a96e0"

baseline = Path(os.environ.get(
    "QUAIL_BASELINE_DIR", f"/tmp/quail-preplanned-baseline-{BASELINE_COMMIT}"))
image = build_worker_image(local_python_sources=("quail_b",)).add_local_dir(
    baseline / "quail", "/opt/quail-baseline/quail",
    ignore=["__pycache__", "*.pyc"])
app = modal.App("quail-milestone1")
results = modal.Volume.from_name("quail-results")
volumes = {
    "/results": results,
    "/root/.cache/huggingface": modal.Volume.from_name("quail-hf-cache"),
    "/root/.cache/kernels": modal.Volume.from_name("quail-kernel-cache"),
}


def _run(label, output_dir, query_ids):
    """Warm the selected implementation and measure each query."""
    import hashlib
    import json
    from dataclasses import asdict
    from pathlib import Path

    import pyarrow.parquet as pq

    import quail
    from quail.bench.quailb import queries, register_sets
    from quail.planner.plan import EngineConfig
    from quail.runtime.compute import InProcessComputeProvider
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
                       compute_provider=InProcessComputeProvider()) as session:
        register_sets(session, build_sets("/results/quailb_data", 0.1, 1))
        defs = queries(session)
        for qid in query_ids.split(","):
            query = defs[qid][1]()
            warmup = query.run()
            warmup.count()
            query = defs[qid][1]()
            result = query.run()
            row_count = result.count()
            qdir = output / qid
            qdir.mkdir(exist_ok=True)
            for kind, tables in result.answer_tables.items():
                for key, answers in tables.items():
                    name = ("-".join(map(str, key)) if isinstance(key, tuple)
                            else str(key))
                    pq.write_table(answers, qdir / f"{kind}-{name}.parquet")
            pairs = sum(stage["tuples"] for stage in result.report["stages"]
                        if stage["op"] == "join")
            seconds = result.report["wall_s"]
            nodes = {}
            for node in result.plan.topological_nodes():
                metrics = result.node_metrics.get(node.node_id)
                nodes[node.node_id] = dict(
                    type=node.type_name, fields=node.explain_fields(),
                    metrics=(None if metrics is None else {
                        k: v for k, v in asdict(metrics).items()
                        if k != "extension"}))
            record = {
                "label": label, "query": qid,
                "source_sha256": source_hash.hexdigest(),
                "sf": 0.1, "lf": 1, "model": "qwen3-4b-fp8", "gpus": 1,
                "report": result.report, "warmup_report": warmup.report,
                "explain": result.explain(), "nodes": nodes,
                "rows": row_count, "query_seconds": seconds,
                "evaluated_document_pairs": pairs,
                "document_pairs_per_second": pairs / seconds,
                "usd_per_query": seconds / 3600 * H100_USD_PER_HOUR,
            }
            (qdir / "summary.json").write_text(json.dumps(record, indent=2))
            print(f"[{label}] {qid}: {seconds} seconds, {pairs} pairs, "
                  f"{row_count} rows, warmup {warmup.report['wall_s']} s",
                  flush=True)


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=volumes)
def compare(prediction: str, query_ids: str) -> str:
    import json
    import subprocess
    import sys
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(f"/results/ablations/join-continuous-batching-{stamp}")
    output.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader",
    ], text=True).strip()
    (output / "setup.json").write_text(json.dumps({
        "prediction": prediction, "baseline_commit": BASELINE_COMMIT,
        "queries": query_ids.split(","),
        "gpu_uuid": gpu, "order": ["preplanned", "continuous"],
        "warmup": "One unmeasured run per query per implementation",
    }, indent=2))
    for label, python_path in (("preplanned", "/opt/quail-baseline:/root"),
                               ("continuous", "/root")):
        environment = {**os.environ, "PYTHONPATH": os.pathsep.join(
            [python_path, *(path for path in sys.path if path)]
        )}
        completed = subprocess.run([
            sys.executable, "-c",
            inspect.getsource(_run) + "\nimport sys\n_run(*sys.argv[1:])",
            label, str(output), query_ids,
        ], env=environment, cwd="/tmp", check=False)
        results.commit()
        if completed.returncode:
            raise RuntimeError(
                f"{label} exited with status {completed.returncode}")
    return str(output)


@app.local_entrypoint()
def main(prediction: str, queries: str = ",".join(QUERIES)):
    if not prediction:
        raise ValueError("state the prediction before running")
    call = compare.spawn(prediction, queries)
    print(f"function call id: {call.object_id}", flush=True)
    print(f"result volume path: {call.get()}", flush=True)
