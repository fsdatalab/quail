"""Run QUAIL-B with one H100! container per query chunk.

The four container command uses the same equal query split as the stock vLLM
runner. Each container loads one model and runs its queries in order.

    run_log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-quailb-parallel.log"
    uv run modal run -m quail.bench.quailb_parallel \
      --model qwen3-4b-fp8 --sf 0.1 --containers 4 \
      --prediction "State the expected result before starting." \
      2>&1 | tee "$run_log"
"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import modal

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install(
        "vllm==0.26.0",
        "huggingface_hub[hf_transfer]",
        "transformers>=5.2.0",
        "pandas",
        "pyarrow",
        "numpy",
        "datasets",
    )
    .env({
        "VLLM_LOGGING_LEVEL": "WARNING",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
        "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
        "TRITON_CACHE_DIR": "/root/.cache/kernels/triton",
    })
    .add_local_python_source("quail")
)

app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name(
    "quail-results", create_if_missing=True)
kernel_cache = modal.Volume.from_name(
    "quail-kernel-cache", create_if_missing=True)

VOLUMES = {
    "/root/.cache/huggingface": hf_cache,
    "/root/.cache/kernels": kernel_cache,
    "/results": results_vol,
}
DATA_DIR = "/results/quailb_data"


@app.function(image=image, timeout=1200, volumes=VOLUMES)
def ensure_data(sf: float, lf: int):
    from quail.bench.quailb import build_sets

    build_sets(DATA_DIR, sf, lf)
    results_vol.commit()


@app.function(
    image=image,
    gpu="H100!",
    memory=98304,
    timeout=7200,
    max_containers=8,
    volumes=VOLUMES,
)
def run_query_chunk(
    model: str,
    sf: float,
    lf: int,
    query_ids_csv: str,
    chunk_index: int,
    run_label: str,
    prediction: str,
    ground_truth_collection: str | None,
    ground_truth_workload: str | None,
) -> str:
    from quail.bench.quailb import run_suite
    from quail.runtime.worker import _execute_payload

    query_ids = tuple(
        query_id.strip() for query_id in query_ids_csv.split(",")
        if query_id.strip())
    out_path = (
        f"/results/benchmarks/quailb/chunks/{run_label}/"
        f"chunk-{chunk_index}.json")
    suite = run_suite(
        DATA_DIR,
        sf=sf,
        lf=lf,
        gpus=1,
        only=query_ids,
        out_path=out_path,
        model=model,
        accuracy=True,
        ground_truth_collection=ground_truth_collection,
        ground_truth_workload=ground_truth_workload,
        prediction=prediction,
        artifact_stem=f"{run_label}-chunk-{chunk_index}",
        execute=_execute_payload,
    )
    suite["query_chunk"] = {
        "index": chunk_index,
        "query_ids": list(query_ids),
    }
    Path(out_path).write_text(json.dumps(suite, indent=2))
    results_vol.commit()
    return json.dumps(suite)


@app.function(image=image, timeout=300, volumes=VOLUMES)
def save_report(report_json: str, volume_path: str) -> str:
    path = Path("/results") / volume_path.lstrip("/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_json)
    results_vol.commit()
    return f"/results/{volume_path.lstrip('/')}"


def _merge_suites(parts, query_ids, run_label, started, elapsed,
                  function_call_ids):
    from quail.bench.evaluate import summarize_queries

    base = parts[0]
    for part in parts[1:]:
        for field in ("sf", "lf", "gpus", "model", "corpus_id",
                      "ground_truth"):
            if part[field] != base[field]:
                raise ValueError(f"query chunks disagree on {field}")

    by_query = {}
    for part in parts:
        for row in part["passes"]["single"]["queries"]:
            if row["query"] in by_query:
                raise ValueError(f"duplicate query {row['query']}")
            by_query[row["query"]] = row
    missing = [query_id for query_id in query_ids
               if query_id not in by_query]
    if missing:
        raise ValueError(f"missing queries {missing}")
    rows = [by_query[query_id] for query_id in query_ids]

    run_id = (
        f"qb_{started.strftime('%Y%m%dT%H%M%SZ')}_"
        f"{uuid.uuid4().hex[:8]}")
    aggregate_path = f"benchmarks/quailb/runs/{run_id}/{run_label}.json"
    merged = dict(base)
    merged.update(
        run_id=run_id,
        artifact_stem=run_label,
        started_at=started.isoformat(),
        finished_at=datetime.now(timezone.utc).isoformat(),
        raw_volume_path=f"/results/benchmarks/quailb/runs/{run_id}",
        aggregate_volume_path=f"/results/{aggregate_path}",
        passes={
            "single": {
                "queries": rows,
                "pass_wall_s": round(elapsed, 1),
                "summary": summarize_queries(
                    rows,
                    base["pricing"]["h100_usd_per_hour"],
                    base["gpus"],
                ),
            }
        },
        parallel={
            "containers": len(parts),
            "gpu_per_container": "H100!",
            "model_copies": len(parts),
            "chunks": [part["query_chunk"] for part in parts],
            "function_call_ids": function_call_ids,
            "part_run_ids": [part["run_id"] for part in parts],
            "part_raw_volume_paths": [
                part["raw_volume_path"] for part in parts],
        },
    )
    merged["pricing"] = dict(
        base["pricing"], concurrent_h100_containers=len(parts))
    return merged, aggregate_path


@app.local_entrypoint()
def main(
    model: str = "qwen3-4b-fp8",
    sf: float = 0.1,
    lf: int = 1,
    query: str = "",
    containers: int = 4,
    prediction: str = "",
    ground_truth_collection: str = "",
    ground_truth_workload: str = "",
):
    from quail.bench.quailb import QUERY_ORDER, split_query_ids

    if not prediction:
        raise ValueError("pass --prediction before starting the benchmark")
    if containers < 1 or containers > 8:
        raise ValueError("containers must be between 1 and 8")
    query_ids = tuple(
        item.strip() for item in query.split(",") if item.strip()
    ) if query else QUERY_ORDER
    unknown = sorted(set(query_ids) - set(QUERY_ORDER))
    if unknown:
        raise ValueError(f"unknown queries {unknown}")

    started = datetime.now(timezone.utc)
    run_label = (
        f"{started.strftime('%Y%m%dT%H%M%SZ')}-quailb-sf{sf}-lf{lf}-"
        f"{model}-parallel{containers}")

    data_call = ensure_data.spawn(sf, lf)
    print(f"function call id: {data_call.object_id} (data)", flush=True)
    data_call.get()

    chunks = split_query_ids(query_ids, containers)
    calls = []
    call_ids = []
    t0 = time.time()
    for index, chunk in enumerate(chunks):
        call = run_query_chunk.spawn(
            model=model,
            sf=sf,
            lf=lf,
            query_ids_csv=",".join(chunk),
            chunk_index=index,
            run_label=run_label,
            prediction=prediction,
            ground_truth_collection=(ground_truth_collection or None),
            ground_truth_workload=(ground_truth_workload or None),
        )
        calls.append(call)
        call_ids.append(call.object_id)
        print(
            f"function call id: {call.object_id} "
            f"(chunk {index}, {','.join(chunk)})",
            flush=True,
        )

    parts = []
    for index, call in enumerate(calls):
        print(f"waiting for chunk {index}", flush=True)
        parts.append(json.loads(call.get()))
        print(f"chunk {index} finished", flush=True)
    elapsed = time.time() - t0

    report, aggregate_path = _merge_suites(
        parts, query_ids, run_label, started, elapsed, call_ids)
    save_call = save_report.spawn(
        json.dumps(report, indent=2), aggregate_path)
    print(f"function call id: {save_call.object_id} (save)", flush=True)
    saved = save_call.get()
    summary = report["passes"]["single"]["summary"]
    print(f"saved {saved}", flush=True)
    print(json.dumps({
        "run_id": report["run_id"],
        "aggregate_volume_path": report["aggregate_volume_path"],
        "queries_completed": summary["queries_completed"],
        "queries_failed": summary["queries_failed"],
        "query_runtime_s": summary["query_runtime_s"],
        "parallel_wall_s": round(elapsed, 1),
        "function_call_ids": call_ids,
    }, indent=2), flush=True)
