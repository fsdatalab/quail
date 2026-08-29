"""Run all three QUAIL-B methods in one H100! container per family.

Each family container runs Quail first. It then releases Quail GPU state,
loads vLLM once, and runs stock vLLM and pipelined vLLM.

    run_log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-quailb-parallel.log"
    uv run modal run --detach -m quail.bench.quailb_parallel \
      --model qwen3-4b-fp8 --sf 0.1 \
      --prediction "State the expected result before starting." \
      2>&1 | tee "$run_log"

The local entrypoint starts one remote function. Detached mode keeps that
function and every family call running if the local process disconnects.
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
    .add_local_python_source("quail", "baselines")
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
    timeout=21600,
    max_containers=8,
    volumes=VOLUMES,
)
def run_query_family(
    model: str,
    sf: float,
    lf: int,
    query_ids_csv: str,
    run_label: str,
    prediction: str,
    ground_truth_collection: str,
) -> str:
    from baselines.stock_vllm.run import BASELINES, _run_query_batches
    from quail.bench.quailb import query_family_name, run_suite
    from quail.runtime.worker import (
        _execute_payload,
        release_booted_models,
    )

    query_ids = tuple(
        query_id.strip() for query_id in query_ids_csv.split(",")
        if query_id.strip())
    family = query_family_name(query_ids)
    workload = None if ground_truth_collection else family
    out_path = (
        f"/results/benchmarks/quailb/families/{run_label}/"
        f"{family}-quail.json")
    print(
        f"[{family}] running Quail for {len(query_ids)} queries",
        flush=True,
    )
    suite = run_suite(
        DATA_DIR,
        sf=sf,
        lf=lf,
        gpus=1,
        only=query_ids,
        out_path=out_path,
        model=model,
        accuracy=True,
        ground_truth_collection=ground_truth_collection or None,
        ground_truth_workload=workload,
        prediction=prediction,
        artifact_stem=f"{run_label}-{family}-quail",
        execute=_execute_payload,
    )
    suite["query_family"] = {
        "name": family,
        "query_ids": list(query_ids),
    }
    Path(out_path).write_text(json.dumps(suite, indent=2))
    results_vol.commit()

    transition = release_booted_models()
    print(
        f"[{family}] released Quail GPU state: {transition}",
        flush=True,
    )
    print(
        f"[{family}] running stock vLLM and pipelined vLLM",
        flush=True,
    )
    baseline_output = _run_query_batches(
        model=model,
        sf=sf,
        query_ids_csv=",".join(query_ids),
        reps=1,
        ground_truth_workload=workload or "",
        prediction=prediction,
        baselines=BASELINES,
        paired_run_id=run_label,
        lf=lf,
        method_order="method-major",
        ground_truth_collection=(ground_truth_collection or None),
    )
    family_result = {
        "query_family": family,
        "query_ids": list(query_ids),
        "ground_truth_collection": suite["ground_truth"]["collection_id"],
        "ground_truth_workload": workload,
        "gpu": "H100!",
        "same_modal_container": True,
        "engine_order": ["quail", "vllm_baselines"],
        "baseline_order": ["stock_vllm", "pipelined_vllm"],
        "engine_transition": transition,
        "quail": suite,
        "baseline_reports": baseline_output["reports"],
        "baseline_execution_order": baseline_output["execution_order"],
    }
    family_path = (
        f"/results/benchmarks/quailb/families/{run_label}/"
        f"{family}.json")
    Path(family_path).write_text(json.dumps(family_result, indent=2))
    results_vol.commit()
    return json.dumps(family_result)


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
        for field in ("sf", "lf", "gpus", "model", "corpus_id"):
            if part[field] != base[field]:
                raise ValueError(f"query families disagree on {field}")

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
    scored = all(
        "tokens_processed" in row
        for row in rows if "error" not in row
    )
    if scored:
        summary = summarize_queries(
            rows,
            base["pricing"]["h100_usd_per_hour"],
            base["gpus"],
        )
    else:
        good = [row for row in rows if "error" not in row]
        wall_s = sum(float(row["wall_s"]) for row in good)
        boot_s = sum(float(row.get("boot_s") or 0.0) for row in good)
        summary = {
            "queries_completed": len(good),
            "queries_failed": len(rows) - len(good),
            "query_runtime_s": round(wall_s, 2),
            "boot_s": round(boot_s, 2),
            "runtime_with_boot_s": round(wall_s + boot_s, 2),
            "fresh_tokens": sum(
                int(row.get("fresh_tokens") or 0) for row in good),
            "accuracy_scored": False,
        }

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
        ground_truth={
            part["query_family"]["name"]: part["ground_truth"]
            for part in parts
        },
        passes={
            "single": {
                "queries": rows,
                "pass_wall_s": round(elapsed, 1),
                "summary": summary,
            }
        },
        parallel={
            "containers": len(parts),
            "gpu_per_container": "H100!",
            "model_copies": len(parts),
            "container_strategy": "one per query family",
            "methods_per_container": [
                "quail", "stock_vllm", "pipelined_vllm"],
            "engine_order": ["quail", "vllm_baselines"],
            "families": [part["query_family"] for part in parts],
            "function_call_ids": function_call_ids,
            "part_run_ids": [part["run_id"] for part in parts],
            "part_raw_volume_paths": [
                part["raw_volume_path"] for part in parts],
        },
    )
    merged["pricing"] = dict(
        base["pricing"], concurrent_h100_containers=len(parts))
    return merged, aggregate_path


def _run_all(
    model: str = "qwen3-4b-fp8",
    sf: float = 0.1,
    lf: int = 1,
    query: str = "",
    prediction: str = "",
    ground_truth_collection: str = "",
):
    from baselines.stock_vllm.run import BASELINES, _merge_reports
    from quail.bench.quailb import (
        QUERY_ORDER,
        query_family_name,
        split_query_families,
    )

    if not prediction:
        raise ValueError("pass --prediction before starting the benchmark")
    query_ids = tuple(
        item.strip() for item in query.split(",") if item.strip()
    ) if query else QUERY_ORDER
    unknown = sorted(set(query_ids) - set(QUERY_ORDER))
    if unknown:
        raise ValueError(f"unknown queries {unknown}")

    started = datetime.now(timezone.utc)
    run_label = (
        f"{started.strftime('%Y%m%dT%H%M%SZ')}-quailb-sf{sf}-lf{lf}-"
        f"{model}-families")

    data_call = ensure_data.spawn(sf, lf)
    print(f"function call id: {data_call.object_id} (data)", flush=True)
    data_call.get()

    families = split_query_families(query_ids)
    calls = []
    call_ids = {}
    t0 = time.time()
    for family_ids in families:
        family = query_family_name(family_ids)
        call = run_query_family.spawn(
            model=model,
            sf=sf,
            lf=lf,
            query_ids_csv=",".join(family_ids),
            run_label=run_label,
            prediction=prediction,
            ground_truth_collection=ground_truth_collection,
        )
        calls.append((family, call))
        call_ids[family] = call.object_id
        print(
            f"function call id: {call.object_id} "
            f"({family}, {','.join(family_ids)})",
            flush=True,
        )

    parts = []
    for family, call in calls:
        print(f"waiting for {family}", flush=True)
        parts.append(json.loads(call.get()))
        print(f"{family} finished", flush=True)
    elapsed = time.time() - t0

    quail_parts = [part["quail"] for part in parts]
    quail_report, quail_path = _merge_suites(
        quail_parts, query_ids, run_label, started, elapsed, call_ids)

    family_metadata = {
        "id": run_label,
        "one_container_per_query_family": True,
        "same_gpu_for_all_methods_in_family": True,
        "gpu": "H100!",
        "methods": ["quail", *BASELINES],
        "engine_order": ["quail", "vllm_baselines"],
        "baseline_order": ["stock_vllm", "pipelined_vllm"],
        "all_family_queries_finish_before_next_method": True,
        "quail_released_before_vllm_load": True,
        "quail_kv_cleared_before_each_query": True,
        "vllm_prefix_cache_reset_before_each_configuration": True,
        "family_function_call_ids": call_ids,
        "families": [
            {
                "query_family": part["query_family"],
                "query_ids": part["query_ids"],
                "ground_truth_collection":
                    part["ground_truth_collection"],
                "engine_transition": part["engine_transition"],
            }
            for part in parts
        ],
    }

    reports = {"quail": quail_report}
    paths = {"quail": quail_path}
    for baseline in BASELINES:
        report = _merge_reports(
            [part["baseline_reports"][baseline] for part in parts],
            len(parts),
        )
        report["family_run"] = family_metadata
        report["ground_truth_collection"] = {
            part["query_family"]: part["ground_truth_collection"]
            for part in parts
        }
        report["ground_truth"] = {
            part["query_family"]:
                part["baseline_reports"][baseline].get("ground_truth")
            for part in parts
        }
        reports[baseline] = report
        paths[baseline] = f"{baseline}/{run_label}/summary.json"

    manifest_path = (
        f"benchmarks/quailb/family-runs/{run_label}/manifest.json")
    manifest = {
        "run_id": run_label,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "parallel_wall_s": round(elapsed, 1),
        "query_ids": list(query_ids),
        "family_run": family_metadata,
        "result_volume_paths": {
            method: f"/results/{path}" for method, path in paths.items()
        },
        "manifest_volume_path": f"/results/{manifest_path}",
    }

    payloads = {
        **{method: (json.dumps(report, indent=2), paths[method])
           for method, report in reports.items()},
        "manifest": (json.dumps(manifest, indent=2), manifest_path),
    }
    save_calls = {}
    for name, (payload, path) in payloads.items():
        call = save_report.spawn(payload, path)
        save_calls[name] = call
        print(
            f"function call id: {call.object_id} (save {name})",
            flush=True,
        )
    saved = {name: call.get() for name, call in save_calls.items()}

    summary = quail_report["passes"]["single"]["summary"]
    final = {
        "run_id": run_label,
        "saved": saved,
        "queries_completed": summary["queries_completed"],
        "queries_failed": summary["queries_failed"],
        "parallel_wall_s": round(elapsed, 1),
        "function_call_ids": call_ids,
    }
    print(json.dumps(final, indent=2), flush=True)
    return json.dumps(final)


@app.function(image=image, timeout=43200, memory=4096, volumes=VOLUMES)
def run_all(
    model: str,
    sf: float,
    lf: int,
    query: str,
    prediction: str,
    ground_truth_collection: str,
) -> str:
    """Run and merge every requested query family."""
    return _run_all(
        model=model,
        sf=sf,
        lf=lf,
        query=query,
        prediction=prediction,
        ground_truth_collection=ground_truth_collection,
    )


@app.local_entrypoint()
def main(
    model: str = "qwen3-4b-fp8",
    sf: float = 0.1,
    lf: int = 1,
    query: str = "",
    prediction: str = "",
    ground_truth_collection: str = "",
):
    if not prediction:
        raise ValueError("pass --prediction before starting the benchmark")
    call = run_all.spawn(
        model=model,
        sf=sf,
        lf=lf,
        query=query,
        prediction=prediction,
        ground_truth_collection=ground_truth_collection,
    )
    print(f"function call id: {call.object_id} (all families)", flush=True)
    print(call.get(), flush=True)
