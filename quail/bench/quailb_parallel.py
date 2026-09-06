"""Run Quail and vLLM on one physical H100 per query family.

Quail runs in a fresh child process. Stock vLLM and pipelined vLLM run
in a second child process and share one loaded model. SGLang runs in a
separate container because its package version conflicts with vLLM.

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

from quail.runtime.worker import build_worker_image


image = build_worker_image()
sglang_image = build_worker_image(runtime_package="sglang==0.5.19")

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
    include_baselines: bool,
) -> str:
    from quail.bench.process_isolation import (
        run_backend_group_in_fresh_process,
    )
    from quail.bench.quailb import query_family_name

    query_ids = tuple(
        query_id.strip() for query_id in query_ids_csv.split(",")
        if query_id.strip())
    family = query_family_name(query_ids)
    process_groups = [("quail",)]
    if include_baselines:
        process_groups.append(("stock_vllm", "pipelined_vllm"))

    process_results = []
    suites = {}
    for methods in process_groups:
        process_result = run_backend_group_in_fresh_process(
            data_dir=DATA_DIR,
            model=model,
            sf=sf,
            lf=lf,
            query_ids=query_ids,
            run_label=run_label,
            prediction=prediction,
            ground_truth_collection=ground_truth_collection,
            methods=methods,
        )
        process_results.append(process_result)
        suites.update(process_result["suites"])

    gpu_uuids = {
        gpu_uuid
        for process_result in process_results
        for gpu_uuid in process_result["gpu_uuids"]
    }
    if len(gpu_uuids) != 1:
        raise RuntimeError(
            "backend child processes did not see one physical GPU: "
            f"{sorted(gpu_uuids)}"
        )

    family_result = {
        "query_family": family,
        "query_ids": list(query_ids),
        "ground_truth_collection": suites["quail"]["ground_truth"][
            "collection_id"
        ],
        "ground_truth_workload": None,
        "gpu": "H100!",
        "same_modal_container": True,
        "same_physical_gpu": True,
        "gpu_uuids": sorted(gpu_uuids),
        "process_groups": [list(methods) for methods in process_groups],
        "process_cleanup": [
            {
                "methods": process_result["methods"],
                **process_result["process_cleanup"],
            }
            for process_result in process_results
        ],
        "engine_order": [
            method for methods in process_groups for method in methods
        ],
        "suites": suites,
    }
    family_path = (
        f"/results/benchmarks/quailb/families/{run_label}/"
        f"{family}-same-gpu.json")
    Path(family_path).write_text(json.dumps(family_result, indent=2))
    results_vol.commit()
    return json.dumps(family_result)


@app.function(
    image=sglang_image,
    gpu="H100!",
    memory=98304,
    timeout=21600,
    max_containers=8,
    volumes=VOLUMES,
)
def run_sglang_query_family(
    model: str,
    sf: float,
    lf: int,
    query_ids_csv: str,
    run_label: str,
    prediction: str,
    ground_truth_collection: str,
) -> str:
    """Run one query family through the SGLang backend."""
    from quail.bench.process_isolation import run_backend_group

    query_ids = tuple(
        query_id.strip() for query_id in query_ids_csv.split(",")
        if query_id.strip()
    )
    result = run_backend_group(
        data_dir=DATA_DIR,
        model=model,
        sf=sf,
        lf=lf,
        query_ids=query_ids,
        run_label=run_label,
        prediction=prediction,
        ground_truth_collection=ground_truth_collection,
        methods=("pipelined_sglang",),
    )
    results_vol.commit()
    return json.dumps(result)


def _merge_suites(parts, query_ids, run_label, started, elapsed,
                  function_call_ids, methods):
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
            "methods_per_container": list(methods),
            "engine_order": list(methods),
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


@app.function(image=image, timeout=43200, memory=4096, volumes=VOLUMES)
def run_all(
    model: str = "qwen3-4b-fp8",
    sf: float = 0.1,
    lf: int = 1,
    query: str = "",
    prediction: str = "",
    ground_truth_collection: str = "",
    include_baselines: bool = True,
    include_sglang: bool = True,
):
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
        f"{model}-"
        f"{'families' if include_baselines else 'quail-only'}")

    data_call = ensure_data.spawn(sf, lf)
    print(f"function call id: {data_call.object_id} (data)", flush=True)
    data_call.get()

    families = split_query_families(query_ids)
    family_calls = []
    sglang_calls = []
    call_ids = {}
    t0 = time.time()
    for family_ids in families:
        family = query_family_name(family_ids)
        family_call = run_query_family.spawn(
            model=model,
            sf=sf,
            lf=lf,
            query_ids_csv=",".join(family_ids),
            run_label=run_label,
            prediction=prediction,
            ground_truth_collection=ground_truth_collection,
            include_baselines=include_baselines,
        )
        family_calls.append((family, family_call))
        call_ids[f"{family}:quail_vllm"] = family_call.object_id
        print(
            f"function call id: {family_call.object_id} "
            f"({family}, Quail and vLLM, {','.join(family_ids)})",
            flush=True,
        )
        if include_baselines and include_sglang:
            sglang_call = run_sglang_query_family.spawn(
                model=model,
                sf=sf,
                lf=lf,
                query_ids_csv=",".join(family_ids),
                run_label=run_label,
                prediction=prediction,
                ground_truth_collection=ground_truth_collection,
            )
            sglang_calls.append((family, sglang_call))
            call_ids[f"{family}:sglang"] = sglang_call.object_id
            print(
                f"function call id: {sglang_call.object_id} "
                f"({family}, SGLang, {','.join(family_ids)})",
                flush=True,
            )

    family_parts = []
    for family, call in family_calls:
        print(f"waiting for {family} Quail and vLLM", flush=True)
        family_parts.append(json.loads(call.get()))
        print(f"{family} Quail and vLLM finished", flush=True)
    sglang_parts = []
    for family, call in sglang_calls:
        print(f"waiting for {family} SGLang", flush=True)
        sglang_parts.append(json.loads(call.get()))
        print(f"{family} SGLang finished", flush=True)
    elapsed = time.time() - t0

    methods = (
        (
            "quail",
            "stock_vllm",
            "pipelined_vllm",
            *(("pipelined_sglang",) if include_sglang else ()),
        )
        if include_baselines else ("quail",)
    )
    reports = {}
    paths = {}
    for method in methods:
        if method == "pipelined_sglang":
            method_parts = [
                part["suites"][method] for part in sglang_parts
            ]
            container_methods = (method,)
        else:
            method_parts = [
                part["suites"][method] for part in family_parts
            ]
            container_methods = tuple(
                item
                for group in family_parts[0]["process_groups"]
                for item in group
            )
        report, path = _merge_suites(
            method_parts,
            query_ids,
            f"{run_label}-{method}",
            started,
            elapsed,
            call_ids,
            container_methods,
        )
        reports[method] = report
        paths[method] = path

    family_metadata = {
        "id": run_label,
        "one_container_per_query_family": not sglang_calls,
        "same_gpu_for_all_methods_in_family": not sglang_calls,
        "same_gpu_for_quail_and_vllm": include_baselines,
        "gpu": "H100!",
        "methods": list(methods),
        "engine_order": list(methods),
        "baseline_order": list(methods[1:]),
        "all_family_queries_finish_before_next_method": True,
        "quail_and_vllm_share_a_container": include_baselines,
        "quail_uses_a_fresh_process": True,
        "vllm_backends_share_a_process": include_baselines,
        "sglang_uses_a_separate_container": bool(sglang_calls),
        "quail_kv_cleared_before_each_query": True,
        "vllm_prefix_cache_reset_before_each_configuration": (
            include_baselines
        ),
        "family_function_call_ids": call_ids,
        "families": [
            {
                "query_family": part["query_family"],
                "query_ids": part["query_ids"],
                "ground_truth_collection":
                    part["ground_truth_collection"],
                "gpu_uuids": part["gpu_uuids"],
                "process_groups": part["process_groups"],
                "process_cleanup": part["process_cleanup"],
            }
            for part in family_parts
        ],
    }

    for report in reports.values():
        report["family_run"] = family_metadata

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
    saved = {}
    for name, (payload, path) in payloads.items():
        volume_file = Path("/results") / path
        volume_file.parent.mkdir(parents=True, exist_ok=True)
        volume_file.write_text(payload)
        saved[name] = f"/results/{path}"
    results_vol.commit()

    summary = reports["quail"]["passes"]["single"]["summary"]
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


@app.local_entrypoint()
def main(
    model: str = "qwen3-4b-fp8",
    sf: float = 0.1,
    lf: int = 1,
    query: str = "",
    prediction: str = "",
    ground_truth_collection: str = "",
    include_baselines: bool = True,
    include_sglang: bool = True,
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
        include_baselines=include_baselines,
        include_sglang=include_sglang,
    )
    print(f"function call id: {call.object_id} (all families)", flush=True)
    print(call.get(), flush=True)
