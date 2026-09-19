"""Run Quail and vLLM on one physical H100 per query family.

Quail runs in a fresh child process. Stock vLLM and pipelined vLLM run
in a second child process and share one loaded model. SGLang runs in a
separate container because its package version conflicts with vLLM.

    run_log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-quailb-parallel.log"
    uv run modal run --detach -m quail.bench.quailb_parallel \
      --model qwen3-4b-fp8 --sf 0.1 \
      2>&1 | tee "$run_log"

The local entrypoint starts one remote function. Detached mode keeps that
function and every family call running if the local process disconnects.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import modal

from quail.bench.images import gpu_image, sglang_image
from quail.bench.results import combine_measurements, write_json

image = gpu_image()
sglang_gpu_image = sglang_image()

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
def ensure_data(sf: float, query_ids: list[str], collection_id: str):
    """Write the queries' tables to the volume and resolve the label collection."""
    import pyarrow.parquet as pq

    import quail_b as benchmark
    from quail.bench.substrait import read_plan

    names = {
        relation.table for query_id in query_ids
        for relation in read_plan(benchmark.get_query(query_id).plan).relations
    }
    directory = Path(DATA_DIR) / f"sf{sf}"
    directory.mkdir(parents=True, exist_ok=True)
    for name in sorted(names):
        path = directory / f"{name}.parquet"
        if not path.exists():
            pq.write_table(benchmark.load_table(name, scale_factor=sf), path)
    suite = benchmark.load_benchmark(
        query_ids, scale_factor=sf, data_dir=directory,
        collection_id=collection_id or None)
    results_vol.commit()
    return suite.ground_truth.collection_id


def _methods(csv: str) -> list[str]:
    """The method names in a comma-separated list."""
    return [item.strip() for item in csv.split(",") if item.strip()]


def _run_family(process_groups, result_name, model, sf, query_ids_csv,
                run_dir, ground_truth_collection, root=None) -> str:
    """Run one query family's methods, one child process per group.

    Every group runs on the one GPU of this container, in a fresh
    process. The family's result, without the per-method suites, is
    saved under `families/` in the run directory.
    """
    from quail.bench.process_isolation import run_backend_group_in_fresh_process
    from quail_b.queries import query_family_name

    query_ids = tuple(
        query_id.strip() for query_id in query_ids_csv.split(",")
        if query_id.strip())
    family = query_family_name(query_ids)
    process_results = []
    suites = {}
    for methods in process_groups:
        process_result = run_backend_group_in_fresh_process(
            data_dir=DATA_DIR, model=model, sf=sf, query_ids=query_ids,
            run_dir=run_dir, ground_truth_collection=ground_truth_collection,
            methods=methods, root=root)
        process_results.append(process_result)
        suites.update(process_result["suites"])
    gpu_uuids = {
        gpu_uuid for process_result in process_results
        for gpu_uuid in process_result["gpu_uuids"]
    }
    if len(gpu_uuids) != 1:
        raise RuntimeError(
            "backend child processes did not see one physical GPU: "
            f"{sorted(gpu_uuids)}")
    result = {
        "query_family": family,
        "query_ids": list(query_ids),
        "ground_truth_collection": next(iter(suites.values()))["collection_id"],
        "gpu": "H100!",
        "gpu_uuids": sorted(gpu_uuids),
        "process_groups": [list(methods) for methods in process_groups],
        "process_cleanup": [
            {"methods": process_result["methods"],
             **process_result["process_cleanup"]}
            for process_result in process_results
        ],
        "result_path": f"families/{family}{result_name}.json",
        "suites": suites,
    }
    write_json(Path(run_dir) / result["result_path"],
               {key: value for key, value in result.items() if key != "suites"})
    return json.dumps(result)


@app.function(
    image=image,
    gpu="H100!",
    memory=98304,
    timeout=36000,
    max_containers=8,
    volumes=VOLUMES,
)
def run_query_family(
    model: str,
    sf: float,
    query_ids_csv: str,
    run_dir: str,
    ground_truth_collection: str,
    include_baselines: bool,
    include_quail: bool = True,
    include_dumb_vllm: bool = False,
    baselines: str = "stock_vllm,pipelined_vllm",
) -> str:
    """Run one query family through Quail and the vLLM baselines.

    baselines names the vLLM baseline methods that run when
    include_baselines is set.
    """
    process_groups = [("quail",)] if include_quail else []
    if include_baselines:
        process_groups.append(tuple(_methods(baselines)))
    if include_dumb_vllm:
        process_groups.append(("dumb_vllm",))
    try:
        return _run_family(process_groups, "", model, sf, query_ids_csv,
                           run_dir, ground_truth_collection)
    finally:
        results_vol.commit()
        kernel_cache.commit()


@app.function(
    image=sglang_gpu_image,
    gpu="H100!",
    memory=98304,
    timeout=36000,
    max_containers=8,
    volumes=VOLUMES,
)
def run_sglang_query_family(
    model: str,
    sf: float,
    query_ids_csv: str,
    run_dir: str,
    ground_truth_collection: str,
) -> str:
    """Run one query family through the SGLang backend."""
    try:
        return _run_family([("pipelined_sglang",)], "-sglang", model, sf,
                           query_ids_csv, run_dir, ground_truth_collection)
    finally:
        results_vol.commit()
        kernel_cache.commit()


def _merge_suites(parts, query_ids, run_id, started, elapsed,
                  function_call_ids, methods):
    base = parts[0]
    fields = ("scale_factor", "corpus_id", "collection_id", "metadata",
              "gpu_count", "gpu_hourly_rate_usd")
    by_query = {}
    skipped = {}
    for part in parts:
        if part["run_id"] != run_id:
            raise ValueError("query families disagree on run_id")
        for field in fields:
            if part[field] != base[field]:
                raise ValueError(f"query families disagree on {field}")
        family = part["query_family"]["name"]
        for item in part["queries"]:
            if item["id"] in by_query:
                raise ValueError(f"duplicate query {item['id']}")
            by_query[item["id"]] = dict(item, directory=f"{family}/{item['id']}")
        skipped.update(part.get("skipped_queries", {}))
    if set(by_query) | set(skipped) != set(query_ids):
        raise ValueError("completed queries do not match the requested queries")
    merged = dict(base)
    merged.pop("query_family")
    merged.update(
        queries=[by_query[query_id] for query_id in query_ids
                 if query_id in by_query],
        skipped_queries=skipped,
        started_at=started.isoformat(),
        finished_at=datetime.now(timezone.utc).isoformat(),
        parallel={
            "wall_s": round(elapsed, 1),
            "containers": len(parts),
            "methods_per_container": list(methods),
            "function_call_ids": function_call_ids,
        })
    return merged


@app.function(image=image, timeout=43200, memory=4096, volumes=VOLUMES)
def run_all(
    run_dir: str,
    model: str = "qwen3-4b-fp8",
    sf: float = 0.1,
    query: str = "",
    ground_truth_collection: str = "",
    include_baselines: bool = True,
    include_sglang: bool = True,
    include_quail: bool = True,
    include_dumb_vllm: bool = False,
    baselines: str = "stock_vllm,pipelined_vllm",
):
    from quail_b import select_queries
    from quail_b.queries import query_family_name, split_query_families

    only = [item.strip() for item in query.split(",")] if query else None
    query_ids = tuple(spec.id for spec in select_queries(only, scale_factor=sf))
    # not resolve(): /results is a mount inside the container
    directory = Path(run_dir)
    if not directory.is_relative_to("/results") or directory == Path("/results"):
        raise ValueError("run directory must be inside the /results volume mount")
    directory.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    manifest = {
        "run_id": directory.name,
        "status": "running",
        "started_at": started.isoformat(),
        "model": model,
        "sf": sf,
        "query_ids": list(query_ids),
        "summaries": {},
        "function_call_ids": {},
    }
    manifest_path = directory / "manifest.json"
    write_json(manifest_path, manifest)
    results_vol.commit()

    try:
        data_call = ensure_data.spawn(sf, query_ids, ground_truth_collection)
        manifest["function_call_ids"]["data"] = data_call.object_id
        print(f"function call id: {data_call.object_id} (data)", flush=True)
        ground_truth_collection = data_call.get()
        manifest["collection_id"] = ground_truth_collection

        families = split_query_families(query_ids)
        family_calls = []
        sglang_calls = []
        call_ids = manifest["function_call_ids"]
        for family_ids in families:
            family = query_family_name(family_ids)
            if include_quail or include_baselines or include_dumb_vllm:
                family_call = run_query_family.spawn(
                    model=model,
                    sf=sf,
                    query_ids_csv=",".join(family_ids),
                    run_dir=run_dir,
                    ground_truth_collection=ground_truth_collection,
                    include_baselines=include_baselines,
                    include_quail=include_quail,
                    include_dumb_vllm=include_dumb_vllm,
                    baselines=baselines,
                )
                family_calls.append((family, family_call))
                call_ids[f"{family}:quail_vllm"] = family_call.object_id
                print(
                    f"function call id: {family_call.object_id} "
                    f"({family}, Quail and vLLM, {','.join(family_ids)})",
                    flush=True,
                )
            if include_sglang:
                sglang_call = run_sglang_query_family.spawn(
                    model=model,
                    sf=sf,
                    query_ids_csv=",".join(family_ids),
                    run_dir=run_dir,
                    ground_truth_collection=ground_truth_collection,
                )
                sglang_calls.append((family, sglang_call))
                call_ids[f"{family}:sglang"] = sglang_call.object_id
                print(
                    f"function call id: {sglang_call.object_id} "
                    f"({family}, SGLang, {','.join(family_ids)})",
                    flush=True,
                )

        write_json(manifest_path, manifest)
        results_vol.commit()
        return _finish_run(directory, manifest, family_calls, sglang_calls,
                           query_ids, started)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        results_vol.commit()


def _finish_run(directory, manifest, family_calls, sglang_calls, query_ids,
                started):
    """Wait for a run's family calls, merge their suites, write the reports.

    family_calls and sglang_calls pair a family name with its Modal
    function call. The manifest is updated in place and written.
    """
    manifest_path = directory / "manifest.json"
    call_ids = manifest["function_call_ids"]
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
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()

    # the family results name the methods they ran
    methods = list(family_parts[0]["suites"]) if family_parts else []
    if sglang_parts:
        methods.append("pipelined_sglang")
    reports = {}
    paths = {}
    for method in methods:
        parts = sglang_parts if method == "pipelined_sglang" else family_parts
        reports[method] = _merge_suites(
            [part["suites"][method] for part in parts],
            query_ids, directory.name, started, elapsed, call_ids,
            tuple(item for group in parts[0]["process_groups"]
                  for item in group))
        paths[method] = f"{method}/run.json"

    from quail_b import report as write_report

    for method, report in reports.items():
        write_json(directory / paths[method], report)
        write_report(directory / method, rescore=False)
    combine_measurements(directory, list(reports))
    manifest.update(
        status="complete",
        parallel_wall_s=round(elapsed, 1),
        families={
            **{part["query_family"]: part["result_path"]
               for part in family_parts},
            **{f"{part['query_family']}-sglang": part["result_path"]
               for part in sglang_parts},
        },
        summaries=paths,
    )
    completed = reports["quail"]["queries"] if "quail" in reports else []
    final = {
        "run_id": directory.name,
        "run_dir": str(directory),
        "manifest": str(manifest_path),
        "queries_completed": sum(
            item["status"] == "complete" for item in completed),
        "queries_failed": sum(item["status"] != "complete" for item in completed),
        "parallel_wall_s": round(elapsed, 1),
        "function_call_ids": call_ids,
    }
    print(json.dumps(final, indent=2), flush=True)
    return json.dumps(final)


@app.function(image=image, timeout=43200, memory=4096, volumes=VOLUMES)
def finish_run(run_dir: str) -> str:
    """Finish a run whose orchestrator died while its families ran.

    Reads the run's manifest, waits for the family calls it names, and
    merges their results as run_all would have.
    """
    import modal

    directory = Path(run_dir)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    family_calls, sglang_calls = [], []
    for name, call_id in manifest["function_call_ids"].items():
        family, _, kind = name.partition(":")
        if kind == "quail_vllm":
            family_calls.append((family, modal.FunctionCall.from_id(call_id)))
        elif kind == "sglang":
            sglang_calls.append((family, modal.FunctionCall.from_id(call_id)))
    started = datetime.fromisoformat(manifest["started_at"])
    manifest["status"] = "running"
    try:
        return _finish_run(directory, manifest, family_calls, sglang_calls,
                           tuple(manifest["query_ids"]), started)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        results_vol.commit()


@app.local_entrypoint()
def main(
    output_dir: str = "/results/benchmarks/quailb",
    model: str = "qwen3-4b-fp8",
    sf: float = 0.1,
    query: str = "",
    ground_truth_collection: str = "",
    include_baselines: bool = True,
    include_sglang: bool = True,
    include_quail: bool = True,
    include_dumb_vllm: bool = False,
    baselines: str = "stock_vllm,pipelined_vllm",
    finish: str = "",
):
    if finish:
        # finish an earlier run whose orchestrator died
        call = finish_run.spawn(finish)
        print(f"function call id: {call.object_id} (finish {finish})",
              flush=True)
        print(call.get(), flush=True)
        return
    run_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid.uuid4().hex[:8]}")
    run_dir = str(Path(output_dir) / run_id)
    print(f"run directory: {run_dir}", flush=True)
    call = run_all.spawn(
        run_dir=run_dir,
        model=model,
        sf=sf,
        query=query,
        ground_truth_collection=ground_truth_collection,
        include_baselines=include_baselines,
        include_sglang=include_sglang,
        include_quail=include_quail,
        include_dumb_vllm=include_dumb_vllm,
        baselines=baselines,
    )
    print(f"function call id: {call.object_id} (all families)", flush=True)
    print(call.get(), flush=True)
