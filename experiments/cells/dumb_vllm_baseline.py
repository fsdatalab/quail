"""Measure one QUAIL-B query with vLLM at its default settings.

The `dumb_vllm` backend passes vLLM only the model name: default
`max_num_seqs`, `max_num_batched_tokens`, `gpu_memory_utilization`,
and CUDA graph capture, on one H100. Filters run operator-at-a-time
and join pairs are submitted anchor-major, as for stock vLLM.

    run_log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-dumb-vllm.log"
    uv run modal run --detach experiments/cells/dumb_vllm_baseline.py \
      --query BIO-3 --sf 0.1 2>&1 | tee "$run_log"

Results, answers, GPU identity, and process cleanup are saved on the
quail-results volume under the printed run directory.
"""

import json
from datetime import datetime, timezone

from quail.bench.quailb_parallel import (
    VOLUMES,
    _run_family,
    app,
    ensure_data,
    image,
    kernel_cache,
    results_vol,
)

METHOD = "dumb_vllm"


@app.function(
    image=image,
    gpu="H100!",
    memory=98304,
    timeout=21600,
    volumes=VOLUMES,
)
def run_dumb_vllm_query_family(
    model: str,
    sf: float,
    query_ids_csv: str,
    run_dir: str,
    ground_truth_collection: str,
) -> str:
    """Run one query family through vLLM at its default settings."""
    try:
        return _run_family([(METHOD,)], "-dumb-vllm", model, sf,
                           query_ids_csv, run_dir, ground_truth_collection)
    finally:
        results_vol.commit()
        kernel_cache.commit()


@app.local_entrypoint()
def main(query: str = "BIO-3", sf: float = 0.1, model: str = "qwen3-4b-fp8"):
    query_ids = [item.strip() for item in query.split(",") if item.strip()]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = f"/results/benchmarks/quailb/family-runs/{stamp}-dumb-vllm"
    prediction_text = (
        "BIO-3 at sf=0.1 is a filter over 500 reports and a join of the "
        "survivors against 1,127 terms, about 309,000 pairs. Stock vLLM "
        "took 26.3 s for the same filter on BIO-1. With default settings "
        "(1,024 sequences and 16,384 batched tokens per step instead of "
        "4,096 and 25,305) the filter should take about the same time and "
        "the join 70 to 130 s, so 100 to 160 s in total, compared with "
        "79.4 s for Quail. Fresh tokens near 8 million: 2.1 million for "
        "the filter and 6 million for the join pairs. Answers should match "
        "stock vLLM exactly, since both run greedy decoding on the same "
        "prompts."
    )
    print(f"run directory: {run_dir}", flush=True)
    print(f"prediction: {prediction_text}", flush=True)
    data_call = ensure_data.spawn(sf, query_ids, "")
    print(f"function call id: {data_call.object_id} (data)", flush=True)
    collection = data_call.get()
    print(f"ground truth collection: {collection}", flush=True)
    call = run_dumb_vllm_query_family.spawn(
        model=model, sf=sf, query_ids_csv=",".join(query_ids),
        run_dir=run_dir, ground_truth_collection=collection)
    print(f"function call id: {call.object_id} ({METHOD}, {query})", flush=True)
    result = json.loads(call.get())
    print(f"process cleanup: {result['process_cleanup']}", flush=True)
    print(f"family result volume path: {run_dir}/{result['result_path']}",
          flush=True)
    print(f"suite volume path: {run_dir}/{METHOD}/", flush=True)
    for item in result["suites"][METHOD]["queries"]:
        measurements = item.get("measurements", {})
        print(json.dumps({
            "id": item["id"], "status": item["status"],
            "runtime_s": item.get("runtime_s"),
            "fresh_tokens": measurements.get("fresh_tokens"),
            "boot_s": measurements.get("boot_s"),
            "capacity": measurements.get("backend_metrics", {}).get("capacity"),
        }), flush=True)
