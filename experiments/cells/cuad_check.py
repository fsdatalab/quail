"""The CUAD queries through QUAIL-B, on Quail and stock vLLM.

Runs CUAD-1 to CUAD-15 at one scale factor on one model, one backend
per container, against the published labels. Each run saves a QUAIL-B
run record and report under
/results/benchmarks/quailb/cuad-<run>-<model>-<backend>/. The tables
are downloaded once into the Hugging Face cache volume so that a
backend's refusals are found before its engine boots: a decision model
refuses the nine extraction queries, and stock vLLM refuses CUAD-11's
GROUP BY.

Prediction for Qwen3 4B, from the 2,500-character contract windows of
run 20261011T060458Z (Quail span F1 47.5 with locating on, stock vLLM
52.0): on the extraction queries Quail's span token F1 lands within 10
points of stock vLLM's, lower on the clause fields (CUAD-3, 7, 8, 9)
than on the short ones (CUAD-1, 2), and both engines agree with the
lawyers' Yes or No answers on the filters within 3 points of each
other. Measured on run 20261011T183123Z: Quail trailed stock vLLM by
up to 32 points on the clause fields (CUAD-3: 0.23 against 0.55) and
matched it within 3 points on the filters.

Prediction for Decision-2.0-Kai 0.6B: the planner refuses the nine
extraction queries, so six run. Its agreement with the lawyers on the
two filter queries lands within 10 points of Qwen3 4B's (CUAD-5 0.71,
CUAD-6 0.27 on run 20261011T183123Z), the governing law
classifications (CUAD-11, 12) within 10 points of Qwen3 4B's 0.91, and
the join queries (CUAD-13, 14) keep Qwen3 4B's shape, recall above 0.9
and precision below 0.3. Every query finishes in under 10 s on Quail.

    uv run modal run --detach -m experiments.cells.cuad_check
        [--model decision-2.0-kai-0.6b-bf16] [--backends quail]
        [--sf 0.1] [--only CUAD-5,CUAD-13]
        2>&1 | tee /tmp/cuad_check.log
"""

import json
import time

import modal

try:
    from quail.bench.images import gpu_image
    image = gpu_image(("experiments", "/root/experiments"))
except ImportError:    # a container without the local quail package
    image = None

app = modal.App("quail-milestone1")
VOLUMES = {
    "/root/.cache/huggingface": modal.Volume.from_name(
        "quail-hf-cache", create_if_missing=True),
    "/root/.cache/kernels": modal.Volume.from_name(
        "quail-kernel-cache", create_if_missing=True),
    "/results": modal.Volume.from_name("quail-results", create_if_missing=True),
}
TABLES = "/root/.cache/huggingface/quail-b-tables"
QUERIES = tuple(f"CUAD-{i}" for i in range(1, 16))


def _summary(record) -> dict:
    """Pull the numbers to report from each query of a run record."""
    out = {}
    for item in record["queries"]:
        metrics = item.get("metrics") or {}
        accuracy = metrics.get("accuracy") or {}
        out[item["id"]] = {
            "status": item["status"],
            "runtime_s": item.get("runtime_s"),
            "answer_accuracy": (accuracy.get("answer_accuracy") or {}).get(
                "accuracy"),
            "label_accuracy": (accuracy.get("label_accuracy") or {}).get(
                "accuracy"),
            "span_token_f1": (accuracy.get("span_accuracy") or {}).get(
                "token_f1"),
            "span_exact_match": (accuracy.get("span_accuracy") or {}).get(
                "exact_match"),
            "output_precision": (accuracy.get("output_accuracy") or {}).get(
                "precision"),
            "output_recall": (accuracy.get("output_accuracy") or {}).get(
                "recall"),
            "input_tokens": metrics.get("input_tokens"),
            "fresh_tokens": metrics.get("fresh_tokens"),
            "regret_tokens": metrics.get("regret_tokens"),
            "input_tokens_per_second": metrics.get("input_tokens_per_second"),
            "cost_usd": metrics.get("cost_usd"),
            "error": item.get("error"),
        }
    return out


@app.function(image=image, gpu="H100!", memory=98304, timeout=7200,
              volumes=VOLUMES)
def run_cuad(run: str, model: str, backend: str, sf: float = 0.1,
             only=None, locate: bool = True) -> dict:
    """Run the CUAD queries on one backend and save the QUAIL-B record."""
    import quail
    from quail.bench import quailb
    from quail_b.data import build_sets

    quailb.EXTRACT_OPTIONS.update(locate=locate)
    queries = list(only or QUERIES)
    data_dir = build_sets(TABLES, sf)
    VOLUMES["/root/.cache/huggingface"].commit()
    output = (f"/results/benchmarks/quailb/cuad-{run}-{model}-{backend}"
              f"{'' if locate else '-plain'}")
    started = time.perf_counter()
    try:
        record = quailb.run_suite(
            queries, sf=sf,
            config=quail.EngineConfig(model=model, device="h100-sxm",
                                      backend=backend),
            data_dir=data_dir, output_dir=output)
    finally:
        VOLUMES["/results"].commit()
    summary = {"run": run, "model": model, "backend": backend, "sf": sf,
               "output": output, "locate": locate,
               "total_s": time.perf_counter() - started,
               "skipped": record.get("skipped_queries", {}),
               "queries": _summary(record)}
    print(json.dumps(summary), flush=True)
    return summary


@app.local_entrypoint()
def main(model: str = "qwen3-4b-fp8", backends: str = "quail,stock_vllm",
         sf: float = 0.1, only: str = "", locate: bool = True):
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    queries = [q for q in only.split(",") if q] or None
    for backend in backends.split(","):
        call = run_cuad.spawn(run, model, backend, sf, queries, locate)
        print(f"{backend}: function call id {call.object_id}", flush=True)
    print(f"run {run}; fetch with modal.FunctionCall.from_id(id).get()",
          flush=True)
