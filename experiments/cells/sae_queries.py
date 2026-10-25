"""Run the draft SAE feature queries and report their cost.

The bench runner scores every query against reference labels, which
the draft feature queries do not have, so this cell runs them through
the same `bench.quailb.run_query` path with accuracy off: one H100,
one session, the published scale 0.1 inputs. For each query it
reports query time, input tokens per second, regret tokens, and
$/query, the same token accounting the benchmark report uses.

    uv run modal run --detach -m experiments.cells.sae_queries \
        2>&1 | tee /tmp/sae_queries.log

Records: /results/ablations/sae-queries-sf01-<timestamp>.json.
"""

import json
from datetime import datetime, timezone

import modal

try:
    from quail.deploy.images import gpu_image
    image = gpu_image().add_local_python_source("experiments")
except ImportError:    # a container without the local quail package
    image = None

app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)
results = modal.Volume.from_name("quail-results", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": hf_cache,
           "/root/.cache/kernels": kernel_cache,
           "/results": results}

MODEL = "qwen3.5-2b-sae-bf16"
QUERIES = ("IMDB-18", "IMDB-19", "BIO-7", "SUPPORT-7", "WRENCH-9")
SCALE_FACTOR = 0.1


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def run(output_path: str) -> dict:
    """Each draft feature query's time, tokens, regret, and cost."""
    import quail
    from bench.quailb import run_query
    from quail.specs import H100_USD_PER_HOUR
    from quail_b.benchmark import load_benchmark
    from quail_b.minimum import token_metrics

    suite = load_benchmark(list(QUERIES), scale_factor=SCALE_FACTOR,
                           accuracy=False)
    config = quail.EngineConfig(gpus=1, model=MODEL, backend="quail",
                                device="h100-sxm")
    record = {"model": MODEL, "scale_factor": SCALE_FACTOR,
              "corpus_id": suite.corpus_id, "queries": {}}
    tokens_store = {}
    with quail.Session(config) as session:
        for spec in suite.queries:
            output = run_query(session, spec, suite.tables)
            metrics = token_metrics(spec, output, suite.tables, tokens_store)
            seconds = output.runtime_s
            metrics.update(
                runtime_s=seconds,
                input_rows={relation.alias: len(suite.tables[relation.table])
                            for relation in spec.info.relations},
                input_tokens_per_second=(
                    metrics["input_tokens"] / seconds
                    if metrics["input_tokens"] and seconds else None),
                cost_usd=seconds / 3600 * H100_USD_PER_HOUR,
            )
            record["queries"][spec.id] = metrics
            print(spec.id, json.dumps(metrics), flush=True)
    results.reload()
    with open(output_path, "w") as out:
        json.dump(record, out, indent=2)
    results.commit()
    return record


@app.local_entrypoint()
def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = f"/results/ablations/sae-queries-sf01-{stamp}.json"
    call = run.spawn(output_path)
    print(f"function call id: {call.object_id}", flush=True)
    print(f"results: {output_path}", flush=True)
    print(json.dumps(call.get(), indent=2))
