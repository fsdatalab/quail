r"""Measure FEV-10 with its pairing done by apply() instead of the equality.

Three ways to ask SUPPORT of a claim and its own Wikipedia page, in one
container on one H100, one unmeasured warmup run each:

- equality: FEV-10 as in QUAIL-B, ``ON c.evidence_wiki_url = e.id``.
- per_batch: the same pairing done by a Python function passed to
  apply(); the join calls it on each batch of survivors the anchor's
  chain hands over, with the KV still pinned.
- barrier: the same function passed to apply_table(); the anchor's
  chain finishes first and the function runs once over every survivor.

    uv run modal run --detach experiments/cells/foreign_pairs.py \
      --prediction "State the prediction before running." \
      2>&1 | tee /tmp/quail-foreign-pairs.log

Results and answer tables are saved on the quail-results volume.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import modal

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
)
app = modal.App("quail-milestone1")
results = modal.Volume.from_name("quail-results")
volumes = {
    "/results": results,
    "/root/.cache/huggingface": modal.Volume.from_name("quail-hf-cache"),
    "/root/.cache/kernels": modal.Volume.from_name("quail-kernel-cache"),
}
VARIANTS = ("equality", "per_batch", "barrier")


def same_page(tables):
    """Pair each claim with the evidence row whose id is its page."""
    claims, evidence = tables["c"], tables["e"]
    return claims.join(evidence, keys=["evidence_wiki_url"],
                       right_keys=["id"]).select(["c", "e"])


def build(session, variant):
    """FEV-10 with its pairing done one of three ways."""
    from quail import col, prompt
    from quail_b import prompts
    from quail_b.queries import (
        FILTER_SELECTIVITY_ESTIMATES,
        JOIN_SELECTIVITY_ESTIMATES,
    )

    claims = session.docs("claims").alias("c").ai_filter(
        prompt(prompts.F11, col("c.claim")),
        selectivity=FILTER_SELECTIVITY_ESTIMATES[prompts.F11])
    evidence = session.docs("evidence").alias("e").ai_filter(
        prompt(prompts.F13, col("e.text")),
        selectivity=FILTER_SELECTIVITY_ESTIMATES[prompts.F13])
    columns = [col("c.evidence_wiki_url"), col("e.id")]
    if variant == "equality":
        query = claims.join(evidence, on=col("c.evidence_wiki_url") == col("e.id"))
    elif variant == "per_batch":
        query = claims.join(evidence).apply(same_page, columns)
    else:
        query = claims.join(evidence).apply_table(same_page, columns)
    return query.ai_filter(
        prompt(prompts.SUPPORT, col("c.claim"), col("e.text")),
        selectivity=JOIN_SELECTIVITY_ESTIMATES[prompts.SUPPORT],
    ).select("c.id", "e.id", order="by_cost")


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=volumes)
def compare(prediction: str) -> str:
    import subprocess
    from dataclasses import asdict

    import pyarrow.parquet as pq

    import quail
    from quail.bench.quailb import register_sets
    from quail.planner.plan import EngineConfig
    from quail.specs import H100_USD_PER_HOUR
    from quail_b.data import build_sets

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(f"/results/ablations/foreign-pairs-{stamp}")
    output.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader",
    ], text=True).strip()
    (output / "setup.json").write_text(json.dumps({
        "prediction": prediction, "variants": list(VARIANTS),
        "gpu_uuid": gpu, "order": list(VARIANTS),
        "warmup": "one unmeasured run per variant, FEV-1 first",
    }, indent=2))
    with quail.Session(EngineConfig(model="qwen3-4b-fp8", gpus=1)) as session:
        register_sets(session, build_sets("/results/quailb_data", 0.1, 1))
        from quail.bench.quailb import queries
        # the first query after the cold boot is not one being compared
        queries(session)["FEV-1"][1]().run().count()
        for variant in VARIANTS:
            build(session, variant).run().count()
            result = build(session, variant).run()
            row_count = result.count()
            vdir = output / variant
            vdir.mkdir(exist_ok=True)
            for kind, tables in result.answer_tables.items():
                for key, answers in tables.items():
                    name = ("-".join(map(str, key)) if isinstance(key, tuple)
                            else str(key))
                    pq.write_table(answers, vdir / f"{kind}-{name}.parquet")
            pq.write_table(result.collect(), vdir / "rows.parquet")
            nodes = {}
            for node in result.plan.topological_nodes():
                metrics = result.node_metrics.get(node.node_id)
                nodes[node.node_id] = dict(
                    type=node.type_name, fields=node.explain_fields(),
                    metrics=(None if metrics is None else {
                        k: v for k, v in asdict(metrics).items()
                        if k != "extension"}))
            pairs = sum(stage["tuples"] for stage in result.report["stages"]
                        if stage["op"] == "join")
            seconds = result.report["wall_s"]
            record = {
                "variant": variant, "sf": 0.1, "model": "qwen3-4b-fp8",
                "gpus": 1, "report": result.report,
                "explain": result.explain(), "nodes": nodes,
                "rows": row_count, "query_seconds": seconds,
                "evaluated_document_pairs": pairs,
                "usd_per_query": seconds / 3600 * H100_USD_PER_HOUR,
            }
            (vdir / "summary.json").write_text(json.dumps(record, indent=2))
            print(f"[{variant}] {seconds} seconds, {pairs} pairs, "
                  f"{row_count} rows, {result.report['fresh_tokens']} "
                  f"fresh tokens, {result.report['regret_tokens']} "
                  f"recomputed", flush=True)
            results.commit()
    return str(output)


@app.local_entrypoint()
def main(prediction: str):
    if not prediction:
        raise ValueError("state the prediction before running")
    call = compare.spawn(prediction)
    print(f"function call id: {call.object_id}", flush=True)
    print(f"result volume path: {call.get()}", flush=True)
