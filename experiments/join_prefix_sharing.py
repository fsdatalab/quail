"""Measure a join whose anchors share prefixes, with and without sharing.

No QUAIL-B join has anchors with shared prefixes, so this joins the
scale 0.1 AGENT snapshots (1,772 anchors) with a table of four issue
categories. Each round runs the planner's choice (sharing on) or with
sharing forced off; the first round is warm-up.

    mkdir -p results/benchmark
    log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-join-prefix-sharing.log"
    uv run modal run --detach experiments/join_prefix_sharing.py 2>&1 | tee "$log"

The summary goes to /results/ablations/join_prefix_sharing/<stamp>/summary.json
and /results/ablations/join_prefix_sharing_<model>.json.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import modal

from quail.bench.images import gpu_image
from quail.specs import H100_USD_PER_HOUR

ROUNDS = ("on", "on", "off", "on", "off")
DATA_DIR = Path("/results/quailb_data/sf0.1")
RUNS_DIR = Path("/results/ablations/join_prefix_sharing")
CATEGORIES = ("a crash or an exception", "wrong output from a function",
              "a missing feature", "a performance problem")

PROMPT_TEXT = (
    "{0}\n\nInstruction: answer TRUE if the issue the agent works on in "
    "the trace above is {1}. Answer FALSE otherwise."
)

PREDICTION_TEXT = (
    "Anchors dominate the work: 1,772 snapshots of about 9,800 tokens "
    "against four partners of a few tokens each. With sharing, fresh "
    "tokens drop from about 17.4 M plus partner rows to about 5.5 M "
    "plus partner rows, and query time drops by 2 to 3 times. Answers "
    "match between the two configurations on at least 99 percent of "
    "pairs."
)

app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
kernel_cache = modal.Volume.from_name(
    "quail-kernel-cache", create_if_missing=True)

image = gpu_image()


class ForceNoSharing:
    """Turn prefix sharing off on every join when enabled."""

    name = "force_no_join_sharing"

    def __init__(self):
        self.enabled = False

    def rewrite(self, graph, context):
        from quail.physical import AiJoin, PhysicalGraph

        if not self.enabled:
            return None
        nodes = tuple(
            replace(node, share_prefixes=False) if isinstance(node, AiJoin)
            else node for node in graph.nodes)
        return PhysicalGraph(nodes, graph.root)


@app.function(
    image=image,
    gpu="H100!",
    memory=98304,
    timeout=7200,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/kernels": kernel_cache,
        "/results": results_vol,
    },
)
def measure(stamp: str, model: str) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    import quail
    from quail.builtins import built_in_registry
    from quail.physical import AiJoin

    results_vol.reload()
    traces = pq.read_table(DATA_DIR / "agent_traces.parquet",
                           columns=["id", "trace"])
    categories = pa.table({"id": [f"c{i}" for i in range(len(CATEGORIES))],
                           "category": list(CATEGORIES)})
    force = ForceNoSharing()
    registry = built_in_registry().register_physical_rule(force)
    config = quail.EngineConfig(model=model, device="h100-sxm")
    runs = []
    first = None
    with quail.Session(config, registry=registry) as session:
        session.register("traces", quail.DocumentProvider.from_table(
            traces, id_col="id"))
        session.register("categories", quail.DocumentProvider.from_table(
            categories, id_col="id"))
        for round_index, setting in enumerate(ROUNDS):
            force.enabled = setting == "off"
            query = (session.docs("traces").alias("t")
                     .ai_join(session.docs("categories").alias("c"),
                              quail.prompt(PROMPT_TEXT, quail.col("t.trace"),
                                           quail.col("c.category")),
                              selectivity=0.3, anchor="t")
                     .select("t.id", "c.id"))
            result = query.run()
            table = result.execute_stream().read_all()
            pairs = set(zip(table.column(0).to_pylist(),
                            table.column(1).to_pylist()))
            if first is None:
                first = pairs
            join = next(node for node in result.plan.nodes
                        if isinstance(node, AiJoin))
            report = result.report
            runtime_s = report["model_wall_s"] + report["finish_s"]
            evaluated = len(CATEGORIES) * traces.num_rows
            runs.append({
                "round": round_index, "sharing": setting,
                "share_prefixes": join.share_prefixes,
                "runtime_s": runtime_s,
                "model_wall_s": report["model_wall_s"],
                "fresh_tokens": report["fresh_tokens"],
                "borrowed_prefix_tokens": next(
                    (metrics.get("borrowed_prefix_tokens", 0)
                     for metrics in report["node_metrics"].values()
                     if "borrowed_prefix_tokens" in metrics), 0),
                "matched_pairs": len(pairs),
                "pair_agreement": 1 - len(pairs ^ first) / evaluated,
                "cost_usd": runtime_s / 3600 * H100_USD_PER_HOUR,
            })
            print(json.dumps(runs[-1]), flush=True)
    summary = {"model": model, "device": config.device,
               "anchors": traces.num_rows, "partners": len(CATEGORIES),
               "prediction": PREDICTION_TEXT, "rounds": list(ROUNDS),
               "runs": runs}
    out = RUNS_DIR / stamp
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (RUNS_DIR.parent / f"join_prefix_sharing_{model}.json").write_text(
        json.dumps(summary, indent=2))
    results_vol.commit()
    return summary


@app.local_entrypoint()
def main(model: str = "qwen3-4b-fp8"):
    from datetime import datetime, timezone

    print(f"prediction: {PREDICTION_TEXT}", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    call = measure.spawn(stamp, model)
    print(f"function call id: {call.object_id}", flush=True)
    summary = call.get()
    for run in summary["runs"]:
        print(json.dumps(run), flush=True)
    print(f"summary: {RUNS_DIR / stamp / 'summary.json'}", flush=True)
