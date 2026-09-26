"""Measure a join both ways: two-call (tree) attention against unified.

The tree_attention planner rule picks the path by a roofline. On
QUAIL-B it picks "unified" only for the LePaRD joins: a 208-token
anchor read by partners of about 63 rows each. This runs one such
query and one the rule leaves on the two-call path, each forced both
ways in one session, and records runtime and fresh tokens.

    mkdir -p results/benchmark
    log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-join-attention-paths.log"
    uv run modal run --detach experiments/join_attention_paths.py 2>&1 | tee "$log"

The summary goes to /results/ablations/join_attention_paths_qwen3-4b-fp8.json.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import modal

from quail.bench.images import gpu_image

QUERIES = ("LEP-4", "FEV-4")
ROUNDS = ("tree", "unified", "tree", "unified")
DATA_DIR = "/results/quailb_data/sf0.1"
SUMMARY_PATH = Path("/results/ablations/join_attention_paths_qwen3-4b-fp8.json")

PREDICTION_TEXT = (
    "On LEP-4 (208-token anchors, partners of about 63 rows, 5.6 per "
    "anchor) the roofline says unified is cheaper: unified will run "
    "within 5 percent of tree, either way. On FEV-4 (439-token anchors, "
    "16-row partners, 41 per anchor) the roofline says tree: tree will "
    "be at least 5 percent faster. Fresh tokens and answers match "
    "between paths on both queries."
)

app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
kernel_cache = modal.Volume.from_name(
    "quail-kernel-cache", create_if_missing=True)

image = gpu_image()


class ForceJoinAttention:
    """Overwrite every join's attention with the path under test."""

    name = "force_join_attention"

    def __init__(self):
        self.path = "tree"

    def rewrite(self, graph, context):
        from quail.physical import AiJoin, PhysicalGraph

        nodes = tuple(
            replace(node, attention=self.path) if isinstance(node, AiJoin)
            else node for node in graph.nodes)
        return PhysicalGraph(nodes, graph.root)


def _answer_digest(output) -> str:
    digest = hashlib.sha256()
    for name in sorted(output.join_answers):
        table = output.join_answers[name]
        digest.update(name.encode())
        digest.update(table.sort_by([
            (column, "ascending") for column in table.column_names
        ]).to_pandas().to_csv(index=False).encode())
    return digest.hexdigest()


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
def measure(query_ids: list[str]) -> dict:
    import pyarrow.parquet as pq

    import quail
    from quail.bench.quailb import run_query
    from quail.builtins import built_in_registry
    from quail_b.benchmark import select_queries
    from quail_b.data import CORPUS_COLUMNS

    force = ForceJoinAttention()
    registry = built_in_registry().register_physical_rule(force)
    config = quail.EngineConfig(model="qwen3-4b-fp8", device="h100-sxm")
    runs = []
    with quail.Session(config, registry=registry) as session:
        for spec in select_queries(list(query_ids), scale_factor=0.1):
            tables = {
                relation.table: pq.read_table(
                    Path(DATA_DIR) / f"{relation.table}.parquet",
                    columns=list(CORPUS_COLUMNS[relation.table]))
                for relation in spec._info.relations
            }
            for path in ROUNDS:
                force.path = path
                started = time.perf_counter()
                output = run_query(session, spec, tables)
                runs.append({
                    "query": spec.id,
                    "attention": path,
                    "runtime_s": round(output.runtime_s, 3),
                    "wall_s": round(time.perf_counter() - started, 3),
                    "fresh_tokens": output.measurements.get("fresh_tokens"),
                    "answers_sha256": _answer_digest(output),
                })
                print(json.dumps(runs[-1]), flush=True)
    summary = {
        "model": "qwen3-4b-fp8",
        "device": "h100-sxm",
        "scale_factor": 0.1,
        "prediction": PREDICTION_TEXT,
        "rounds": list(ROUNDS),
        "runs": runs,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    return summary


@app.local_entrypoint()
def main(queries: str = ",".join(QUERIES)):
    print(f"prediction: {PREDICTION_TEXT}", flush=True)
    call = measure.spawn(queries.split(","))
    print(f"function call id: {call.object_id}", flush=True)
    summary = call.get()
    for run in summary["runs"]:
        print(json.dumps(run), flush=True)
    print(f"summary: {SUMMARY_PATH}", flush=True)
