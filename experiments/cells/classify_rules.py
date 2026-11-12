"""Classify BIO-5 and AGENT-5 under each label scoring rule on Qwen3 4B.

Each query runs three times in one container: with the planner's
choice, with trie_tree forced, and with trie_decode forced. Every run
goes through quail-b, so its record has the query time, input tokens
per second, cost, and label agreement with the reference labels, and
its directory has the labels per document.

trie_tree reads a row only where the labels split. With Qwen3's
tokenizer, the 26 BIO-5 organ classes all split at the answer cue, so
trie_tree reads 1 row per term, where it read 22 when it scored each
label through the end of its deciding word. The AGENT-5 domain labels
read 7 rows, compared with 18, and the root causes 1, compared with 5.

Prediction, against run 20261009T031041Z of this cell (BIO-5 trie_decode
5.30 s with the planner's choice and 71.98% label agreement; trie_tree
7.08 s and 72.12%): on BIO-5, trie_tree returns the same label as
trie_decode on every term, since both pick the most likely first token,
and finishes within 10% of trie_decode's time. On AGENT-5, the three
rules' label agreement differs by at most 1 point, and their query
times by at most 3%.

    run_log="results/classify-rules-$(date -u +%Y%m%dT%H%M%SZ).log"
    uv run modal run --detach -m experiments.cells.classify_rules::compare \
        2>&1 | tee "$run_log"

Run records are saved under /results/ablations/classify-rules-<run>/
on the quail-results volume, with a summary beside them.
"""

import json
import time

from bench.quailb_parallel import DATA_DIR, VOLUMES, app, ensure_data, results_vol
from quail.deploy.images import gpu_image

image = gpu_image()
MODEL = "qwen3-4b-fp8"
QUERIES = (("BIO-5", 0.5), ("AGENT-5", 0.1))
RULES = (None, "trie_tree", "trie_decode")


def _force(rule, original):
    """Make the planner's classification scoring pick one rule."""
    from quail.labels import LETTERS_SCORING
    from quail.planner.operators.classify import ClassifyScoring

    if rule is None:
        ClassifyScoring.choose = original
        return

    def choose(self, expected_rows, head_tokens, frame_tokens, labels, resident,
               lettered=None, probabilities=False):
        head, frame, ids = (lettered if rule == LETTERS_SCORING
                            else (head_tokens, frame_tokens, labels))
        return rule, self.estimate(rule, expected_rows, head, frame, ids, resident)

    ClassifyScoring.choose = choose


def _summary(record) -> dict:
    """Pull the numbers to report from one query's run record."""
    (query,) = record["queries"]
    metrics = query.get("metrics", {})
    plan = query.get("measurements", {}).get("executed_plan") or {}
    nodes = plan.get("nodes", []) if isinstance(plan, dict) else []
    return {
        "status": query.get("status"),
        "runtime_s": query.get("runtime_s"),
        "input_tokens": metrics.get("input_tokens"),
        "input_tokens_per_second": metrics.get("input_tokens_per_second"),
        "cost_usd": metrics.get("cost_usd"),
        "accuracy": metrics.get("accuracy"),
        "scoring": {node["attributes"]["spec"]["name"]:
                    node["attributes"]["spec"]["scoring"]
                    for node in nodes
                    if node.get("type") == "quail.ai_classify"},
        "collection_id": record.get("collection_id"),
    }


@app.function(image=image, gpu="H100!", memory=98304, timeout=5400,
              volumes=VOLUMES)
def run_rules(run: str, queries=QUERIES, rules=RULES) -> dict:
    """Run each query under each rule and save the records and a summary."""
    from pathlib import Path

    import quail
    from bench.quailb import data_directory, run_suite
    from quail.planner.operators.classify import ClassifyScoring

    original = ClassifyScoring.choose
    root = Path(f"/results/ablations/classify-rules-{run}")
    summary = {"run": run, "model": MODEL, "queries": {}}
    started = time.perf_counter()
    try:
        for query_id, scale_factor in queries:
            collection = ensure_data.local(scale_factor, [query_id], "")
            summary["queries"][query_id] = {"sf": scale_factor, "rules": {}}
            for rule in rules:
                name = rule or "planner"
                _force(rule, original)
                output = root / query_id / name
                record = run_suite(
                    [query_id], scale_factor=scale_factor,
                    config=quail.EngineConfig(model=MODEL, device="h100-sxm"),
                    data_dir=data_directory(DATA_DIR, scale_factor, [query_id]),
                    ground_truth_collection=collection, output_dir=str(output))
                summary["queries"][query_id]["rules"][name] = {
                    "directory": str(output), **_summary(record)}
                print(json.dumps({query_id: {name: summary["queries"][query_id]
                                             ["rules"][name]}}), flush=True)
                results_vol.commit()
    finally:
        ClassifyScoring.choose = original
    summary["total_s"] = time.perf_counter() - started
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    return summary


@app.local_entrypoint()
def compare(queries: str = "", rules: str = ""):
    """Run the cell; queries "BIO-5:0.5,AGENT-5:0.1", rules "planner,trie_tree"."""
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    picked = [(q.split(":")[0], float(q.split(":")[1]))
              for q in queries.split(",") if q] or list(QUERIES)
    forced = ([None if r == "planner" else r for r in rules.split(",") if r]
              or list(RULES))
    call = run_rules.spawn(run, picked, forced)
    print(f"run {run} function call id: {call.object_id}", flush=True)
    print(f"summary: /results/ablations/classify-rules-{run}/summary.json",
          flush=True)
    print(json.dumps(call.get(), indent=2), flush=True)
