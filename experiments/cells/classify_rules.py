"""Classify BIO-5 and AGENT-5 under each label scoring rule on Qwen3 4B.

Each query runs three times in one container: with the planner's
choice, with trie_tree forced, and with trie_decode forced. Every run
goes through quail-b, so its record has the query time, input tokens
per second, cost, and label agreement with the reference labels, and
its directory has the labels per document.

Prediction, against the saved BIO-5 run
/results/benchmarks/quailb/20261002T005838Z-45fc1a7e (trie_decode,
6.19 s, 102,421 input tokens per second, 71.98% label agreement):
trie_decode agrees with that run's labels on at least 99% of the 2,934
terms and finishes in 5.0 to 6.0 s, as its rounds drop from up to 11
to 1; trie_tree feeds and reads 22 trie rows per term instead of 96,
agrees with trie_decode on at least 90% of terms, and lands within 2
points of its label agreement. On AGENT-5 at sf 0.1 the three
classifications' label agreement differs by at most 2 points between
rules; trie_tree reads 3, 18, and 5 rows per trace for progress,
domain, and root cause instead of 15, 40, and 8, and trie_decode runs
at most 3 rounds instead of 8.

    uv run modal run --detach -m experiments.cells.classify_rules \
        2>&1 | tee /tmp/classify_rules.log

Run records are saved under /results/ablations/classify-rules-<run>/
on the quail-results volume, with a summary beside them.
"""

import json
import time

import modal

try:
    from quail.bench.images import gpu_image
    image = gpu_image()
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
DATA_DIR = "/results/quailb_data"
MODEL = "qwen3-4b-fp8"
QUERIES = (("BIO-5", 0.5), ("AGENT-5", 0.1))
RULES = (None, "trie_tree", "trie_decode")


def _force(rule, original):
    """Make the planner's label_scoring rule pick one scoring rule."""
    from quail.labels import LETTERS_SCORING
    from quail.planner.label_scoring import ClassifyScoring

    if rule is None:
        ClassifyScoring.choose = original
        return

    def choose(self, live, head_tokens, frame_tokens, labels, resident,
               lettered=None, probabilities=False, starts=()):
        head, frame, ids, flags = (
            (*lettered, ()) if rule == LETTERS_SCORING
            else (head_tokens, frame_tokens, labels, starts))
        return rule, self.estimate(rule, live, head, frame, ids, resident, flags)

    ClassifyScoring.choose = choose


def _summary(record) -> dict:
    """Pull the numbers to report from one query's run record."""
    (query,) = record["queries"]
    metrics = query["metrics"]
    nodes = query["measurements"]["executed_plan"]["nodes"]
    return {
        "status": query["status"],
        "runtime_s": query["runtime_s"],
        "input_tokens": metrics["input_tokens"],
        "input_tokens_per_second": metrics["input_tokens_per_second"],
        "fresh_tokens": metrics["fresh_tokens"],
        "cost_usd": metrics["cost_usd"],
        "label_accuracy": metrics["accuracy"]["label_accuracy"],
        "per_predicate": metrics["accuracy"]["per_predicate"],
        "scoring": {node["attributes"]["spec"]["name"]:
                    node["attributes"]["spec"]["scoring"]
                    for node in nodes if node["type"] == "quail.ai_classify"},
        "collection_id": record["collection_id"],
    }


@app.function(image=image, gpu="H100!", memory=98304, timeout=5400,
              volumes=VOLUMES)
def run_rules(run: str) -> dict:
    """Run each query under each rule and save the records and a summary."""
    from pathlib import Path

    import quail
    from quail.bench import quailb
    from quail.planner.label_scoring import ClassifyScoring

    original = ClassifyScoring.choose
    root = Path(f"/results/ablations/classify-rules-{run}")
    summary = {"run": run, "model": MODEL, "queries": {}}
    started = time.perf_counter()
    for query_id, sf in QUERIES:
        summary["queries"][query_id] = {"sf": sf, "rules": {}}
        for rule in RULES:
            name = rule or "planner"
            _force(rule, original)
            output = root / query_id / name
            record = quailb.run_suite(
                [query_id], sf=sf,
                config=quail.EngineConfig(model=MODEL, device="h100-sxm"),
                data_dir=f"{DATA_DIR}/sf{sf}", output_dir=str(output))
            summary["queries"][query_id]["rules"][name] = {
                "directory": str(output), **_summary(record)}
            print(json.dumps({query_id: {name: summary["queries"][query_id]
                                         ["rules"][name]}}), flush=True)
            VOLUMES["/results"].commit()
    ClassifyScoring.choose = original
    summary["total_s"] = time.perf_counter() - started
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    VOLUMES["/results"].commit()
    return summary


@app.local_entrypoint()
def main():
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    call = run_rules.spawn(run)
    print(f"run {run} function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=2), flush=True)
