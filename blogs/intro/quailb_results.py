"""Load and validate the saved QUAIL-B results used by the launch blog."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from quail_b.queries import get_query
from quail_b.run import _query_hash

SOURCE_RUN = "/results/benchmarks/quailb/20260922T190951Z-efa30103"
CORPUS = "c_1aa2c4f0d0b6c816fd37aa5748c33341"
COLLECTION = "gt_cd3ebdb784f64b9e028e50ea73cdedd0"
SOL_COLLECTION = "gt_be81cb241d74555dc2da79b5b0662554"
REFERENCE_MODEL = "qwen3-32b-fp8"
QUERY_ORDER = (
    *[f"IMDB-{number}" for number in range(1, 11)],
    *[f"BIO-{number}" for number in range(1, 5)],
    *[f"FEV-{number}" for number in range(1, 11)],
    *[f"LEP-{number}" for number in range(1, 6)],
    "AGENT-1",
    "AGENT-2",
)
PIPELINE_QUERIES = (
    "IMDB-4",
    "IMDB-5",
    "IMDB-6",
    "IMDB-7",
    "FEV-4",
    "FEV-6",
    "LEP-4",
    "LEP-5",
)
METHOD_FILES = {
    "quail": "quailb-gigatoken-quail.json",
    "stock_vllm": "quailb-gigatoken-stock.json",
    "pipelined_vllm": "quailb-gigatoken-pipelined.json",
}
METHOD_ENGINES = {
    "quail": "quail",
    "stock_vllm": "stock_vllm",
    "pipelined_vllm": "pipelined_vllm",
}
CURRENT_TO_SOL_QUERY = {"LEP-5": "LEP-7"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _measurement(item: dict) -> dict:
    metrics = item["metrics"]
    accuracy = metrics["accuracy"]
    answers = accuracy["answer_accuracy"]
    output = accuracy["output_accuracy"]
    return {
        "runtime_s": float(item["runtime_s"]),
        "input_tokens": int(metrics["input_tokens"]),
        "input_tokens_per_second": float(metrics["input_tokens_per_second"]),
        "fresh_tokens": int(metrics["fresh_tokens"]),
        "regret_tokens": metrics["regret_tokens"],
        "cost_usd": float(metrics["cost_usd"]),
        "answers_evaluated": int(answers["evaluated"]),
        "answers_correct": int(answers["correct"]),
        "predicted_rows": int(output["predicted_rows"]),
        "expected_rows": int(output["expected_rows"]),
        "matching_rows": int(output["matching_rows"]),
    }


def _validate_suite(suite: dict, method: str) -> None:
    if suite["status"] != "complete":
        raise ValueError(f"{method} run is not complete")
    expected = {
        "scale_factor": 0.1,
        "corpus_id": CORPUS,
        "collection_id": COLLECTION,
        "reference_model": REFERENCE_MODEL,
        "gpu_count": 1,
    }
    for field, value in expected.items():
        if suite[field] != value:
            raise ValueError(f"{method} has unexpected {field}: {suite[field]}")
    metadata = suite["metadata"]
    if metadata["engine"] != METHOD_ENGINES[method]:
        raise ValueError(f"{method} has unexpected engine {metadata['engine']}")
    if metadata["model"] != "qwen3-4b-fp8":
        raise ValueError(f"{method} has unexpected model {metadata['model']}")
    if metadata["prompt_format"] != "raw-v1":
        raise ValueError(f"{method} does not use raw prompts")
    items = {item["id"]: item for item in suite["queries"]}
    if tuple(items) != QUERY_ORDER:
        raise ValueError(f"{method} query order does not match QUAIL-B")
    for query, item in items.items():
        if item["status"] != "complete":
            raise ValueError(f"{method} {query} is not complete")
        if item["definition_hash"] != _query_hash(get_query(query)):
            raise ValueError(f"{method} {query} definition changed")


def _validate_collections(workdir: Path) -> None:
    old = _load(workdir / "quailb-sol-collection.json")
    new = _load(workdir / "quailb-run-collection.json")
    if old["collection_id"] != SOL_COLLECTION:
        raise ValueError("unexpected SoL reference collection")
    if new["collection_id"] != COLLECTION:
        raise ValueError("unexpected run reference collection")
    if old["corpus_id"] != new["corpus_id"] or new["corpus_id"] != CORPUS:
        raise ValueError("SoL and measured runs use different corpora")
    changed = {
        key
        for key in set(old["label_sets"]) & set(new["label_sets"])
        if old["label_sets"][key] != new["label_sets"][key]
    }
    if changed:
        raise ValueError(f"reference labels changed: {sorted(changed)}")
    removed = set(old["label_sets"]) - set(new["label_sets"])
    if removed != {"quailb.biodex.report.involves_female_patient"}:
        raise ValueError(f"unexpected removed labels: {sorted(removed)}")
    added = set(new["label_sets"]) - set(old["label_sets"])
    expected_added = {
        "quailb.biodex.reaction.is_cardiovascular",
        "quailb.biodex.reaction.is_neurological",
    }
    if added != expected_added:
        raise ValueError(f"unexpected added labels: {sorted(added)}")


def load_results(workdir: Path) -> tuple[dict, dict]:
    """Return measured rows and SoL estimates after validating their identity."""
    workdir = Path(workdir)
    suites = {
        method: _load(workdir / filename)
        for method, filename in METHOD_FILES.items()
    }
    for method, suite in suites.items():
        _validate_suite(suite, method)
    run_ids = {suite["run_id"] for suite in suites.values()}
    if len(run_ids) != 1:
        raise ValueError(f"methods came from different runs: {sorted(run_ids)}")
    _validate_collections(workdir)

    rows = {
        method: {
            item["id"]: _measurement(item)
            for item in suite["queries"]
        }
        for method, suite in suites.items()
    }
    base = _load(workdir / "quailb-gigatoken-sol-base.json")
    bio4 = _load(workdir / "quailb-gigatoken-sol-bio4.json")
    if base["corpus_id"] != CORPUS or base["collection_id"] != SOL_COLLECTION:
        raise ValueError("base SoL source does not match the measured corpus")
    if bio4["corpus_id"] != CORPUS or bio4["collection_id"] != COLLECTION:
        raise ValueError("BIO-4 SoL source does not match the measured run")
    if bio4["plan_sha256"] != hashlib.sha256(
        get_query("BIO-4").plan_bytes
    ).hexdigest():
        raise ValueError("BIO-4 SoL plan changed")

    sol = {}
    for query in QUERY_ORDER:
        if query == "BIO-4":
            estimate = bio4["estimate"]
            sol[query] = {
                "runtime_s": float(estimate["sol_s"]),
                "input_tokens": rows["quail"][query]["input_tokens"],
                "fresh_tokens": int(estimate["tokens"]),
            }
            continue
        source = CURRENT_TO_SOL_QUERY.get(query, query)
        if base["query_hashes"][source] != _query_hash(get_query(query)):
            raise ValueError(f"{query} SoL definition changed")
        estimate = base["sol"][source]
        sol[query] = {
            "runtime_s": float(estimate["sol_s"]),
            "input_tokens": int(estimate["requested_tokens"]),
            "fresh_tokens": int(estimate["tokens"]),
        }
    return rows, sol
