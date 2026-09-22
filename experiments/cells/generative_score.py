"""Compare AI.SCORE on the rerankers with Qwen3 4B, Qwen3 32B, and DiffusionGemma.

Every model scores the same QUAIL-B predicates at scale factor 0.1,
with each predicate's template as QUAIL-B writes it:

- IMDB-1, BIO-1, FEV-1, LEP-1: one score per document.
- IMDB-2: one score per review and aspect pair.
- LEP-2: one score per excerpt and passage pair. Its labels are the
  dataset's citation links; the other labels are Qwen3 32B answers.

On the generative models the cell also runs AI_FILTER on the four
filter predicates and counts rows where `score > 0.5` differs from
the AI_FILTER answer.

Run from the repository root and tee every line. The local entrypoint
prints the planner's estimate for every model and query before it
submits the GPU functions:

    uv run modal run --detach experiments/cells/generative_score.py \
      2>&1 | tee results/generative-score.log

Each model writes, under /results/ai-score/generative-<run id>/:

    <model>.json            summary: time, tokens, cost, accuracy
    <model>/<query>.parquet one row per scored document or pair

--models picks the models and --queries the QUAIL-B query ids.
"""

import json
import os
import time
import uuid

import modal
import numpy as np

from quail.bench.images import gpu_image

app = modal.App("quail-milestone1")
image = gpu_image()
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache", create_if_missing=True)
volumes = {
    "/root/.cache/huggingface": hf_cache,
    "/root/.cache/kernels": kernel_cache,
    "/results": results_vol,
}

SCALE_FACTOR = 0.1
MODELS = (
    "qwen3-reranker-0.6b-bf16",
    "qwen3-reranker-4b-bf16",
    "qwen3-4b-fp8",
    "qwen3-32b-fp8",
    "diffusion-gemma-26b-a4b-fp8",
)
QUERIES = ("IMDB-1", "BIO-1", "FEV-1", "LEP-1", "IMDB-2", "LEP-2")


def _predicate(query_id):
    """The QUAIL-B predicate one single-operator query asks."""
    from quail_b.predicates import PREDICATES
    from quail_b.queries import get_query

    (operator,) = get_query(query_id)._info.operators
    return next(spec for spec in PREDICATES if spec.template == operator.prompt)


def _quoted(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def score_sql(predicate) -> str:
    """AI.SCORE over every document, or every pair, of one predicate."""
    template = _quoted(predicate.template)
    if predicate.kind == "filter":
        return (f"SELECT l.id, AI.SCORE(PROMPT({template}, "
                f"l.{predicate.left_column})) AS score "
                f"FROM {predicate.left_table} l")
    return (f"SELECT l.id, r.id, AI.SCORE(PROMPT({template}, "
            f"l.{predicate.left_column}, r.{predicate.right_column})) AS score "
            f"FROM {predicate.left_table} l "
            f"CROSS JOIN {predicate.right_table} r")


def filter_sql(predicate) -> str:
    """The AI_FILTER query that asks the same filter prompt."""
    return (f"SELECT l.id FROM {predicate.left_table} l "
            f"WHERE AI_FILTER(PROMPT({_quoted(predicate.template)}, "
            f"l.{predicate.left_column}))")


def _session(model, tables, tokenizer=None):
    import quail
    from quail.planner.plan import EngineConfig

    session = quail.Session(
        EngineConfig(model=model, device="h100-sxm", gpus=1),
        **({"tokenizer": tokenizer} if tokenizer is not None else {}))
    for name, table in tables.items():
        session.register(name, quail.DocumentProvider.from_table(
            table, id_col="id", identity=f"quailb-{SCALE_FACTOR}-{name}"))
    return session


def _suite(query_ids):
    """The QUAIL-B tables, checked against the published corpus, and labels."""
    from quail_b.benchmark import load_benchmark

    return load_benchmark(list(query_ids), scale_factor=SCALE_FACTOR)


def roc_auc(scores, labels) -> float | None:
    """Probability a positive outscores a negative; ties count one half."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ordered = scores[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    ends = np.r_[starts[1:], len(ordered)]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end + 1) / 2
    positive_ranks = ranks[labels].sum()
    return float((positive_ranks - positives * (positives + 1) / 2)
                 / (positives * negatives))


def average_precision(scores, labels) -> float | None:
    """Precision summed over each distinct threshold's gain in recall."""
    labels = np.asarray(labels, dtype=bool)
    if not labels.any():
        return None
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    ordered, hits = scores[order], labels[order]
    # a threshold keeps every row tied with its last row
    last = np.r_[ordered[1:] != ordered[:-1], True]
    true_kept = np.cumsum(hits)[last]
    kept = np.arange(1, len(hits) + 1)[last]
    recall_gain = np.diff(np.r_[0, true_kept]) / labels.sum()
    return float((recall_gain * true_kept / kept).sum())


def accuracy(scores, labels) -> dict:
    """Ranking and threshold agreement of scores with reference labels."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    selected = scores > 0.5
    true_selected = int((selected & labels).sum())
    top = np.argsort(-scores, kind="mergesort")[:positives]
    return {
        "rows": len(labels),
        "reference_positives": positives,
        "roc_auc": roc_auc(scores, labels),
        "average_precision": average_precision(scores, labels),
        "precision_at_reference_count": (
            float(labels[top].mean()) if positives else None),
        "selected_above_0_5": int(selected.sum()),
        "agreement_at_0_5": float((selected == labels).mean()),
        "precision_at_0_5": (
            true_selected / int(selected.sum()) if selected.any() else None),
        "recall_at_0_5": true_selected / positives if positives else None,
    }


def _labels_for(table, predicate, truth):
    """The reference answer of every scored row, in the table's order."""
    import pyarrow as pa

    labels = truth.predicates[predicate.key].table
    if predicate.kind == "filter":
        keys = pa.table({"left_id": table.column("l.id").cast(pa.string())})
        on = ["left_id"]
    else:
        keys = pa.table({"left_id": table.column("l.id").cast(pa.string()),
                         "right_id": table.column("r.id").cast(pa.string())})
        on = ["left_id", "right_id"]
    keys = keys.append_column("row", pa.array(np.arange(keys.num_rows)))
    joined = keys.join(labels.select([*on, "answer"]), on).sort_by("row")
    if joined.num_rows != keys.num_rows or joined.column("answer").null_count:
        raise ValueError(f"{predicate.key}: a scored row has no reference label")
    return joined.column("answer").to_numpy(zero_copy_only=False)


def _run(session, sql):
    from quail.execution.execute import execute_query

    result = execute_query(session.sql(sql))
    return result.collect(), result.report


def _timing(report, model) -> dict:
    from quail.specs import H100_USD_PER_HOUR

    requested = report["fresh_tokens"] + report["cached_tokens"]
    return {
        "wall_s": report["wall_s"],
        "estimated_seconds": report.get("estimated_seconds"),
        "fresh_tokens": report["fresh_tokens"],
        "cached_tokens": report["cached_tokens"],
        "requested_input_tokens": requested,
        "input_tokens_per_second": requested / report["wall_s"],
        "usd_per_query": report["wall_s"] / 3600 * H100_USD_PER_HOUR,
        "boot": report.get("boot"),
    }


def _save_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(value, file, indent=2)


@app.function(image=image, gpu="H100!", memory=98304, timeout=4 * 3600,
              volumes=volumes)
def score_model(model: str, query_ids: list[str], run_dir: str,
                prediction: dict) -> str:
    """Score every query with one model and save rows and a summary."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    from quail.specs import MODELS as SPECS

    suite = _suite(query_ids)
    truth = suite.ground_truth
    generative = SPECS[model].role != "reranker"
    summary = {
        "model": model,
        "gpu_device_name": torch.cuda.get_device_name(0),
        "scale_factor": SCALE_FACTOR,
        "corpus_id": suite.corpus_id,
        "reference_collection": truth.collection_id,
        "reference_model": truth.reference_model,
        "prediction": prediction,
        "queries": {},
    }
    summary_path = f"{run_dir}/{model}.json"
    session = _session(model, suite.tables)
    try:
        for query_id in query_ids:
            predicate = _predicate(query_id)
            started = time.perf_counter()
            table, report = _run(session, score_sql(predicate))
            scores = table.column("score").to_numpy()
            labels = _labels_for(table, predicate, truth)
            record = {
                "predicate": predicate.key,
                "label_source": predicate.source_policy,
                "input_documents": {
                    "l": suite.tables[predicate.left_table].num_rows,
                    **({"r": suite.tables[predicate.right_table].num_rows}
                       if predicate.kind == "join" else {})},
                "score": {**_timing(report, model),
                          **accuracy(scores, labels)},
            }
            rows = table.append_column("reference", pa.array(labels))
            if generative and predicate.kind == "filter":
                kept, filter_report = _run(session, filter_sql(predicate))
                answers = np.isin(
                    table.column("l.id").to_numpy(zero_copy_only=False),
                    kept.column("l.id").to_numpy(zero_copy_only=False))
                differ = (scores > 0.5) != answers
                record["ai_filter"] = {
                    **_timing(filter_report, model),
                    "selected": int(answers.sum()),
                    "rows_where_score_disagrees": int(differ.sum()),
                    "largest_disagreeing_distance_from_0_5": (
                        float(np.abs(scores[differ] - 0.5).max())
                        if differ.any() else None),
                }
                rows = rows.append_column("ai_filter", pa.array(answers))
            path = f"{run_dir}/{model}/{query_id}.parquet"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            pq.write_table(rows, path)
            record["rows_volume_path"] = path
            record["cell_s"] = time.perf_counter() - started
            summary["queries"][query_id] = record
            _save_json(summary_path, summary)
            results_vol.commit()
            print(json.dumps({query_id: record}), flush=True)
    finally:
        session.close()
        kernel_cache.commit()
    summary["result_volume_path"] = summary_path
    _save_json(summary_path, summary)
    results_vol.commit()
    return summary_path


def plan_estimates(models, query_ids) -> dict:
    """The planner's estimated seconds per model and query, on the CPU."""
    suite = _suite(query_ids)
    estimates = {}
    for model in models:
        session = _session(model, suite.tables)
        try:
            estimates[model] = {
                query_id: session.sql(score_sql(_predicate(query_id)))
                .plan().estimated_seconds
                for query_id in query_ids
            }
        finally:
            session.close()
    return estimates


@app.local_entrypoint()
def main(models: str = ",".join(MODELS), queries: str = ",".join(QUERIES)):
    selected = [name.strip() for name in models.split(",") if name.strip()]
    query_ids = [name.strip() for name in queries.split(",") if name.strip()]
    unknown = (set(selected) - set(MODELS)) | (set(query_ids) - set(QUERIES))
    if unknown:
        raise ValueError(f"unknown models or queries: {sorted(unknown)}")
    estimates = plan_estimates(selected, query_ids)
    print("PREDICTION (planner estimate, seconds):", flush=True)
    print(json.dumps(estimates, indent=2), flush=True)
    run_dir = f"/results/ai-score/generative-{uuid.uuid4().hex[:12]}"
    print(f"run directory: {run_dir}", flush=True)
    calls = {model: score_model.spawn(model, query_ids, run_dir,
                                      estimates[model])
             for model in selected}
    for model, call in calls.items():
        print(f"function call id: {call.object_id} ({model})", flush=True)
    for model, call in calls.items():
        print(f"{model}: {call.get()}", flush=True)
