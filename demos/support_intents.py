"""Find customer intents in support conversations with SAE features.

Run locally on a machine with one H100 GPU:

    uv run python demos/support_intents.py --output-dir /tmp/support-intents

Or run on Modal with the companion script:

    uv run modal run --detach demos/support_intents_modal.py \
      2>&1 | tee /tmp/quail-support-intents.log

The input is the QUAIL-B support_traces table: customer-support
conversations from tau-bench, with each conversation's opening request,
its domain (airline or retail), and whether the agent completed the task.
The query reads the SAE features of each opening request. The analysis
weights each feature by how rare it is, keeps each request's strongest
weighted features, counts them, prints the requests where each one weighs
most, and compares how often each one occurs in failed and completed
conversations.

The feature lists are saved as features.parquet and the report as
report.json under the output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

MODEL = "qwen3.5-2b-sae-bf16"
DEVICE = "h100-sxm"
QUESTION = "{0}\n\nWhat is the customer asking for?"
# Features kept per request after weighting.
TOP_PER_REQUEST = 3
# A comparison needs this many requests with the feature.
MINIMUM_REQUESTS = 20


def load_conversations(scale_factor: float = 1.0) -> pa.Table:
    """Return the support conversations, one row each."""
    from quail_b import load_table

    table = load_table("support_traces", scale_factor=scale_factor)
    return table.select(["id", "request", "task_id", "domain", "reward"])


def read_features(conversations: pa.Table) -> pa.Table:
    """Return (id, intent) with each opening request's SAE features."""
    import quail

    config = quail.EngineConfig(model=MODEL, device=DEVICE)
    with quail.Session(config) as session:
        session.register("conversations", quail.DocumentProvider.from_table(
            conversations, id_col="id"))
        query = (
            session.docs("conversations").alias("t")
            .ai_sae_features(quail.prompt(QUESTION, quail.col("t.request")),
                             pooling="last", name="intent")
            .select("t.id", "intent"))
        return query.collect()


def flatten(features: pa.ChunkedArray) -> tuple[np.ndarray, ...]:
    """Return (row, feature, strength) arrays, one entry per feature."""
    lists = features.combine_chunks()
    pairs = pc.list_flatten(lists)
    return (pc.list_parent_indices(lists).to_numpy(),
            pairs.field("feature").to_numpy(),
            pairs.field("strength").to_numpy())


def top_features(rows, features, strengths, *, num_features: int,
                 num_requests: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (entries, weights): each request's strongest weighted features.

    A feature that occurs in most requests describes the form of a
    support request, not one intent. The weight multiplies a strength by
    log(num_requests / requests with the feature), as TF-IDF weights a
    term, so such features rank low. ``entries`` indexes the flat arrays.
    """
    requests_with = np.bincount(features, minlength=num_features)
    weights = strengths * np.log(num_requests / requests_with[features])
    order = np.lexsort((-weights, rows))
    first = np.r_[0, np.flatnonzero(np.diff(rows[order])) + 1]
    rank = np.arange(len(order)) - np.repeat(first, np.diff(
        np.r_[first, len(order)]))
    return order[rank < TOP_PER_REQUEST], weights


def examples(entries, rows, features, weights, conversations, feature,
             count: int = 3) -> list[str]:
    """Return the requests where a feature weighs most, one per task."""
    picked = entries[features[entries] == feature]
    picked = picked[np.argsort(-weights[picked])]
    tasks = conversations.column("task_id").to_pylist()
    requests = conversations.column("request").to_pylist()
    seen, found = set(), []
    for row in rows[picked].tolist():
        if tasks[row] not in seen:
            seen.add(tasks[row])
            found.append(requests[row].strip())
        if len(found) == count:
            break
    return found


def analyze(conversations: pa.Table, intents: pa.Table, *,
            num_features: int = 32_768, top: int = 8) -> dict:
    """Return the most frequent intent features and the failure comparison."""
    # the query selects the id first, then the feature lists
    order = pc.index_in(intents.column(0), conversations.column("id"))
    conversations = conversations.take(order)
    rows, features, strengths = flatten(intents.column(1))
    total = conversations.num_rows
    entries, weights = top_features(rows, features, strengths,
                                    num_features=num_features,
                                    num_requests=total)

    def described(feature, **counts):
        return {"feature": feature, **counts,
                "examples": examples(entries, rows, features, weights,
                                     conversations, feature)}

    requests_with = np.bincount(features[entries], minlength=num_features)
    intents_found = [described(feature, requests=int(requests_with[feature]))
                     for feature in np.argsort(-requests_with)[:top].tolist()]
    failed = (conversations.column("reward").to_numpy() < 1.0)
    in_failed = failed[rows[entries]]
    failed_share = np.bincount(features[entries][in_failed],
                               minlength=num_features) / max(failed.sum(), 1)
    completed_share = np.bincount(features[entries][~in_failed],
                                  minlength=num_features) / max(
                                      (~failed).sum(), 1)
    difference = np.where(requests_with >= MINIMUM_REQUESTS,
                          failed_share - completed_share, 0.0)
    more_often_failed = [
        described(feature,
                  share_of_failed=round(float(failed_share[feature]), 3),
                  share_of_completed=round(float(completed_share[feature]), 3))
        for feature in np.argsort(-difference)[:5].tolist()]
    return {"conversations": total,
            "failed": int(failed.sum()),
            "intents": intents_found,
            "more_often_in_failed": more_often_failed}


def run(output_dir: Path, scale_factor: float = 1.0) -> dict:
    """Read the features, analyze them, and save report.json."""
    conversations = load_conversations(scale_factor)
    intents = read_features(conversations)
    output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(intents, output_dir / "features.parquet")
    report = analyze(conversations, intents)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale-factor", type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.scale_factor), indent=2))


if __name__ == "__main__":
    main()
