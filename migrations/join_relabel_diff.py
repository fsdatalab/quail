"""Check whether relabelling under changed join prompts changed answers.

PR #52 reworked how join prompts are built, which moved every join
predicate's `predicate_version` and so invalidated its labels. That
forces a relabel on cost grounds alone. This measures whether the
answers themselves moved, by comparing two ground-truth collections
pair by pair on (left_id, right_id).

Run from the repository root and keep the function call id in the tee
file:

    uv run modal run migrations/join_relabel_diff.py \
        --before-id gt_306dac4fc83883c7a5bcc86f4d103f32 \
        --after-id gt_04231c5de83cdf9e7e68fc03849959d6 \
        2>&1 | tee results/join_relabel_diff.log

Writes /results/migrations/join_relabel_diff_<after_id>.json on the
quail-results volume. Predicates whose row count changed between the
two collections are reported as skipped rather than compared.
"""

import json

import modal

from quail.bench.judge_pass import (
    PREDICATES,
    VOLUME_ROOT,
    _label_dir_by_id,
    data_image,
    results_vol,
)

app = modal.App("quail-milestone1")

NAME = "join_relabel_diff"


@app.function(image=data_image, memory=16384, timeout=1800,
              volumes={"/results": results_vol})
def compare(before_id: str, after_id: str) -> str:
    import pyarrow.parquet as pq

    results_vol.reload()
    root = VOLUME_ROOT / "collections"
    with open(root / before_id / "summary.json") as f:
        before = json.load(f)["label_sets"]
    with open(root / after_id / "summary.json") as f:
        after = json.load(f)["label_sets"]

    def answers(spec, label_set_id):
        table = pq.read_table(
            _label_dir_by_id(spec, label_set_id) / "labels.parquet",
            columns=["left_id", "right_id", "answer"])
        return {(row["left_id"], row["right_id"]): row["answer"]
                for row in table.to_pylist()}

    predicates = {}
    for spec in PREDICATES:
        if spec.kind != "join" or spec.key not in before:
            continue
        b, a = before[spec.key], after[spec.key]
        if b["rows"] != a["rows"]:
            predicates[spec.key] = {
                "compared": False, "reason": "row count changed",
                "before_rows": b["rows"], "after_rows": a["rows"]}
            continue
        old = answers(spec, b["label_set_id"])
        new = answers(spec, a["label_set_id"])
        shared = old.keys() & new.keys()
        predicates[spec.key] = {
            "compared": True,
            "predicate_key": spec.key,
            "source_policy": spec.source_policy,
            "pairs": len(shared),
            "keys_only_in_before": len(old.keys() - new.keys()),
            "keys_only_in_after": len(new.keys() - old.keys()),
            "answer_differences": sum(old[k] != new[k] for k in shared),
        }

    payload = {
        "cell": NAME,
        "before_collection_id": before_id,
        "after_collection_id": after_id,
        "predicates": predicates,
    }
    out = VOLUME_ROOT.parent.parent.parent / "migrations"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{NAME}_{after_id}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    results_vol.commit()
    payload["volume_path"] = str(path)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return json.dumps(payload, sort_keys=True)


@app.local_entrypoint()
def main(before_id: str, after_id: str):
    call = compare.spawn(before_id, after_id)
    print(f"function call id (join_relabel_diff): {call.object_id}",
          flush=True)
    print(call.get(), flush=True)
