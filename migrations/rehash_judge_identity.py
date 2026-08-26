"""One-shot migration: move label sets to the ids the current
``JUDGE_SPEC`` implies, without running the model.

Ran once on 2026-08-26, when ``max_num_batched_tokens`` and
``max_num_seqs`` were dropped from ``JUDGE_SPEC``. Those knobs cannot
change a greedy one-token decode, but they were hashed into
``JUDGE_ID`` and from there into every ``label_set_id``, so removing
them moved all 23 label sets without changing a single answer. It
migrated 15 filter predicates and 17,057 labels; the 8 join predicates
needed a real relabel because their own ``predicate_version`` had
moved. See ``reports/2026-08-26-bio6-full-terms-and-judge-identity.md``.

Kept for the record and reusable the next time a field that cannot
change an answer leaves ``JUDGE_SPEC``.

Report first, write nothing::

    uv run modal run migrations/rehash_judge_identity.py

Then apply::

    uv run modal run migrations/rehash_judge_identity.py --apply
"""

import json
import os
from pathlib import Path

import modal

from quail.bench.judge_pass import (
    PREDICATES,
    PredicateSpec,
    VOLUME_ROOT,
    _atomic_json,
    _atomic_parquet,
    _expected_parts,
    _expected_rows,
    _label_dir,
    _label_dir_by_id,
    _load_corpus,
    _part_bounds,
    _part_path,
    _parts_stats,
    data_image,
    judgment_identity,
    label_set_identity,
    results_vol,
)

app = modal.App("quail-milestone1")


def _rehash_rows(rows: list[dict], label_set_id: str) -> list[dict]:
    """Same answers, new identity: label_set_id is stored on every row
    and judgment_id is derived from it, so both have to be rewritten
    when a label set moves. example_full_hash is stored, so this needs
    no corpus and no model."""
    out = []
    for row in rows:
        row = dict(row)
        row["label_set_id"] = label_set_id
        row["judgment_id"] = judgment_identity(
            label_set_id, row["example_full_hash"])
        out.append(row)
    return out


def _rehash_label_dir(spec: PredicateSpec, old_dir: Path, identity: dict,
                      corpus_rows: dict[str, list[dict]],
                      expected_rows: int) -> dict:
    """Rewrite one label set under a new id, rebuilding its parts at
    the boundaries the current code writes.

    The source is labels.parquet, not the old parts directory: a parts
    directory can hold more than one generation of files (see
    _parts_stats), so reading it whole would double rows. Parts are
    rebuilt rather than copied because the boundaries move with
    filter-group membership and with a join's right-hand table, and a
    later labeling pass resumes by testing for those exact names."""
    import pyarrow.parquet as pq

    compact = old_dir / "labels.parquet"
    if not compact.exists():
        raise FileNotFoundError(f"{spec.key}: {compact} is missing")
    table = pq.read_table(compact)
    if table.num_rows != expected_rows:
        raise ValueError(
            f"{spec.key}: {compact} has {table.num_rows} rows, expected "
            f"{expected_rows}")
    by_left: dict[str, list[dict]] = {}
    for row in _rehash_rows(table.to_pylist(), identity["label_set_id"]):
        by_left.setdefault(row["left_id"], []).append(row)

    left_rows = corpus_rows[spec.left_table]
    ordered = []
    for start, end in _part_bounds(spec, corpus_rows):
        part_rows = []
        for left in left_rows[start:end]:
            found = by_left.get(str(left["id"]))
            if found is None:
                raise ValueError(
                    f"{spec.key}: {compact} has no row for left id "
                    f"{left['id']}")
            part_rows.extend(found)
        if spec.kind == "join" and spec.source_policy != "lepard_passage_id":
            part_rows.sort(key=lambda row: (row["left_id"], row["right_id"]))
        _atomic_parquet(_part_path(spec, identity, start, end), part_rows)
        ordered.extend(part_rows)
    if len(ordered) != expected_rows:
        raise ValueError(
            f"{spec.key}: rebuilt {len(ordered)} rows, expected "
            f"{expected_rows}")
    # written here rather than through _compact_label_parts, which
    # keeps whatever labels.parquet it finds with a matching row count
    _atomic_parquet(_label_dir(spec, identity) / "labels.parquet", ordered)
    return _parts_stats(_expected_parts(spec, identity, corpus_rows))


@app.function(
    image=data_image, memory=8192, timeout=3600,
    volumes={"/results": results_vol})
def rehash_label_sets(collection_id: str | None = None,
                      apply: bool = False) -> str:
    """Move a collection's label sets to the ids the current
    JUDGE_SPEC implies, without running the model.

    Valid only when the judge change cannot alter an answer - dropping
    a scheduler capacity knob from JUDGE_SPEC, say. A predicate whose
    own predicate_version or corpus_full_hash moved is reported as
    needing a real relabel and is left untouched, as is one already
    sitting at its new id. The old directories stay on the volume."""
    results_vol.reload()
    corpora = sorted(p for p in (VOLUME_ROOT / "corpora").iterdir()
                     if p.is_dir())
    if len(corpora) != 1:
        raise ValueError(f"expected one corpus, found {len(corpora)}")
    corpus_dir, corpus_manifest, corpus_rows = _load_corpus(corpora[0].name)
    if collection_id is None:
        with open(corpus_dir / "active_collection.json") as f:
            collection_id = json.load(f)["collection_id"]
    with open(VOLUME_ROOT / "collections" / collection_id
              / "summary.json") as f:
        old_ids = {key: entry["label_set_id"]
                   for key, entry in json.load(f)["label_sets"].items()}

    plan = []
    for spec in PREDICATES:
        identity = label_set_identity(spec, corpus_manifest["corpus_id"],
                                      corpus_manifest["corpus_full_hash"])
        step = {"key": spec.key, "legacy_code": spec.legacy_code,
                "old_label_set_id": old_ids.get(spec.key),
                "new_label_set_id": identity["label_set_id"]}
        if step["old_label_set_id"] is None:
            step["action"] = "absent"
            plan.append(step)
            continue
        old_dir = _label_dir_by_id(spec, step["old_label_set_id"])
        with open(old_dir / "manifest.json") as f:
            old_manifest = json.load(f)
        if (old_manifest["predicate_full_hash"]
                != identity["predicate_full_hash"]):
            step["action"] = "relabel"
            step["reason"] = "predicate_version changed"
        elif (old_manifest["corpus_full_hash"]
                != identity["corpus_full_hash"]):
            step["action"] = "relabel"
            step["reason"] = "corpus_full_hash changed"
        elif step["old_label_set_id"] == step["new_label_set_id"]:
            step["action"] = "unchanged"
        else:
            step["action"] = "rehash"
        step["rows"] = old_manifest["rows"]
        step["true_rows"] = old_manifest["true_rows"]
        if step["action"] == "rehash" and apply:
            new_dir = _label_dir(spec, identity)
            expected_rows = _expected_rows(spec, corpus_rows)
            stats = _rehash_label_dir(spec, old_dir, identity, corpus_rows,
                                      expected_rows)
            for field in ("rows", "true_rows", "false_rows", "source_rows"):
                if stats[field] != old_manifest[field]:
                    raise ValueError(
                        f"{spec.key}: {field} changed from "
                        f"{old_manifest[field]!r} to {stats[field]!r}")
            _atomic_json(new_dir / "manifest.json", {
                **old_manifest, **identity,
                "expected_rows": expected_rows,
                "compact_path": str(new_dir / "labels.parquet"),
                "compact_rows": stats["rows"],
                "rehashed_from": str(old_dir)})
            step["written"] = str(new_dir)
        plan.append(step)

    if apply:
        results_vol.commit()
    result = {
        "collection_id": collection_id,
        "corpus_id": corpus_manifest["corpus_id"],
        "judge_id": JUDGE_ID,
        "applied": apply,
        "counts": {action: sum(1 for s in plan if s["action"] == action)
                   for action in sorted({s["action"] for s in plan})},
        "plan": plan,
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return json.dumps(result, sort_keys=True)


@app.local_entrypoint()
def main(collection: str | None = None, apply: bool = False):
    """Report the plan, or apply it with ``--apply``.

    ``--collection`` picks a collection other than the corpus's active
    one.
    """
    call = rehash_label_sets.spawn(collection, apply)
    print(f"function call id (rehash_label_sets): {call.object_id}",
          flush=True)
    print(call.get(), flush=True)
