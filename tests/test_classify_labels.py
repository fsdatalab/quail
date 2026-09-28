"""CPU checks for the classification labeling cell."""

import json
import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quail.bench.labeling as labeling
from experiments.cells import classify_labels as cell
from quail_b.data import GROUND_TRUTH_ROOT
from quail_b.labels import load_ground_truth
from quail_b.predicates import MODEL_NAME, PREDICATE_BY_KEY, PREDICATES

# billing refund request, billing refund status, billing address change,
# technical login problem, one word per token
BILLING = [(1, 2, 3), (1, 2, 4), (1, 5, 6), (7, 8, 9)]


def test_trie_lists_each_prefix_once():
    trie = cell.label_trie(BILLING)
    assert trie == {(): [1, 7], (1,): [2, 5], (1, 2): [3, 4], (1, 5): [6],
                    (7,): [8], (7, 8): [9]}
    with pytest.raises(ValueError, match="no tokens"):
        cell.label_trie([(1,), ()])


def test_scores_sum_prefix_log_probabilities_and_ties_go_first():
    log = math.log
    logprobs = {(): {1: log(0.8), 7: log(0.1)}, (1,): {2: log(0.7), 5: log(0.2)},
                (1, 2): {3: log(0.6), 4: log(0.3)}, (1, 5): {6: log(0.9)},
                (7,): {8: log(0.8)}, (7, 8): {9: log(0.9)}}
    winner, scores = cell.score_labels(BILLING, logprobs)
    assert winner == 0
    assert [round(math.exp(s), 3) for s in scores] == [0.336, 0.168, 0.144, 0.072]
    winner, _ = cell.score_labels([(1,), (2,)], {(): {1: -1.0, 2: -1.0}})
    assert winner == 0


def test_shards_cover_whole_parts():
    for rows, shards in ((17_711, 8), (50_000, 2), (4_144, 1), (300, 4)):
        bounds = cell.shard_bounds(rows, shards)
        assert len(bounds) <= shards
        assert bounds[0][0] == 0 and bounds[-1][1] == rows
        assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))
        assert all(start % cell.ROWS_PER_PART == 0 for start, _ in bounds)


def _corpus(root, corpus_id, sf, claims):
    table = {"rows": len(claims), "ordered_rows_full_hash": corpus_id}
    labeling._atomic_json(root / "corpora" / corpus_id / "manifest.json", {
        "corpus_id": corpus_id, "corpus_full_hash": corpus_id + "-full",
        "scale_factor": sf,
        "tables": {"claims": table, "reviews": {"rows": 1,
                                                "ordered_rows_full_hash": "r"}}})
    pq.write_table(pa.table({"id": [c[0] for c in claims],
                             "claim": [c[1] for c in claims]}),
                   root / "corpora" / corpus_id / "claims.parquet")


def test_labels_copy_by_content_and_join_a_reused_collection(
        monkeypatch, tmp_path):
    root = tmp_path / GROUND_TRUTH_ROOT
    spec = PREDICATE_BY_KEY["quailb.fever.claim.topic"]
    old = next(p for p in PREDICATES if p.left_table == "reviews")
    monkeypatch.setattr(labeling, "ROOT", root)
    monkeypatch.setattr(labeling, "PREDICATES", (old,))
    monkeypatch.setattr(cell, "PUBLISHED_CORPORA",
                        {1.0: "c_big", 0.1: "c_small"})
    _corpus(root, "c_big", 1.0, [("c0", "Paris"), ("c1", "Messi"),
                                 ("c2", "Lincoln")])
    _corpus(root, "c_small", 0.1, [("s0", "Lincoln"), ("s1", "Paris")])

    manifest, rows = cell.corpus_rows(1.0, spec)
    identity = cell.classify_identity(spec, manifest)
    answers = {"Paris": "geography", "Messi": "sports", "Lincoln": "history"}
    cell.write_part(labeling._part_path(spec, identity, 0, 3), [
        cell._label_row(spec, identity, "c_big", row, answers[row["claim"]],
                        [-1.0] * len(spec.labels))
        for row in rows])
    source = cell.complete_manifest(spec, identity, 3)
    assert source["label_rows"]["history"] == 1
    small = cell.copy_labels(0.1, spec, source)
    assert small["label_set_id"] != source["label_set_id"]
    copied = pq.read_table(small["compact_path"]).to_pylist()
    assert [(r["left_id"], r["label"]) for r in copied] == [
        ("s0", "history"), ("s1", "geography")]

    labeling._atomic_json(root / "collections/gt_old/manifest.json", {
        "collection_id": "gt_old", "status": "complete",
        "corpus_id": "c_small", "scale_factor": 0.1,
        "label_sets": {old.key: "ls_old"}})
    old_dir = labeling._label_dir(old, {"label_set_id": "ls_old"})
    labeling._atomic_json(old_dir / "manifest.json", {
        "label_set_id": "ls_old", "status": "complete", "corpus_id": "c_small",
        "corpus_full_hash": "c_small-full", "rows": 1, "true_rows": 1,
        "source_rows": {MODEL_NAME: 1},
        "predicate": {"kind": "filter", "template": old.template,
                      "left_table": "reviews"}})
    summary = labeling.activate_reused_collection(
        0.1, "c_small", "gt_old", "", new_specs=(spec,))
    assert summary["reused_predicates"] == 1
    assert summary["new_predicates"] == 1
    assert summary["label_sets"][spec.key]["label_rows"]["history"] == 1

    truth = load_ground_truth(root=tmp_path, scale_factor=0.1,
                              collection_id=summary["collection_id"],
                              templates={spec.template})
    labels = truth.predicates[spec.key]
    assert labels.kind == "classify"
    assert labels.answer("s0") == "history"
    assert json.loads((root / "corpora/c_small/active_collection.raw-v1.json")
                      .read_text())["collection_id"] == summary["collection_id"]


def test_quail_runner_skips_classification_queries():
    from quail.bench import quailb
    from quail_b.queries import get_query

    assert quailb.runs_on_quail(get_query("IMDB-4"))
    assert not quailb.runs_on_quail(get_query("IMDB-11"))
