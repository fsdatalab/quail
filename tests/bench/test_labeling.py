"""CPU checks for the QUAIL-B labeling pass."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import bench.labeling as labeling
from bench.labeling import (
    _check_reused_label_set,
    _compact_label_parts,
    _corpus_identity,
    _saved_verification_sample,
)
from quail_b.data import DATA_SEED, SOURCE_REVISIONS, corpus_identity
from quail_b.predicates import MODEL_NAME, PREDICATES


def _spec(key):
    return next(spec for spec in PREDICATES if spec.key == key)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_corpus_identity_label_reuse_and_saved_parts(monkeypatch, tmp_path):
    rows = {"reviews": [{"id": "r0", "body": "a"}, {"id": "r1", "body": "b"}]}
    assert (corpus_identity(rows, 0.1, DATA_SEED, SOURCE_REVISIONS)
            == _corpus_identity(rows, 0.1))
    assert _corpus_identity(rows, 0.1) == _corpus_identity(
        {"reviews": list(rows["reviews"])}, 0.1)
    for changed in ({"reviews": list(reversed(rows["reviews"]))},
                    {"reviews": [{"id": "r0", "body": "a"},
                                 {"id": "r1", "body": "changed"}]}):
        assert (_corpus_identity(rows, 0.1)["corpus_id"]
                != _corpus_identity(changed, 0.1)["corpus_id"])

    # the fixed join order changes only the labels whose anchor was automatic
    for key, changed in (
        ("quailb.fever.passage.supports_claim", True),
        ("quailb.imdb.review.discusses_ending", False),
        ("quailb.lepard.excerpt.cites_passage", False),
    ):
        old = labeling._label_set_identity(
            _spec(key), "c_test", "0" * 64, judge=labeling.QUAIL_JUDGE_SPEC)
        new = labeling.label_set_identity(_spec(key), "c_test", "0" * 64)
        assert (new["label_set_id"] != old["label_set_id"]) == changed

    monkeypatch.setattr(labeling, "ROOT", tmp_path)
    spec = _spec("quailb.imdb.review.mentions_positive_aspect")
    table_manifest = {"rows": 2, "ordered_rows_full_hash": "abc"}
    _write_json(tmp_path / "corpora/c_original/manifest.json", {
        "corpus_id": "c_original", "corpus_full_hash": "full-original",
        "tables": {"reviews": table_manifest}})
    _write_json(
        tmp_path / "label_sets" / spec.workload / spec.slug / "ls_old/manifest.json",
        {"status": "complete", "label_set_id": "ls_old",
         "corpus_id": "c_original", "corpus_full_hash": "full-original"})
    source_collection = {"collection_id": "gt_middle", "corpus_id": "c_middle",
                         "label_sets": {spec.key: "ls_old"}}
    target = {"tables": {"reviews": dict(table_manifest)}}

    reused = _check_reused_label_set(spec, "ls_old", source_collection, target)
    assert reused["source_collection_id"] == "gt_middle"
    assert reused["source_corpus_id"] == "c_original"

    target["tables"]["reviews"]["ordered_rows_full_hash"] = "changed"
    with pytest.raises(ValueError, match="table reviews changed"):
        _check_reused_label_set(spec, "ls_old", source_collection, target)

    # saved label parts resume a run and compact once
    spec = _spec("quailb.imdb.review.discusses_ending")
    parts = tmp_path / "label_sets" / spec.workload / spec.slug / "ls_test" / "parts"
    parts.mkdir(parents=True)
    saved = [{"answer": i % 2 == 0, "label_source": MODEL_NAME,
              "left_id": f"rv{i}", "right_id": None} for i in range(85)]
    # only the first of two parts is on disk, the resume case
    pq.write_table(pa.Table.from_pylist(saved), parts / "part_000000_000085.parquet")
    corpus = {"reviews": [{"id": f"rv{i}", "body": f"review {i}"}
                          for i in range(170)]}

    sample = _saved_verification_sample(
        corpus, {spec.key: {"label_set_id": "ls_test"}}, specs=(spec,))

    assert len(sample.rows[spec.key]) == 16
    assert sample.rows[spec.key][0] == (
        {"id": "rv0", "body": "review 0"}, None, True)

    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    for name, ids, answers in (("000000_000002", ["a", "b"], [True, False]),
                               ("000002_000003", ["c"], [True]),
                               ("000000_000003", ["a", "b", "c"],
                                [True, False, True])):
        pq.write_table(pa.table({"id": ids, "answer": answers}),
                       parts_dir / f"part_{name}.parquet")
    # the last part is a leftover generation of boundaries, which the
    # caller does not name and compaction must therefore ignore
    parts = [parts_dir / "part_000000_000002.parquet",
             parts_dir / "part_000002_000003.parquet"]

    path, rows = _compact_label_parts(tmp_path, parts)
    second_path, second_rows = _compact_label_parts(tmp_path, parts)

    assert rows == second_rows == 3
    assert path == second_path == tmp_path / "labels.parquet"
    assert pq.read_table(path).to_pylist() == [
        {"id": "a", "answer": True},
        {"id": "b", "answer": False},
        {"id": "c", "answer": True},
    ]
