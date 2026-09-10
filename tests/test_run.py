"""Run and rescore published-format inputs without an inference engine."""

import json
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quail_b
from quail_b.data import (
    DATA_SEED,
    GROUND_TRUTH_ROOT,
    PUBLISHED_CORPORA,
    SOURCE_REVISIONS,
    corpus_identity,
)
from quail_b.run import _write_json


def _inputs(root, sf):
    spec = quail_b.get_query("IMDB-4")
    tables = {
        "reviews": pa.table({"id": ["r0", "r1"], "body": ["good", "bad"]}),
        "aspects": pa.table({"id": ["a0"], "aspect": ["acting"]}),
    }
    corpus_id = PUBLISHED_CORPORA[sf]
    corpus = root / GROUND_TRUTH_ROOT / "corpora" / corpus_id
    corpus.mkdir(parents=True)
    manifest = corpus_identity(tables, sf, DATA_SEED, SOURCE_REVISIONS)
    manifest["corpus_id"] = corpus_id
    _write_json(corpus / "manifest.json", manifest)
    for name, table in tables.items():
        pq.write_table(table, corpus / f"{name}.parquet")
    collection_id = f"gt_{sf}"
    collection = root / GROUND_TRUTH_ROOT / "collections" / collection_id
    collection.mkdir(parents=True)
    label_sets = {}
    predicates = [
        ("filter", template, None) for template in spec.aliases[0].filters
    ] + [("join", spec.joins[0].template, "aspects")]
    for index, (kind, template, right) in enumerate(predicates):
        key = f"predicate_{index}"
        label_id = f"ls_{sf}_{index}"
        label_sets[key] = label_id
        directory = root / GROUND_TRUTH_ROOT / "label_sets" / key / label_id
        directory.mkdir(parents=True)
        _write_json(directory / "manifest.json", {
            "status": "complete", "rows": 2, "source_rows": {"test": 2},
            "predicate": {
                "key": key, "template": template, "kind": kind,
                "left_table": "reviews", "left_column": "body",
                "right_table": right, "right_column": "aspect" if right else None,
            },
        })
        pq.write_table(pa.table({
            "predicate_key": [key, key], "label_set_id": [label_id, label_id],
            "left_id": ["r0", "r1"], "right_id": ["a0" if right else None] * 2,
            "answer": [True, True],
        }), directory / "labels.parquet")
    _write_json(collection / "manifest.json", {
        "status": "complete", "collection_id": collection_id,
        "corpus_id": corpus_id, "scale_factor": sf,
        "label_sets": label_sets, "summary": {"model": "test"},
    })
    _write_json(corpus / "active_collection.json", {"collection_id": collection_id})
    return collection_id


def test_join_runs_at_all_scales_and_report_cli(tmp_path):
    calls = []

    def execute(spec, tables):
        calls.append(spec.id)
        assert set(tables) == {"reviews", "aspects"}
        return quail_b.RunOutput(
            {("r", index): pa.table({"r": ["r0", "r1"], "answer": [True, True]})
             for index in range(len(spec.aliases[0].filters))},
            {0: pa.table({
                "r": ["r0", "r1"], "a": ["a0", "a0"], "answer": [True, True]})},
            pa.table({"r": ["r0", "r1"], "a": ["a0", "a0"]}),
            runtime_s=2.0)

    for sf in PUBLISHED_CORPORA:
        _inputs(tmp_path, sf)
        destination = tmp_path / f"run-{sf}"
        record = quail_b.run(
            execute, queries=["IMDB-4"], scale_factor=sf,
            output_dir=destination, root=tmp_path, gpu_count=2,
            gpu_hourly_rate_usd=3.6)
        metrics = record["queries"][0]["metrics"]
        assert metrics["input_rows"] == {"r": 2, "a": 1}
        assert metrics["document_pairs_per_second"] == 1.0
        assert metrics["cost_usd"] == 0.004
        assert metrics["accuracy"]["output_accuracy"]["exact_match"]
        before = (destination / "report.md").read_text()
        quail_b.report(destination, rescore=False)
        assert (destination / "report.md").read_text() == before
        completed = subprocess.run(
            [sys.executable, "-m", "quail_b", "report", str(destination),
             "--root", str(tmp_path)], capture_output=True, text=True, timeout=30)
        assert completed.returncode == 0, completed.stderr
        assert str(destination / "report.md") in completed.stdout
    assert calls == ["IMDB-4"] * 3


def test_saved_answers_survive_scoring_failure_and_can_move(tmp_path):
    import shutil

    collection = _inputs(tmp_path, 0.1)

    def execute(spec, tables):
        return quail_b.RunOutput(
            None, None, pa.table({"r": ["r0"], "a": ["a0"]}), runtime_s=2.0)

    destination = tmp_path / "run"
    quail_b.run(execute, queries=["IMDB-4"], output_dir=destination, root=tmp_path)
    record = json.loads((destination / "run.json").read_text())
    assert record["queries"][0]["metrics"]["document_pairs_per_second"] is None
    shutil.copytree(destination, tmp_path / "copied")
    quail_b.report(tmp_path / "copied", root=tmp_path)
    record["queries"][0]["files"]["rows"] = "../../outside.parquet"
    _write_json(destination / "run.json", record)
    with pytest.raises(ValueError, match="outside the query directory"):
        quail_b.report(destination, root=tmp_path)

    def missing_label(spec, tables):
        table = tmp_path / GROUND_TRUTH_ROOT / "label_sets/predicate_2/ls_0.1_2"
        labels = pq.read_table(table / "labels.parquet")
        pq.write_table(labels.slice(0, 1), table / "labels.parquet")
        return quail_b.RunOutput(
            None, {0: pa.table({"r": ["r1"], "a": ["a0"], "answer": [True]})},
            pa.table({"r": ["r1"], "a": ["a0"]}), runtime_s=2.0)

    quail_b.run(
        missing_label, queries=["IMDB-4"], output_dir=tmp_path / "pinned",
        root=tmp_path, collection_id=collection)
    with pytest.raises(ValueError, match="loaded 1 rows"):
        quail_b.report(tmp_path / "pinned", root=tmp_path)
    assert (tmp_path / "pinned/IMDB-4/rows.parquet").exists()
