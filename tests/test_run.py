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
        ("filter", filter_spec.prompt, None)
        for filter_spec in spec._info.filters
    ] + [("join", spec._info.joins[0].prompt, "aspects")]
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
            {filter_spec.id: pa.table({
                "r": ["r0", "r1"], "answer": [True, True]})
             for filter_spec in spec._info.filters},
            {"join-1": pa.table({
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
        assert record["schema_version"] == 2
        metrics = record["queries"][0]["metrics"]
        assert metrics["input_rows"] == {"r": 2, "a": 1}
        assert metrics["input_tokens"] is None
        assert metrics["input_tokens_per_second"] is None
        assert metrics["fresh_tokens"] is None
        assert metrics["minimum_tokens"] is None
        assert metrics["regret_tokens"] is None
        assert metrics["evaluated_document_pairs"] == 2
        assert metrics["document_pairs_per_second"] == 1.0
        assert metrics["cost_usd"] == 0.004
        assert metrics["cost_usd_per_million_input_tokens"] is None
        assert metrics["kv_regret_percent"] is None
        assert metrics["accuracy"]["output_accuracy"]["exact_match"]
        plan_path = destination / "IMDB-4/plan.substrait"
        assert plan_path.read_bytes() == quail_b.get_query("IMDB-4").plan_bytes
        assert record["queries"][0]["files"]["plan"] == "plan.substrait"
        before = (destination / "report.md").read_text()
        quail_b.report(destination, rescore=False)
        assert (destination / "report.md").read_text() == before
        completed = subprocess.run(
            [sys.executable, "-m", "quail_b", "report", str(destination),
             "--root", str(tmp_path)], capture_output=True, text=True, timeout=30)
        assert completed.returncode == 0, completed.stderr
        assert str(destination / "report.md") in completed.stdout
    assert calls == ["IMDB-4"] * 3


@pytest.mark.parametrize("hourly_rate", [None, 0.0, 3.6])
def test_prompt_pieces_give_the_minimum_and_the_regret(
        tmp_path, monkeypatch, hourly_rate):
    import quail_b.minimum

    def encode(texts):
        return [[byte + 1 for byte in text.encode("utf-8")] for text in texts]

    loaded = []
    monkeypatch.setattr(quail_b.minimum, "load_tokenizer",
                        lambda name: loaded.append(name) or encode)
    _inputs(tmp_path, 0.1)
    spec = quail_b.get_query("IMDB-4")
    pieces = {
        "tokenizer": "test-tokenizer", "preamble": [1, 2],
        "filters": [
            {"id": filter_spec.id, "tail": [10 + index, 20]}
            for index, filter_spec in enumerate(spec._info.filters)
        ],
        "joins": [{"id": "join-1", "anchor": "r", "frame": [30, 31],
                   "label": [40], "tail": [50, 51, 52]}],
    }

    def execute(spec, tables):
        return quail_b.RunOutput(
            {filter_spec.id: pa.table({
                "r": ["r0", "r1"], "answer": [True, True]})
             for filter_spec in spec._info.filters},
            {"join-1": pa.table({
                "r": ["r0", "r1"], "a": ["a0", "a0"], "answer": [True, True]})},
            pa.table({"r": ["r0", "r1"], "a": ["a0", "a0"]}),
            runtime_s=2.0, measurements={"fresh_tokens": 1000},
            prompt_pieces=pieces)

    destination = tmp_path / "run"
    record = quail_b.run(execute, queries=["IMDB-4"], output_dir=destination,
                         root=tmp_path, gpu_count=2,
                         gpu_hourly_rate_usd=hourly_rate)
    metrics = record["queries"][0]["metrics"]
    # the preamble once, then "good" and "bad" (4 and 3 tokens, sharing
    # nothing); per review the two filter tails and the frame, which
    # share nothing; per pair the label, "acting", and the tail
    stages = len(spec._info.filters)
    assert stages == 2
    minimum = (2 + 4 + 3) + 2 * (2 * 2 + 2) + 2 * (1 + 6 + 3)
    assert metrics["minimum_tokens"] == minimum
    assert metrics["input_tokens"] == 65
    assert metrics["input_tokens_per_second"] == 32.5
    assert metrics["regret_tokens"] == 1000 - minimum
    cost = None if hourly_rate is None else 2 / 3600 * 2 * hourly_rate
    cost_per_million = None if cost is None else cost / 65 * 1e6
    regret_percent = 100 * (1000 - minimum) / 1000
    assert metrics["cost_usd_per_million_input_tokens"] == cost_per_million
    assert metrics["kv_regret_percent"] == regret_percent
    assert loaded == ["test-tokenizer"]
    saved = json.loads((destination / "IMDB-4/prompt_pieces.json").read_text())
    assert saved == pieces

    quail_b.report(destination, root=tmp_path)
    rescored = json.loads((destination / "run.json").read_text())
    assert rescored["queries"][0]["metrics"] == metrics
    table = pq.read_table(destination / "measurements.parquet")
    assert table.to_pylist()[0] == {
        "query": "IMDB-4", "runtime_s": 2.0, "fresh_tokens": 1000,
        "input_tokens": 65, "input_tokens_per_second": 32.5,
        "minimum_tokens": minimum, "regret_tokens": 1000 - minimum,
        "evaluated_document_pairs": 2, "input_rows": 3,
        "answers_evaluated": 2 * stages + 2, "answers_correct": 2 * stages + 2,
        "predicted_rows": 2, "expected_rows": 2, "matching_rows": 2,
        "cost_usd": cost,
        "cost_usd_per_million_input_tokens": cost_per_million,
        "kv_regret_percent": regret_percent,
    }
    assert "| IMDB-4 | r: 2, a: 1 | 65 | 1000 |" in (
        destination / "report.md").read_text()

    report_text = (destination / "report.md").read_text()
    assert "$/million input tokens | KV regret (%)" in report_text
    assert "| 95.9 |" in report_text

    def empty(spec, tables):
        output = execute(spec, tables)
        output.filter_answers = {
            key: table.slice(0, 0) for key, table in output.filter_answers.items()}
        output.join_answers = {
            key: table.slice(0, 0) for key, table in output.join_answers.items()}
        output.rows = output.rows.slice(0, 0)
        output.measurements = {"fresh_tokens": 0}
        return output

    empty_record = quail_b.run(
        empty, queries=["IMDB-4"], output_dir=tmp_path / "empty", root=tmp_path,
        gpu_hourly_rate_usd=3.6)
    empty_metrics = empty_record["queries"][0]["metrics"]
    assert empty_metrics["input_tokens"] == 0
    assert empty_metrics["cost_usd_per_million_input_tokens"] is None
    assert empty_metrics["kv_regret_percent"] is None

    def broken(spec, tables):
        output = execute(spec, tables)
        output.prompt_pieces = {"tokenizer": "test-tokenizer", "joins": []}
        return output

    with pytest.raises(ValueError, match="missing filter operators"):
        quail_b.run(broken, queries=["IMDB-4"], output_dir=tmp_path / "broken",
                    root=tmp_path)

    def without_fresh(spec, tables):
        output = execute(spec, tables)
        output.measurements = {}
        return output

    with pytest.raises(ValueError, match="fresh_tokens"):
        quail_b.run(without_fresh, queries=["IMDB-4"],
                    output_dir=tmp_path / "no-fresh", root=tmp_path)

    def too_few(spec, tables):
        output = execute(spec, tables)
        output.measurements = {"fresh_tokens": 1}
        return output

    with pytest.raises(ValueError, match="below the minimum"):
        quail_b.run(too_few, queries=["IMDB-4"],
                    output_dir=tmp_path / "too-few", root=tmp_path)


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
            None,
            {"join-1": pa.table({
                "r": ["r1"], "a": ["a0"], "answer": [True]})},
            pa.table({"r": ["r1"], "a": ["a0"]}), runtime_s=2.0)

    quail_b.run(
        missing_label, queries=["IMDB-4"], output_dir=tmp_path / "pinned",
        root=tmp_path, collection_id=collection)
    with pytest.raises(ValueError, match="loaded 1 rows"):
        quail_b.report(tmp_path / "pinned", root=tmp_path)
    assert (tmp_path / "pinned/IMDB-4/rows.parquet").exists()
