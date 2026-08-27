"""CPU checks for QUAIL-B ground-truth loading and scoring."""

import json
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

import quail
from quail.bench.evaluate import (
    GROUND_TRUTH_ROOT,
    BenchmarkEvaluator,
    GroundTruthCollection,
    LocalVolumeFiles,
    PredicateLabels,
    add_query_metrics,
    corpus_identity,
    load_ground_truth,
    summarize_queries,
)
from quail.catalog import DocumentProvider
from quail.planner.plan import EngineConfig


FILTER = "Judge the review.\n\n{0}\nAnswer TRUE or FALSE."
JOIN = "Judge the pair.\n\n{0}\nAspect: {1}\nAnswer TRUE or FALSE."


def _predicate(key, template, kind, left_table, right_table=None):
    columns = {"reviews": "body", "aspects": "aspect"}
    return {
        "key": key,
        "template": template,
        "kind": kind,
        "left_table": left_table,
        "left_column": columns[left_table],
        "right_table": right_table,
        "right_column": columns[right_table] if right_table else None,
    }


def _truth():
    filter_key = "test.review.filter"
    join_key = "test.review.aspect"
    return GroundTruthCollection(
        collection_id="gt_test",
        corpus_id="c_test",
        scale_factor=0.1,
        reference_model="qwen3-32b-fp8",
        predicates={
            filter_key: PredicateLabels(
                key=filter_key,
                label_set_id="ls_filter",
                predicate=_predicate(
                    filter_key, FILTER, "filter", "reviews"),
                answers={("r0", None): True, ("r1", None): False},
                source_rows={"qwen3-32b-fp8": 2},
            ),
            join_key: PredicateLabels(
                key=join_key,
                label_set_id="ls_join",
                predicate=_predicate(
                    join_key, JOIN, "join", "reviews", "aspects"),
                answers={
                    ("r0", "a0"): True,
                    ("r0", "a1"): False,
                    ("r1", "a0"): False,
                    ("r1", "a1"): True,
                },
                source_rows={"qwen3-32b-fp8": 4},
            ),
        },
    )


def _query(tmp_path):
    pq.write_table(pa.table({
        "id": ["r0", "r1"],
        "body": ["good film", "bad film"],
    }), tmp_path / "reviews.parquet")
    pq.write_table(pa.table({
        "id": ["a0", "a1"],
        "aspect": ["acting", "ending"],
    }), tmp_path / "aspects.parquet")
    sess = quail.Session(EngineConfig(gpus=1), tokenizer=str.split)
    sess.register("reviews", DocumentProvider.from_parquet(
        str(tmp_path / "reviews.parquet"), id_col="id"))
    sess.register("aspects", DocumentProvider.from_parquet(
        str(tmp_path / "aspects.parquet"), id_col="id"))
    query = (sess.docs("reviews").alias("r")
             .ai_filter(quail.prompt(FILTER, quail.col("r.body")))
             .ai_join(
                 sess.docs("aspects").alias("a"),
                 quail.prompt(JOIN, quail.col("r.body"),
                              quail.col("a.aspect")),
                 anchor="r")
             .select("r.id", "a.id"))
    return sess, query


def _run(query, join_answers):
    def execute(_payload):
        return {
            "filters": {"r": {0: [1], 1: [0]}},
            "joins": [{
                "rows": {0: join_answers},
                "anchor_index": [0],
                "partner_index": [[0], [1]],
                "anchor": "r",
                "partners": ["a"],
            }],
            "wall_s": 2.0,
            "boot_s": 3.0,
            "boot_kind": "cold",
            "boot": {},
            "fresh_tokens": 100,
            "store": None,
            "peak_gib": 1.0,
        }

    return query.run(_execute=execute)


def test_evaluator_scores_answers_and_final_rows(tmp_path):
    sess, query = _query(tmp_path)
    try:
        result = _run(query, [0, 0])
        corpus = {
            "reviews": [{"id": "r0", "body": "good film"},
                        {"id": "r1", "body": "bad film"}],
            "aspects": [{"id": "a0", "aspect": "acting"},
                        {"id": "a1", "aspect": "ending"}],
        }
        evaluation = BenchmarkEvaluator(_truth(), corpus).evaluate(
            query, result)
    finally:
        sess.close()

    assert evaluation["answer_accuracy"]["evaluated"] == 4
    assert evaluation["answer_accuracy"]["correct"] == 3
    assert evaluation["answer_accuracy"]["accuracy"] == 0.75
    assert evaluation["output_accuracy"] == {
        "predicted_rows": 0,
        "expected_rows": 1,
        "matching_rows": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "exact_match": False,
        "false_positive_rows": 0,
        "false_negative_rows": 1,
    }
    assert evaluation["input_document_rows"] == 4
    assert evaluation["unique_input_documents"] == 4


def test_query_cost_token_and_document_metrics(tmp_path):
    sess, query = _query(tmp_path)
    try:
        result = _run(query, [1, 0])
        corpus = {
            "reviews": [{"id": "r0", "body": "good film"},
                        {"id": "r1", "body": "bad film"}],
            "aspects": [{"id": "a0", "aspect": "acting"},
                        {"id": "a1", "aspect": "ending"}],
        }
        evaluation = BenchmarkEvaluator(_truth(), corpus).evaluate(
            query, result)
    finally:
        sess.close()

    row = add_query_metrics(
        {"wall_s": 2.0, "boot_s": 3.0, "fresh_tokens": 100},
        evaluation, h100_usd_per_hour=3.6, gpus=1)
    assert row["runtime_s"] == 2.0
    assert row["runtime_with_boot_s"] == 5.0
    assert row["documents_per_second"] == 2.0
    assert row["tokens_per_second"] == 50.0
    assert row["inference_calls"] == 4
    assert row["inference_cost_usd"] == 0.002
    assert row["inference_cost_per_token_usd"] == 0.00002
    assert row["inference_cost_per_million_tokens_usd"] == 20.0
    assert row["cost_with_boot_usd"] == 0.005
    assert row["accuracy"]["output_accuracy"]["exact_match"] is True

    summary = summarize_queries([row], 3.6, 1)
    assert summary["tokens_processed"] == 100
    assert summary["documents_per_second"] == 2.0
    assert summary["answer_accuracy"]["accuracy"] == 1.0


def test_load_ground_truth_from_volume_layout(tmp_path):
    collection_id = "gt_test"
    label_set_id = "ls_filter"
    predicate_key = "test.review.filter"
    collection_dir = (tmp_path / GROUND_TRUTH_ROOT / "collections"
                      / collection_id)
    label_dir = (tmp_path / GROUND_TRUTH_ROOT / "label_sets" / "test"
                 / "review_filter" / label_set_id)
    collection_dir.mkdir(parents=True)
    (label_dir / "parts").mkdir(parents=True)
    collection = {
        "status": "complete",
        "collection_id": collection_id,
        "corpus_id": "c_test",
        "scale_factor": 0.1,
        "label_sets": {predicate_key: label_set_id},
        "summary": {"model": "qwen3-32b-fp8"},
    }
    (collection_dir / "manifest.json").write_text(json.dumps(collection))
    old_dir = (tmp_path / GROUND_TRUTH_ROOT / "collections" / "gt_old")
    old_dir.mkdir(parents=True)
    (old_dir / "manifest.json").write_text(json.dumps({
        "status": "complete",
        "collection_id": "gt_old",
        "corpus_id": "c_test",
        "scale_factor": 0.1,
    }))
    corpus_dir = (tmp_path / GROUND_TRUTH_ROOT / "corpora" / "c_test")
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "active_collection.json").write_text(json.dumps({
        "collection_id": collection_id,
    }))
    manifest = {
        "status": "complete",
        "rows": 2,
        "source_rows": {"qwen3-32b-fp8": 2},
        "predicate": _predicate(
            predicate_key, FILTER, "filter", "reviews"),
    }
    (label_dir / "manifest.json").write_text(json.dumps(manifest))
    pq.write_table(pa.Table.from_pylist([
        {"predicate_key": predicate_key, "label_set_id": label_set_id,
         "answer": True, "left_id": "r0", "right_id": None},
        {"predicate_key": predicate_key, "label_set_id": label_set_id,
         "answer": False, "left_id": "r1", "right_id": None},
    ]), label_dir / "parts" / "part_000.parquet")

    loaded = load_ground_truth(
        LocalVolumeFiles(tmp_path), scale_factor=0.1,
        corpus_id="c_test")

    assert loaded.collection_id == collection_id
    assert loaded.answer(predicate_key, "r0") is True
    assert loaded.answer(predicate_key, "r1") is False

    files = LocalVolumeFiles(tmp_path)
    files.write_json("benchmarks/quailb/runs/qb_test/query.json",
                     {"query": "TEST-1"})
    saved = json.loads((tmp_path / "benchmarks/quailb/runs/qb_test"
                        / "query.json").read_text())
    assert saved == {"query": "TEST-1"}
    answer_table = pa.table({"document": [0, 1],
                             "answer": [True, False]})
    files.write_parquet(
        "benchmarks/quailb/runs/qb_test/answers.parquet", answer_table)
    assert pq.read_table(
        tmp_path / "benchmarks/quailb/runs/qb_test/answers.parquet"
    ).equals(answer_table)


def test_corpus_identity_matches_judge_pass_implementation():
    from quail.bench.judge_pass import _corpus_identity
    from quail.bench.quailb import DATA_SEED, SOURCE_REVISIONS

    rows = {"reviews": [{"id": "r0", "body": "text"}]}
    expected = _corpus_identity(rows, 0.1)
    got = corpus_identity(rows, 0.1, DATA_SEED, SOURCE_REVISIONS)

    assert got == expected


def test_report_writer_creates_markdown_and_plot(tmp_path):
    from quail.bench.quailb import _artifact_stem
    from reports.make_quailb_eval_plots import (
        make_plot,
        plot_path_for,
        write_report,
    )

    artifact_stem = _artifact_stem(
        datetime(2026, 8, 25, 12, tzinfo=timezone.utc),
        0.1, 1, "qwen3-4b-fp8")
    assert artifact_stem.startswith("20260825T120000Z-")
    answer = {
        "evaluated": 4, "correct": 3, "accuracy": 0.75,
        "precision": 1.0, "recall": 0.5, "f1": 2 / 3,
        "true_positive": 1, "true_negative": 2,
        "false_positive": 0, "false_negative": 1,
    }
    row = {
        "query": "TEST-1", "runtime_s": 2.0, "boot_s": 3.0,
        "inference_cost_usd": 0.002, "cost_with_boot_usd": 0.005,
        "tokens_processed": 100,
        "inference_cost_per_million_tokens_usd": 20.0,
        "documents_per_second": 2.0,
        "accuracy": {
            "answer_accuracy": answer,
            "output_accuracy": {"f1": 0.5},
        },
    }
    data = {
        "model": "qwen3-4b-fp8", "sf": 0.1, "gpus": 1,
        "artifact_stem": artifact_stem,
        "corpus_id": "c_test", "prediction": "Accuracy will exceed 70%.",
        "ground_truth": {
            "collection_id": "gt_test",
            "reference_model": "qwen3-32b-fp8",
        },
        "pricing": {"h100_usd_per_hour": 3.6},
        "raw_volume_path": "/results/benchmarks/quailb/runs/qb_test",
        "aggregate_volume_path": (
            "/results/benchmarks/quailb/runs/qb_test/"
            "20260825T120000Z-quailb-sf0.1-lf1-qwen3-4b-fp8.json"),
        "passes": {
            "warm": {
                "queries": [row],
                "summary": {
                    "queries_completed": 1, "query_runtime_s": 2.0,
                    "tokens_processed": 100, "inference_cost_usd": 0.002,
                    "cost_with_boot_usd": 0.005,
                    "answer_accuracy": answer,
                },
            },
        },
    }
    input_path = tmp_path / "results" / "benchmark" / "summary.json"
    input_path.parent.mkdir(parents=True)
    input_path.write_text(json.dumps(data))
    report_path = (tmp_path / "results" / "benchmark"
                   / f"{data['artifact_stem']}.md")
    plot_path = (tmp_path / "reports" / "plots" / "benchmark"
                 / f"{data['artifact_stem']}.png")

    make_plot(data, plot_path)
    write_report(data, input_path, report_path, plot_path)

    assert plot_path.read_bytes().startswith(b"\x89PNG")
    report = report_path.read_text()
    assert "Accuracy will exceed 70%." in report
    assert "Cost per 1M tokens" in report
    assert ("Figure: ../../reports/plots/benchmark/"
            f"{data['artifact_stem']}.png") in report
    assert ("Ground truth loading happens before the run starts. It is "
            "excluded from every runtime and cost metric.") in report
    assert data["aggregate_volume_path"] in report
    generated_plot = plot_path_for(data, report_path)
    assert generated_plot.name == f"{data['artifact_stem']}.png"
