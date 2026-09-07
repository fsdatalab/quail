"""CPU checks for QUAIL-B label loading and scoring, with no engine."""

import json
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from quailb.data import corpus_identity
from quailb.labels import (
    GROUND_TRUTH_ROOT,
    GroundTruthCollection,
    LocalVolumeFiles,
    PredicateLabels,
    _validate_label_set_corpora,
    load_ground_truth,
)
from quailb.queries import AliasSpec, JoinSpec, QuerySpec, queries
from quailb.scoring import (
    Evaluator,
    RunOutput,
    add_query_metrics,
    rows_from_answers,
    summarize_queries,
)

FILTER = "Judge the review.\n\n{0}\nAnswer TRUE or FALSE."
JOIN = "Judge the pair.\n\n{0}\nAspect: {1}\nAnswer TRUE or FALSE."
SPEC = QuerySpec(
    "TEST-1", "one filter then one join",
    (AliasSpec("r", "reviews", "body", (FILTER,)),
     AliasSpec("a", "aspects", "aspect")),
    (JoinSpec(JOIN, ("r", "a")),),
    ("r.id", "a.id"),
)
CORPUS = {
    "reviews": [{"id": "r0", "body": "good film"},
                {"id": "r1", "body": "bad film"}],
    "aspects": [{"id": "a0", "aspect": "acting"},
                {"id": "a1", "aspect": "ending"}],
}


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


def _output(join_answers, rows):
    return RunOutput(
        filter_answers={("r", 0): pa.table({
            "r": ["r0", "r1"], "answer": [True, False]})},
        join_answers={0: pa.table({
            "r": ["r0", "r0"], "a": ["a0", "a1"],
            "answer": [bool(answer) for answer in join_answers]})},
        rows=pa.table({"r": [row[0] for row in rows],
                       "a": [row[1] for row in rows]},
                      schema=pa.schema([("r", pa.string()),
                                        ("a", pa.string())])),
    )


def test_evaluator_scores_answers_and_final_rows():
    evaluation = Evaluator(_truth(), CORPUS).evaluate(
        SPEC, _output([0, 0], []))

    assert evaluation["answer_accuracy"]["evaluated"] == 4
    assert evaluation["answer_accuracy"]["correct"] == 3
    assert evaluation["answer_accuracy"]["accuracy"] == 0.75
    assert [item["op"] for item in evaluation["per_predicate"]] == [
        "filter", "join"
    ]
    # r0 passes the filter and pairs with a0; r1 pairs with a1 but is
    # filtered out, so one row is expected and none was returned
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


def test_rows_from_answers_joins_true_pairs_of_surviving_documents():
    output = _output([1, 1], [])
    rows = rows_from_answers(spec=SPEC, filter_answers=output.filter_answers,
                             join_answers=output.join_answers)
    # r0 passes its filter and pairs with both aspects; r1 was never asked
    assert rows.sort_by("a").to_pydict() == {"a": ["a0", "a1"], "r": ["r0", "r0"]}


def test_query_cost_token_and_document_metrics():
    evaluation = Evaluator(_truth(), CORPUS).evaluate(
        SPEC, _output([1, 0], [("r0", "a0")]))

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


def fever_truth():
    """Labels for FEV-9 over a three claim, three evidence corpus."""
    from quailb.judge_pass import PREDICATES, predicate_payload
    from quailb.prompts import F11, F13, REFUTE, SUPPORT

    corpus = {
        "claims": pa.table({"id": ["c0", "c1", "c2"],
                            "claim": ["person one", "person two", "a place"]}),
        "evidence": pa.table({"id": ["e0", "e1", "e2"],
                              "text": ["person one", "person two", "a place"]}),
    }
    true_pairs = {
        SUPPORT: {("c0", "e0"), ("c1", "e1"), ("c2", "e0"), ("c1", "e2")},
        REFUTE: {("c1", "e0"), ("c2", "e0"), ("c0", "e2")},
    }
    predicates = {}
    for spec in PREDICATES:
        if spec.template not in (F11, F13, SUPPORT, REFUTE):
            continue
        ids = corpus[spec.left_table]["id"].to_pylist()
        answers = (
            {(row_id, None): index < 2 for index, row_id in enumerate(ids)}
            if spec.kind == "filter" else {
                (left, right): (left, right) in true_pairs[spec.template]
                for left in ids
                for right in corpus[spec.right_table]["id"].to_pylist()
            }
        )
        predicates[spec.key] = PredicateLabels(
            spec.key, f"ls_{spec.key}", predicate_payload(spec), answers, {},
        )
    truth = GroundTruthCollection("gt_fev9", "c_fev9", 0.1, None, predicates)
    return corpus, truth


def test_fev9_expected_rows_follow_the_join_chain_and_every_filter():
    corpus, truth = fever_truth()
    from quailb.prompts import F11, F13

    spec = queries()["FEV-9"]
    assert [alias.filters for alias in spec.aliases] == [
        (F11,), (F13,), (F11,), (F13,)]
    assert [join.aliases for join in spec.joins] == [
        ("c1", "e1"), ("c2", "e1"), ("c2", "e2")]

    rows = Evaluator(truth, corpus).expected_rows(spec)

    # c2 is filtered out; e1 supports c0, refutes c1, and c1 has e1 as
    # different supporting evidence
    assert rows.to_pydict() == {
        "c1": ["c0"], "c2": ["c1"], "e1": ["e0"], "e2": ["e1"]}


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
    assert loaded.key_for_template(FILTER) == predicate_key

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


def _reused_label_layout(tmp_path):
    files = LocalVolumeFiles(tmp_path)
    predicate_key = "test.review.filter"
    table_manifest = {
        "rows": 2,
        "full_hash": "same-review-table-hash",
    }
    for corpus_id in ("c_source", "c_target"):
        path = (tmp_path / GROUND_TRUTH_ROOT / "corpora" / corpus_id)
        path.mkdir(parents=True)
        (path / "manifest.json").write_text(json.dumps({
            "corpus_id": corpus_id,
            "corpus_full_hash": f"hash-{corpus_id}",
            "tables": {"reviews": table_manifest},
        }))
    source_path = (tmp_path / GROUND_TRUTH_ROOT / "collections"
                   / "gt_source")
    source_path.mkdir(parents=True)
    (source_path / "manifest.json").write_text(json.dumps({
        "status": "complete",
        "collection_id": "gt_source",
        "corpus_id": "c_source",
        "label_sets": {predicate_key: "ls_source"},
    }))
    collection = {
        "corpus_id": "c_target",
        "reused_label_sets": {
            predicate_key: {
                "source_collection_id": "gt_source",
                "source_corpus_id": "c_source",
                "required_tables": ["reviews"],
                "verified_table_manifests": {
                    "reviews": table_manifest,
                },
            },
        },
    }
    manifests = {
        predicate_key: {
            "label_set_id": "ls_source",
            "corpus_id": "c_source",
            "predicate": _predicate(
                predicate_key, FILTER, "filter", "reviews"),
        },
    }
    return files, collection, manifests


def test_reused_label_set_accepts_identical_table_manifest(tmp_path):
    files, collection, manifests = _reused_label_layout(tmp_path)

    _validate_label_set_corpora(files, collection, manifests)


def test_reused_label_set_accepts_transitive_collection_reuse(tmp_path):
    files, collection, manifests = _reused_label_layout(tmp_path)
    source_path = (tmp_path / GROUND_TRUTH_ROOT / "collections"
                   / "gt_source" / "manifest.json")
    source = json.loads(source_path.read_text())
    source["corpus_id"] = "c_intermediate"
    source_path.write_text(json.dumps(source))

    _validate_label_set_corpora(files, collection, manifests)


def test_reused_label_set_rejects_changed_table_manifest(tmp_path):
    files, collection, manifests = _reused_label_layout(tmp_path)
    target_path = (tmp_path / GROUND_TRUTH_ROOT / "corpora" / "c_target"
                   / "manifest.json")
    target = json.loads(target_path.read_text())
    target["tables"]["reviews"]["full_hash"] = "changed"
    target_path.write_text(json.dumps(target))

    try:
        _validate_label_set_corpora(files, collection, manifests)
    except ValueError as exc:
        assert "table reviews changed" in str(exc)
    else:
        raise AssertionError("changed table manifest was accepted")


def test_corpus_identity_matches_judge_pass_implementation():
    from quailb.data import DATA_SEED, SOURCE_REVISIONS
    from quailb.judge_pass import _corpus_identity

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
