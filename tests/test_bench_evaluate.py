"""CPU checks for QUAIL-B ground-truth loading and scoring."""

import json
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quail
from quail.bench.evaluate import (
    GROUND_TRUTH_ROOT,
    BenchmarkEvaluator,
    GroundTruthCollection,
    LocalVolumeFiles,
    PredicateLabels,
    add_query_metrics,
    _validate_label_set_corpora,
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


@pytest.mark.parametrize("backend", [
    "quail", "stock_vllm", "pipelined_vllm", "pipelined_sglang",
])
def test_fev9_filters_every_input_and_reuses_existing_predicate_labels(backend):
    from quail.bench.judge_pass import PREDICATES, predicate_payload
    from quail.bench.quailb import F11, F13, REFUTE, SUPPORT, queries
    from quail.planner import collect_operators
    from quail.planner.plan import Refusal
    from quail.runtime.result import build_result_declaration

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
                for left in ids for right in corpus[spec.right_table]["id"].to_pylist()
            }
        )
        predicates[spec.key] = PredicateLabels(
            spec.key, f"ls_{spec.key}", predicate_payload(spec), answers, {},
        )
    truth = GroundTruthCollection("gt_fev9", "c_fev9", 0.1, None, predicates)
    evaluator = BenchmarkEvaluator(truth, corpus)

    with quail.Session(EngineConfig(backend=backend), tokenizer=lambda text: list(
        text.encode("utf-8"))) as session:
        for name, table in corpus.items():
            session.register(name, DocumentProvider.from_table(table, id_col="id"))
        query = queries(session)["FEV-9"][1]()
        assert not isinstance(query.plan(), Refusal)
        scans, filters, joins = collect_operators(query.logical)
        survivors, pairs = evaluator._expected_answer_tables(query, scans, filters, joins)
        result, _ = build_result_declaration(pairs.values(), survivors, "c1")
        table = result.to_table().select(["c1", "e1", "c2", "e2"])
        assert table.to_pydict() == {"c1": [0], "e1": [0], "c2": [1], "e2": [1]}


def _query(tmp_path, backend="quail"):
    pq.write_table(pa.table({
        "id": ["r0", "r1"],
        "body": ["good film", "bad film"],
    }), tmp_path / "reviews.parquet")
    pq.write_table(pa.table({
        "id": ["a0", "a1"],
        "aspect": ["acting", "ending"],
    }), tmp_path / "aspects.parquet")
    def token_ids(text):
        return [byte + 1 for byte in text.encode("utf-8")]

    tokenizer = str.split if backend == "quail" else token_ids
    sess = quail.Session(
        EngineConfig(gpus=1, backend=backend), tokenizer=tokenizer
    )
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
    def execute(request):
        from quail.execution import PhysicalResponse, export_physical_outputs
        from quail.builtins import built_in_registry
        from quail.physical import (
            AdaptiveJoinPlan,
            PackedFilter,
            decode_graph,
        )
        from quail.runtime.runner import NodeMetrics, NodeResult, RunResult

        graph = decode_graph(request.plan["graph"], built_in_registry().codecs)
        filtered = next(
            node for node in graph.nodes if isinstance(node, PackedFilter)
        )
        adaptive = next(
            node for node in graph.nodes
            if isinstance(node, AdaptiveJoinPlan)
        )
        join = adaptive.join_specs[0]
        nodes = {
            filtered.node_id: NodeResult({
                "ids:r": [0],
                "filter_answers:r": {0: [1], 1: [0]},
            }),
            adaptive.node_id: NodeResult({
                "ids:r": [0] if any(join_answers) else [],
                "ids:a": [0, 1],
                "join_answers:0": {
                    "rows": {0: join_answers},
                    "anchor_index": [0],
                    "partner_index": [[0], [1]],
                    "anchor": "r",
                    "partners": ["a"],
                    "semantics": "full",
                    "selectivity": join["selectivity"],
                    "written_pos": 0,
                },
            }),
        }
        outputs = export_physical_outputs(
            graph, RunResult(None, nodes, NodeMetrics())
        )
        return PhysicalResponse(outputs, {
            "wall_s": 2.0,
            "boot_s": 3.0,
            "boot_kind": "cold",
            "boot": {},
            "fresh_tokens": 100,
            "store": None,
            "peak_gib": 1.0,
        })

    from quail.runtime.local import execute_worker_query

    return execute_worker_query(query, physical_executor=execute)


def _run_request_backend(query, join_answers):
    def execute(request):
        from quail.execution import PhysicalResponse, export_physical_outputs
        from quail.builtins import built_in_registry
        from quail.physical import RequestExecution, decode_graph
        from quail.runtime.result import answer_table
        from quail.runtime.runner import NodeMetrics, NodeResult, RunResult

        graph = decode_graph(request.plan["graph"], built_in_registry().codecs)
        model = next(
            node for node in graph.nodes
            if isinstance(node, RequestExecution)
        )
        filter_table = pa.table({
            "r": pa.array([0, 1], type=pa.int32()),
            "predicate": pa.array([0, 0], type=pa.int32()),
            "answer": pa.array([True, False], type=pa.bool_()),
        }).replace_schema_metadata({
            b"quail.kind": b"filter_answers",
            b"quail.alias": b"r",
        })
        join_table = answer_table(
            {"r": [0, 0], "a": [0, 1]},
            [bool(answer) for answer in join_answers],
            "join_answers",
            metadata={
                "anchor": "r",
                "partners": "a",
                "semantics": "full",
                "written_pos": 0,
            },
        )
        true_aspects = [
            index for index, answer in enumerate(join_answers) if answer
        ]
        nodes = {
            model.node_id: NodeResult({
                "ids:r": [0] if true_aspects else [],
                "ids:a": true_aspects,
                "filter_answers:r": filter_table,
                "join_answers:0": join_table,
            })
        }
        outputs = export_physical_outputs(
            graph, RunResult(None, nodes, NodeMetrics())
        )
        return PhysicalResponse(outputs, {
            "wall_s": 2.0,
            "boot_s": 3.0,
            "boot_kind": "cold",
            "boot": {},
            "fresh_tokens": 100,
            "cached_tokens": 0,
            "regret_tokens": 0,
            "peak_gib": 1.0,
        })

    from quail.runtime.local import execute_worker_query

    return execute_worker_query(query, physical_executor=execute)


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


def test_evaluator_scores_request_backend_answer_relations(tmp_path):
    sess, query = _query(tmp_path, backend="stock_vllm")
    try:
        result = _run_request_backend(query, [0, 0])
        corpus = {
            "reviews": [{"id": "r0", "body": "good film"},
                        {"id": "r1", "body": "bad film"}],
            "aspects": [{"id": "a0", "aspect": "acting"},
                        {"id": "a1", "aspect": "ending"}],
        }
        evaluation = BenchmarkEvaluator(_truth(), corpus).evaluate(
            query, result
        )
    finally:
        sess.close()

    assert evaluation["answer_accuracy"]["evaluated"] == 4
    assert evaluation["answer_accuracy"]["correct"] == 3
    assert [item["op"] for item in evaluation["per_predicate"]] == [
        "filter", "join"
    ]


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


def test_distinct_prefix_regret_adds_shared_prefixes_minus_cross_row_hits():
    from quail.bench.evaluate import (
        cross_row_cached_tokens,
        distinct_prefix_regret,
        scanned_aliases,
    )

    stages = [
        {"op": "filter", "alias": "r", "stage": 0},
        {"op": "join", "anchor": "r", "partners": ["a"]},
        {"op": "join", "anchor": "e", "partners": ["c"]},
    ]
    assert scanned_aliases(stages) == {"r", "e"}
    assert cross_row_cached_tokens({"backend": "quail"}) == 0
    assert cross_row_cached_tokens({
        "backend": "pipelined_vllm",
        "backend_metrics": {"cross_row_cached_tokens": 7},
    }) == 7
    assert cross_row_cached_tokens({"backend": "stock_vllm"}) is None
    assert distinct_prefix_regret(100, 50, 30) == 120
    assert distinct_prefix_regret(100, 50, None) is None


def test_scanned_shared_prefix_tokens_counts_repeated_columns_in_full():
    from quail.bench.evaluate import scanned_shared_prefix_tokens

    class Store:
        def __init__(self, documents):
            self.documents = documents

        def __iter__(self):
            return iter(self.documents)

    stores = {
        ("reviews", "text"): Store([[1, 2, 3], [1, 2, 4], [9]]),
        ("aspects", "name"): Store([[5, 5], [6]]),
    }

    def lookup(provider, column):
        return stores[(provider, column)]

    # within reviews two documents share [1, 2]; one alias of aspects
    # shares nothing
    assert scanned_shared_prefix_tokens(
        [("reviews", "text"), ("aspects", "name")], lookup) == 2
    # a second alias of reviews is the same trie again: its 7 tokens
    # are all shared with the first alias
    assert scanned_shared_prefix_tokens(
        [("reviews", "text"), ("reviews", "text")], lookup) == 2 + 7
