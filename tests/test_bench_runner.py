"""CPU checks for Quail's QUAIL-B runner: specs to queries, results to ids."""

import json
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quail
from quail.bench.quailb import answer_oracle, build_query, queries, run_output
from quail.catalog import DocumentProvider
from quail.planner import collect_operators
from quail.planner.plan import EngineConfig, Refusal
from quail_b.labels import GroundTruthCollection, PredicateLabels
from quail_b.queries import AliasSpec, JoinSpec, QuerySpec
from quail_b.queries import queries as query_specs
from quail_b.scoring import Evaluator

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


def fever_truth():
    """Labels for FEV-9 over a three claim, three evidence corpus."""
    from quail_b.predicates import PREDICATES, predicate_payload
    from quail_b.prompts import F11, F13, REFUTE, SUPPORT

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


def _session(tmp_path, backend="quail"):
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
    return sess


def _query(sess):
    # the same shape as SPEC, with the anchor pinned so the fake
    # executor below knows which alias the join keeps
    return (sess.docs("reviews").alias("r")
            .ai_filter(quail.prompt(FILTER, quail.col("r.body")))
            .ai_join(
                sess.docs("aspects").alias("a"),
                quail.prompt(JOIN, quail.col("r.body"),
                             quail.col("a.aspect")),
                anchor="r")
            .select("r.id", "a.id"))


def _run(query, join_answers):
    def execute(request):
        from quail.builtins import built_in_registry
        from quail.execution import PhysicalResponse, export_physical_outputs
        from quail.physical import (
            AnchoredJoin,
            DocumentInput,
            PackedFilter,
            decode_graph,
        )
        from quail.runtime.runner import NodeMetrics, NodeResult, RunResult

        graph = decode_graph(request.plan["graph"], built_in_registry().codecs)
        filtered = next(
            node for node in graph.nodes if isinstance(node, PackedFilter)
        )
        anchored = next(
            node for node in graph.nodes if isinstance(node, AnchoredJoin)
        )
        join = anchored.stages[0].runtime_spec()
        nodes = {
            **{node.node_id: NodeResult({f"ids:{node.alias}": range(
                len(request.inputs[node.input_id].documents))})
               for node in graph.nodes if isinstance(node, DocumentInput)},
            filtered.node_id: NodeResult({
                "ids:r": [0],
                "filter_answers:r": {0: [1], 1: [0]},
            }),
            anchored.node_id: NodeResult({
                "ids:r": [0] if any(join_answers) else [],
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
        from quail.builtins import built_in_registry
        from quail.execution import PhysicalResponse, export_physical_outputs
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


def test_run_output_translates_indices_to_ids_and_scores(tmp_path):
    sess = _session(tmp_path)
    try:
        result = _run(_query(sess), [1, 0])
        output = run_output(result, SPEC, CORPUS)
    finally:
        sess.close()

    assert output.filter_answers[("r", 0)].to_pydict() == {
        "r": ["r0", "r1"], "answer": [True, False]}
    assert output.join_answers[0].to_pydict() == {
        "r": ["r0", "r0"], "a": ["a0", "a1"], "answer": [True, False]}
    assert output.rows.to_pydict() == {"r": ["r0"], "a": ["a0"]}
    evaluation = Evaluator(_truth(), CORPUS).evaluate(SPEC, output)
    assert evaluation["answer_accuracy"]["accuracy"] == 1.0
    assert evaluation["output_accuracy"]["exact_match"] is True
    assert result.report["shared_prefix_tokens"] == 0
    assert result.report["cross_row_cached_tokens"] == 0
    assert result.report["regret_distinct_tokens"] is None


def test_request_backend_answer_relations_score_the_same_way(tmp_path):
    sess = _session(tmp_path, backend="stock_vllm")
    try:
        result = _run_request_backend(_query(sess), [0, 0])
        output = run_output(result, SPEC, CORPUS)
    finally:
        sess.close()

    evaluation = Evaluator(_truth(), CORPUS).evaluate(SPEC, output)
    assert evaluation["answer_accuracy"]["evaluated"] == 4
    assert evaluation["answer_accuracy"]["correct"] == 3
    assert [item["op"] for item in evaluation["per_predicate"]] == [
        "filter", "join"
    ]
    assert output.rows.num_rows == 0
    assert result.report["shared_prefix_tokens"] == 0
    # the fake stock vLLM run reports no cross row cache hits, so the
    # distinct prefix regret is unknown
    assert result.report["regret_distinct_tokens"] is None


def test_run_output_requires_id_columns_in_the_select_list(tmp_path):
    spec = QuerySpec(
        "TEST-2", "selects a text column",
        (AliasSpec("r", "reviews", "body", (FILTER,)),
         AliasSpec("a", "aspects", "aspect")),
        (JoinSpec(JOIN, ("r", "a")),),
        ("r.body", "a.id"),
    )
    sess = _session(tmp_path)
    try:
        result = _run(_query(sess), [1, 0])
        with pytest.raises(NotImplementedError):
            run_output(result, spec, CORPUS)
    finally:
        sess.close()


@pytest.mark.parametrize("backend", [
    "quail", "stock_vllm", "pipelined_vllm", "pipelined_sglang",
])
def test_fev9_builds_from_its_spec_and_answers_from_labels(backend):
    corpus, truth = fever_truth()
    spec = query_specs()["FEV-9"]
    evaluator = Evaluator(truth, corpus)
    answer = answer_oracle(evaluator)

    with quail.Session(EngineConfig(backend=backend), tokenizer=lambda text: list(
        text.encode("utf-8"))) as session:
        for name, table in corpus.items():
            session.register(name, DocumentProvider.from_table(table, id_col="id"))
        query = build_query(session, spec)
        assert not isinstance(query.plan(), Refusal)
        scans, filters, joins = collect_operators(query.logical)
        assert [scan.alias for scan in scans] == [
            alias.alias for alias in spec.aliases]
        assert {alias: [p.prompt.template for p in chain]
                for alias, chain in filters.items()} == {
            alias.alias: [
                quail.bind_prompt(template, (quail.ColumnRef(
                    alias.alias, alias.table, alias.column),)).template
                for template in alias.filters]
            for alias in spec.aliases}
        assert [tuple(arg.alias for arg in join.predicate.args)
                for join in joins] == [join.aliases for join in spec.joins]
        # c0 is about a person and e0 supports it; c2 is not about a person
        assert answer(filters["c1"][0].prompt, {"c1": 0}) is True
        assert answer(filters["c1"][0].prompt, {"c1": 2}) is False
        assert answer(joins[0].predicate, {"c1": 0, "e1": 0}) is True
        assert answer(joins[0].predicate, {"c1": 0, "e1": 1}) is False
        # the same labels drive the speed of light estimate
        estimate = quail.speed_of_light_estimate(query, answer)
        assert estimate.post_filter_counts == {
            "c1": 2, "e1": 2, "c2": 2, "e2": 2}
        assert len(estimate.join_stages) == 3
        assert queries(session)["FEV-9"][0] == spec.description


def test_benchmark_prompt_text_matches_what_quail_sends():
    """The labels answer quailb's text; Quail must send the same text."""
    from quail_b.predicates import PREDICATES
    from quail_b.rendering import render_filter_prompt, render_join_prompt

    for spec in PREDICATES:
        left = quail.ColumnRef("left", spec.left_table, spec.left_column)
        if spec.kind == "filter":
            prompt = quail.bind_prompt(spec.template, (left,))
            assert prompt.tail.startswith("{0}")
            expected = (prompt.preamble + "doc one"
                        + prompt.tail.replace("{0}", "", 1))
            assert render_filter_prompt(spec.template, "doc one") == expected
        else:
            right = quail.ColumnRef(
                "right", spec.right_table, spec.right_column)
            prompt = quail.bind_join_prompt(spec.template, (left, right))
            for anchor in (0, 1):
                expected = quail.render_join_prompt_text(
                    prompt, ("doc one", "doc two"), anchor=anchor)
                assert render_join_prompt(
                    spec.template, ("doc one", "doc two"), anchor=anchor
                ) == expected, (spec.key, anchor)


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
