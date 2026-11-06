"""CPU checks for Quail's QUAIL-B runner: plans to queries, results to ids."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers.logical import nodes, read_plan
from helpers.oracle import Oracle, oracle_executor, question_text
from helpers.sessions import TOKENIZER, encoder

import quail
from bench.quailb import (
    build_query,
    data_directory,
    join_anchors,
    prompt_pieces,
    register_tables,
    run_output,
)
from bench.substrait import sql_string
from quail.api.config import EngineConfig
from quail.api.execute import execute_query
from quail.catalog import DocumentProvider
from quail.execution.metrics import NodeMetrics, NodeResult, RunResult
from quail.execution.result import answer_table
from quail.execution.types import PhysicalResponse, export_physical_outputs
from quail.physical import RequestExecution
from quail.physical.plan import Refusal, graph_from_envelope
from quail_b.labels import GroundTruthCollection, PredicateLabels
from quail_b.minimum import DocumentTokens, token_metrics
from quail_b.prompts import DISCUSS_ASPECT, F1
from quail_b.queries import get_query
from quail_b.scoring import evaluate

# IMDB-3 is F1 on the reviews, then the DISCUSS_ASPECT join with the aspects
SPEC = get_query("IMDB-3")
CORPUS = {
    "reviews": pa.table({"id": ["r0", "r1"], "body": ["good film", "bad film"]}),
    "aspects": pa.table({"id": ["a0", "a1"], "aspect": ["acting", "ending"]}),
}


def _labels(key, template, answers, right_table=None):
    predicate = {
        "key": key, "template": template,
        "kind": "join" if right_table else "filter",
        "left_table": "reviews", "left_column": "body",
        "right_table": right_table, "right_column": right_table and "aspect",
    }
    return PredicateLabels(key, f"ls_{key}", predicate, answers,
                           {"qwen3-32b-fp8": len(answers)})


def _truth():
    predicates = (
        _labels("test.review.filter", F1, {("r0", None): True, ("r1", None): False}),
        _labels("test.review.aspect", DISCUSS_ASPECT, {
            ("r0", "a0"): True, ("r0", "a1"): False,
            ("r1", "a0"): False, ("r1", "a1"): True,
        }, "aspects"),
    )
    return GroundTruthCollection("gt_test", "c_test", 0.1, "qwen3-32b-fp8",
                                 {labels.key: labels for labels in predicates})


def _token_ids(text):
    return encoder()(text)


def _config(backend):
    return EngineConfig(gpus=1, model="qwen3-4b-fp8", backend=backend,
                        device="h100-sxm")


def _session(tmp_path, backend="quail"):
    for name, table in CORPUS.items():
        pq.write_table(table, tmp_path / f"{name}.parquet")
    sess = quail.Session(_config(backend), tokenizer=TOKENIZER)
    for name in CORPUS:
        sess.register(name, DocumentProvider.from_parquet(
            str(tmp_path / f"{name}.parquet"), id_col="id"))
    return sess


def _query(sess):
    # the same shape as SPEC, with the anchor pinned so the fake
    # executor below knows which alias the join keeps
    return sess.sql(f"""
        SELECT r.id, a.id FROM reviews r
        JOIN aspects a
          ON AI_FILTER(PROMPT({sql_string(DISCUSS_ASPECT)}, r.body, a.aspect),
                       {{'anchor': 'r'}})
        WHERE AI_FILTER(PROMPT({sql_string(F1)}, r.body))
    """)


_REPORT = {"wall_s": 2.0, "boot_s": 3.0, "boot_kind": "cold", "boot": {},
           "fresh_tokens": 2000, "peak_gib": 1.0}


def _execute(query, nodes, **metrics):
    """Run a query on a request backend whose node results nodes() plants."""
    def execute(request):
        graph = graph_from_envelope(request.plan)
        outputs = export_physical_outputs(
            graph, RunResult(None, nodes(graph, request), NodeMetrics()))
        return PhysicalResponse(outputs, {**_REPORT, **metrics})

    return execute_query(query, physical_executor=execute)


def _run(query, join_answers):
    """Run the review query: r0 passes the filter; join_answers[a] is pair (r0, a)."""
    return execute_query(query, physical_executor=oracle_executor(Oracle(
        filters=lambda alias, document, position: document == 0,
        joins=lambda node, assignment: join_answers[assignment["a"]]),
        **_REPORT, store=None))


def _run_request_backend(query, join_answers):
    def nodes(graph, request):
        model = next(
            node for node in graph.nodes if isinstance(node, RequestExecution))
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
            metadata={"anchor": "r", "partners": "a", "semantics": "full",
                      "written_position": 0},
        )
        true_aspects = [
            index for index, answer in enumerate(join_answers) if answer
        ]
        return {model.node_id: NodeResult({
            "rows:r": [0] if true_aspects else [],
            "rows:a": true_aspects,
            "filter_answers:r": filter_table,
            "join_answers:0": join_table,
        })}

    return _execute(query, nodes, cached_tokens=0)


TRACES = pa.table({
    "id": [f"s{i}" for i in range(6)],
    "trace": [f"trace {i} words" for i in range(6)],
    "trajectory_id": ["A", "A", "B", "B", "B", "C"],
    "turn_index": [5, 10, 10, 15, 20, 5],
    "token_count": [2000, 4000, 5000, 7000, 9000, 1000],
})


def test_relational_plans_build_and_run(tmp_path):
    pq.write_table(TRACES, tmp_path / "agent_traces.parquet")
    with quail.Session(_config("quail"), tokenizer=TOKENIZER) as sess:
        sess.register("agent_traces", DocumentProvider.from_parquet(
            str(tmp_path / "agent_traces.parquet"), id_col="id"))
        # DataFusion groups, filters, sorts, and limits the AI filter's rows
        grouped = build_query(sess, get_query("REL-AGENT-6"))
        assert not grouped.whole
        explained = grouped.explain()
        assert "Aggregate: groupBy=[[t.trajectory_id]]" in explained
        assert "Filter: count(Int64(1)) >= Int64(2)" in explained
        assert "fetch=50" in explained
        assert nodes(grouped.logical, "Scan")
        # the plain tests run in DataFusion, before the model reads a row
        tested = build_query(sess, get_query("REL-AGENT-1"))
        (scan,) = nodes(tested.logical, "Scan")
        assert scan.predicates == ()
        explained = tested.explain()
        assert "turn_index >= Int64(10)" in explained
        assert "token_count <= Int64(6000)" in explained
        paged = read_plan(build_query(sess, get_query("REL-AGENT-2")).logical).root
        assert (paged.limit, paged.offset) == (10, 10)
        top = read_plan(build_query(sess, get_query("REL-AGENT-4")).logical).root
        assert [column.name for column in top.input.columns
                if hasattr(column, "name")] == ["recovered_score"]
        assert top.order[0].name == "recovered_score" and top.limit == 20
        # an aggregate with no measures is a DISTINCT, which the filter
        # stops per trajectory
        query = build_query(sess, get_query("REL-AGENT-3"))
        distinct = read_plan(query.logical).root
        assert distinct.distinct and [
            column.column for column in distinct.input.columns
        ] == ["trajectory_id"]
        assert "stop per key: trajectory_id" in query.explain()

        # the fixes: s0, s1 (A), s2, s4 (B), s5 (C): A and B twice
        oracle = Oracle(filters=lambda alias, document, position: document != 3)
        query = build_query(sess, get_query("REL-AGENT-6"))
        result = execute_query(query, physical_executor=oracle_executor(
            oracle, **_REPORT, store=None))
        output = run_output(result, get_query("REL-AGENT-6").info,
                            {"agent_traces": TRACES})
    assert output.rows.to_pydict() == {
        "trajectory_id": ["A", "B"], "fixes": [2, 2], "first_fix": [5, 10],
        "longest": [4000, 9000]}
    assert output.filter_answers["filter-1"].column("t").to_pylist() == [
        "s0", "s1", "s2", "s3", "s4", "s5"]


CONTRACTS = pa.table({
    "id": [f"ct{i}" for i in range(3)],
    "title": ["A", "B", "C"],
    "text": ["audit yearly", "no audit", "audit at will"],
    "token_count": pa.array([2, 2, 3], pa.int32()),
})


@pytest.mark.parametrize("query_id", [
    "IMDB-22", "IMDB-20", "IMDB-21", "IMDB-24", "IMDB-25", "CUAD-19", "CUAD-20"])
def test_a_map_query_reports_every_answer_and_its_pieces(tmp_path, query_id):
    spec = get_query(query_id)
    corpus = {"reviews": CORPUS["reviews"], "contracts": CONTRACTS}
    oracle = Oracle(filters=lambda alias, document, position: document != 1,
                    labels=lambda spec, ids: "negative")
    with _session(tmp_path) as sess:
        sess.register("contracts", DocumentProvider.from_table(
            CONTRACTS, id_col="id"))
        query = build_query(sess, spec)
        result = execute_query(query, physical_executor=oracle_executor(
            oracle, **_REPORT, store=None))
        output = run_output(result, spec.info, corpus)
        pieces = prompt_pieces(query, spec.info, {})
        turn = sess.model.turn
    (relation,) = spec.info.relations
    alias = relation.alias
    filtered = bool(spec.info.filters)
    read = (["r0"] if relation.table == "reviews" else ["ct0", "ct2"]) \
        if filtered else corpus[relation.table].column("id").to_pylist()
    if query_id == "IMDB-25":
        read = []   # every review is labeled negative; only mixed ones are mapped
    assert sorted(output.map_answers) == [m.id for m in spec.info.maps]
    for operator in spec.info.maps:
        answers = output.map_answers[operator.id].to_pydict()
        assert answers[alias] == read
        index = {"r0": 0, "r1": 1, "ct0": 0, "ct1": 1, "ct2": 2}
        assert answers["answer"] == [f"{operator.output} {index[i]}" for i in read]
        assert output.rows.column(operator.output).to_pylist() == answers["answer"]
    # every map shows the document under the shared preamble, so a
    # filter or classification before it in the pipeline leaves its KV
    assert pieces["preamble"] == _token_ids(turn[0] + quail.SHARED_PRE)
    assert [piece["id"] for piece in pieces["maps"]] == [
        m.id for m in spec.info.maps]
    assert all(question_text(piece["tail"]).endswith("ANSWER:")
               for piece in pieces["maps"])


def test_stock_vllm_reports_an_extraction_and_its_pieces(tmp_path):
    spec = get_query("CUAD-1")
    span_type = pa.struct([("start", pa.int64()), ("end", pa.int64())])

    def nodes(graph, request):
        model = next(
            node for node in graph.nodes if isinstance(node, RequestExecution))
        return {model.node_id: NodeResult({
            "rows:c": [0, 1, 2],
            "extract_answers:contract.governing_law": pa.table({
                "c": pa.array([0, 1, 2], pa.int32()),
                "contract.governing_law": pa.array(
                    ["audit yearly", None, "will"], pa.string()),
                "contract.governing_law_span": pa.array(
                    [{"start": 0, "end": 12}, None, {"start": 9, "end": 13}],
                    span_type),
            }),
        })}

    with _session(tmp_path, backend="stock_vllm") as sess:
        sess.register("contracts", DocumentProvider.from_table(
            CONTRACTS, id_col="id"))
        query = build_query(sess, spec)
        result = _execute(query, nodes, cached_tokens=0)
        output = run_output(result, spec.info, {"contracts": CONTRACTS})
        pieces = prompt_pieces(query, spec.info, {})
        turn = sess.model.turn
    assert output.extract_answers["extract-1"].to_pydict() == {
        "c": ["ct0", "ct1", "ct2"], "answer": ["audit yearly", None, "will"]}
    assert output.rows.column_names == ["c", "contract"]
    # the request node carries the extraction's own head and tail
    assert pieces["preamble"] == _token_ids(turn[0] + quail.SHARED_PRE)
    assert [item["id"] for item in pieces["extracts"]] == ["extract-1"]
    assert pieces["extracts"][0]["tail"][-1] == _token_ids(":")[-1]


def test_stock_vllm_reports_maps_and_their_pieces(tmp_path):
    spec = get_query("CUAD-18")

    def nodes(graph, request):
        model = next(
            node for node in graph.nodes if isinstance(node, RequestExecution))
        return {model.node_id: NodeResult({
            "rows:c": [0, 1, 2],
            **{f"map_answers:{operator.output}": pa.table({
                "c": pa.array([0, 1, 2], pa.int32()),
                operator.output: pa.array(
                    [f"{operator.output} {i}" for i in range(3)], pa.string()),
            }) for operator in spec.info.maps},
        })}

    with _session(tmp_path, backend="stock_vllm") as sess:
        sess.register("contracts", DocumentProvider.from_table(
            CONTRACTS, id_col="id"))
        query = build_query(sess, spec)
        result = _execute(query, nodes, cached_tokens=0)
        output = run_output(result, spec.info, {"contracts": CONTRACTS})
        pieces = prompt_pieces(query, spec.info, {})
    for operator in spec.info.maps:
        assert output.map_answers[operator.id].to_pydict() == {
            "c": ["ct0", "ct1", "ct2"],
            "answer": [f"{operator.output} {i}" for i in range(3)]}
    assert [item["id"] for item in pieces["maps"]] == [
        operator.id for operator in spec.info.maps]
    assert pieces["maps"][0]["tail"][-1] == _token_ids(":")[-1]


def test_benchmark_results_and_scoring(tmp_path):
    with _session(tmp_path) as sess:
        quail_result = _run(_query(sess), [1, 0])
    output = run_output(quail_result, SPEC.info, CORPUS)

    assert output.filter_answers["filter-1"].to_pydict() == {
        "r": ["r0", "r1"], "answer": [True, False]}
    assert output.join_answers["join-1"].to_pydict() == {
        "r": ["r0", "r0"], "a": ["a0", "a1"], "answer": [True, False]}
    assert output.rows.to_pydict() == {"r": ["r0"], "a": ["a0"]}
    evaluation = evaluate(SPEC, output, _truth(), CORPUS)
    assert evaluation["answer_accuracy"]["accuracy"] == 1.0
    assert evaluation["output_accuracy"]["exact_match"] is True

    with _session(tmp_path, backend="stock_vllm") as sess:
        query = _query(sess)
        result = _run_request_backend(query, [0, 0])
        output = run_output(result, SPEC.info, CORPUS)
        output.prompt_pieces = prompt_pieces(query, SPEC.info,
                                             join_anchors(result))
        tokenizer_name = sess.model.hf_name

    evaluation = evaluate(SPEC, output, _truth(), CORPUS)
    assert evaluation["answer_accuracy"]["evaluated"] == 4
    assert evaluation["answer_accuracy"]["correct"] == 3
    assert [item["op"] for item in evaluation["per_predicate"]] == ["filter", "join"]
    assert output.rows.num_rows == 0
    pieces = output.prompt_pieces
    assert pieces["tokenizer"] == tokenizer_name
    # the model's user turn opens ahead of the document preamble
    assert pieces["preamble"] == _token_ids(
        sess.model.turn_prefix + quail.SHARED_PRE)
    assert [item["id"] for item in pieces["filters"]] == ["filter-1"]
    assert [(item["id"], item["anchor"]) for item in pieces["joins"]] == [
        ("join-1", "r")]
    stores = {tokenizer_name: DocumentTokens(
        CORPUS, lambda texts: [_token_ids(text) for text in texts])}
    measured = token_metrics(SPEC, output, CORPUS, stores)
    assert measured["minimum_tokens"] > 0
    assert measured["regret_tokens"] == 2000 - measured["minimum_tokens"]


def test_sec_queries_read_their_own_directory_as_engines_see_the_filings(tmp_path):
    assert data_directory(tmp_path, 1.0, ["SEC-9", "SEC-10"]) == tmp_path / "sec"
    assert data_directory(tmp_path, 0.1, ["CUAD-16"]) == tmp_path / "sf0.1"
    # a composed accent is one character once the text is NFC
    pq.write_table(pa.table({
        "id": ["0001"], "text": ["cafe\u0301 agreement"],
        "filing_date": ["2024-01-02"], "filing_month": ["2024-01"]}),
        tmp_path / "sec_filings.parquet")
    with quail.Session(_config("quail"), tokenizer=TOKENIZER) as sess:
        register_tables(sess, tmp_path)
        query = sess.sql("SELECT f.id FROM sec_filings f WHERE f.text_chars = 14 "
                         "AND AI_FILTER(PROMPT('Is {0} a contract?', f.text))")
        assert not isinstance(query.plan(), Refusal)
