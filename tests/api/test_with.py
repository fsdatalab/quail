"""WITH: common table expressions whose results later AI operators read."""

from pathlib import Path

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish, oracle_executor

import quail
from quail.api import execute as execution
from quail.api.query import Query
from quail.api.stages import ASSUMED_ANSWER_TOKENS, cte_table
from quail.catalog import DocumentProvider
from quail.logical import CompileError, cte_schema
from quail.physical import QueryStage, decode_graph, encode_graph
from quail.planner.plan import EngineConfig, PhysicalPlan

CONTRACTS = pa.table({
    "id": [1, 2, 3],
    "text": ["Lease ends on March 3. Tenant may terminate early with notice.",
             "No dates here. Nothing about ending.",
             "Lease ends in 2030. Early termination is forbidden."],
})
# the planted termination clause of each contract; the second has none
CLAUSES = {0: "Tenant may terminate early with notice", 1: None,
           2: "Early termination is forbidden"}

EXTRACT_THEN_SCORE = """
WITH clauses AS (
    SELECT c.id,
           AI.EXTRACT(c.text, {'terms': 'What is the termination clause?'}) AS clause
    FROM contracts c
)
SELECT x.id, x.clause,
       AI.SCORE(PROMPT('Can the tenant terminate early? {0}', x.clause.terms))
           AS score
FROM clauses x
"""


def words(text):
    return text.split()


@pytest.fixture()
def session():
    with quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
                       tokenizer=words) as value:
        value.register("contracts",
                       DocumentProvider.from_table(CONTRACTS, id_col="id"))
        yield value


class Recorder:
    """Run requests on an oracle and keep the documents each stage saw."""

    def __init__(self, oracle):
        self.execute = oracle_executor(oracle)
        self.requests = []
        self.documents = {}   # alias -> each document's tokens, per request

    def __call__(self, request):
        self.requests.append(request)
        for key, value in request.inputs.items():
            self.documents.setdefault(key, []).append(
                [list(document) for document in value.documents])
        return self.execute(request)


def run(query, oracle):
    recorder = Recorder(oracle)
    result = execution.execute_query(query, physical_executor=recorder)
    return result, recorder


def test_extraction_then_score_sends_the_extracted_text_to_the_scorer(session):
    query = session.sql(EXTRACT_THEN_SCORE)
    assert isinstance(query, Query)
    plan = query.plan()
    assert isinstance(plan, PhysicalPlan)
    assert all(isinstance(node, QueryStage) for node in plan.nodes)
    envelope = plan.to_envelope(session.registry.codecs)
    decoded = decode_graph(envelope["graph"], session.registry.codecs)
    assert encode_graph(decoded, session.registry.codecs) == envelope["graph"]
    oracle = Oracle(answers=lambda spec, document: CLAUSES[document],
                    scores=lambda spec, ids: 0.9 if ids[0] == 0 else 0.1)
    result, recorder = run(query, oracle)
    rows = result.collect()
    assert rows.column("x.id").to_pylist() == [1, 2, 3]
    assert rows.column("x.clause").to_pylist() == [
        {"terms": clause} for clause in CLAUSES.values()]
    assert rows.column("score").to_pylist() == [0.9, 0.1, 0.1]
    # the scorer read the extracted clauses, not the contracts; a null
    # answer reaches it as empty text
    (scored,) = recorder.documents["x"]
    assert scored == [words(CLAUSES[0]), [], words(CLAUSES[2])]
    assert len(recorder.requests) == 2
    # every stage counts in the totals; each CTE keeps its own report
    assert result.report["wall_s"] == 2.0
    assert result.report["fresh_tokens"] == 2 * 1234
    assert result.report["ctes"]["clauses"]["wall_s"] == 1.0
    assert result.report["ctes"]["clauses"]["node_metrics"]["ai-extract:0"][
        "evaluated_documents"] == 3
    assert result.plan == query.plan().graph
    assert "AiExtract" in result.explain()
    assert "quail.ai_score" in result.explain()


def test_filters_over_a_materialized_label_share_one_pipeline(session):
    query = session.sql("""
        WITH kinds AS (
            SELECT c.id, c.text,
                   AI.CLASSIFY(PROMPT('What kind of document is {0}?', c.text),
                               ARRAY['lease', 'deed']) AS kind
            FROM contracts c
        ),
        leases AS (
            SELECT k.id, k.kind FROM kinds k
            WHERE AI.IF(PROMPT('is a lease: {0}', k.kind))
        )
        SELECT l.id AS result_id FROM leases l
        WHERE AI.IF(PROMPT('kept: {0}', l.kind))
    """, dialect="bq", order="as_written")
    assert [name for name, _ in query.logical.stages() if name] == ["kinds"]
    # equal labels get equal answers: the executor asks each text once
    oracle = Oracle(
        labels=["lease", "deed", "deed"],
        filters=lambda alias, document, position: [True, False, False][document])
    result, recorder = run(query, oracle)
    assert result.collect().column("result_id").to_pylist() == [1]
    assert [list(request.inputs) for request in recorder.requests] == [
        ["c"], ["k"]]
    # the label column is text in the CTE's table, and the filter read it
    assert recorder.documents["k"] == [[["lease"], ["deed"], ["deed"]]]
    from quail.physical.pipelines import build_pipelines

    graph = query.plan().nodes[-1].graph
    assert any(len(pipeline) == 2 for pipeline in build_pipelines(graph).values())


def test_a_cte_runs_once_however_many_times_it_is_read(session):
    query = session.sql("""
        WITH clauses AS (
            SELECT c.id,
                   AI.EXTRACT(c.text, {'terms': 'What is the termination clause?'})
                       AS clause
            FROM contracts c
        ),
        unused AS (
            SELECT c.id FROM contracts c
            WHERE AI.IF(PROMPT('never asked: {0}', c.text))
        )
        SELECT a.id, b.id AS other
        FROM clauses a JOIN clauses b
          ON AI.IF(PROMPT('same clause? {0} {1}', a.clause.terms, b.clause.terms))
    """, dialect="bq")
    # a CTE nothing reads is left out
    assert [name for name, _ in query.logical.stages() if name] == ["clauses"]
    oracle = Oracle(answers=lambda spec, document: CLAUSES[document],
                    joins=lambda node, assignment: (
                        assignment["a"] == assignment["b"]))
    result, recorder = run(query, oracle)
    assert len(recorder.requests) == 2
    rows = result.collect()
    assert rows.column("a.id").to_pylist() == [1, 2, 3]
    assert rows.column("other").to_pylist() == [1, 2, 3]
    # both aliases read the one result table
    assert recorder.documents["a"] == recorder.documents["b"]


def test_a_cte_no_model_output_feeds_merges_into_one_plan(session):
    # a regular CTE, a FROM subquery without an alias, and a CTE whose
    # computed column is only compared become one plan, as in DuckDB
    merged = session.sql("""
        WITH recent AS (SELECT c.id, c.text FROM contracts c
                        WHERE c.id > 1 AND AI.IF(PROMPT('lease? {0}', c.text))),
        kinds AS (
            SELECT r.id, r.text,
                   AI.CLASSIFY(PROMPT('kind of {0}?', r.text), ARRAY['lease', 'deed'])
                       AS kind
            FROM recent r
        )
        SELECT id FROM (SELECT k.id, k.text FROM kinds k WHERE k.kind = 'lease')
        WHERE AI.IF(PROMPT('keep: {0}', text))
        ORDER BY id LIMIT 5
    """, dialect="bq", order="as_written")
    assert isinstance(merged, Query)
    operators = merged.logical.operators()
    (scan,) = operators.scans
    assert scan.provider == "contracts" and str(scan.predicates[0]) == "c.id > 1"
    assert len(operators.classifies) == 1
    # the CTE's condition stays ahead of the outer one, as written
    assert [predicate.prompt.template.split("\n")[-1]
            for predicate in operators.filters["c"]] == ["lease?", "keep:"]
    oracle = Oracle(labels=["lease", "deed", "lease"],
                    filters=lambda alias, document, position: True)
    result, recorder = run(merged, oracle)
    assert len(recorder.requests) == 1
    assert result.collect().column("c.id").to_pylist() == [3]


def test_an_empty_cte_result_gives_an_empty_result(session):
    query = session.sql("""
        WITH kept AS (
            SELECT c.id,
                   AI.CLASSIFY(PROMPT('kind of {0}?', c.text), ARRAY['lease', 'deed'])
                       AS kind
            FROM contracts c
            WHERE AI.IF(PROMPT('keep: {0}', c.text))
        )
        SELECT k.id, AI.SCORE(PROMPT('score {0}', k.kind)) AS score FROM kept k
    """, dialect="bq")
    oracle = Oracle(filters=lambda alias, document, position: False)
    result, recorder = run(query, oracle)
    rows = result.collect()
    assert rows.num_rows == 0
    assert rows.column_names == ["k.id", "score"]
    assert rows.schema.field("score").type == pa.float64()


def test_explain_plans_later_stages_on_named_assumptions(session, monkeypatch):
    query = session.sql(EXTRACT_THEN_SCORE)
    planned = query.explain()
    assert planned.index("QueryStage clauses:") < planned.index("QueryStage query:")
    assert ("'x' reads the CTE 'clauses', which has not run; planned on 3 rows "
            f"of {ASSUMED_ANSWER_TOKENS} tokens each") in planned
    assert "run:" not in planned
    plan = query.plan()
    assert plan.estimated_seconds == sum(
        stage.estimated_seconds for stage in plan.nodes)

    oracle = Oracle(answers=lambda spec, document: CLAUSES[document],
                    scores=lambda spec, ids: 0.5)
    executor = oracle_executor(oracle)
    original = execution.execute_query
    monkeypatch.setattr(
        "quail.api.query.execute_query",
        lambda q, physical_executor=executor, **options: original(
            q, physical_executor=physical_executor, **options))
    measured = query.explain(analyze=True)
    assert measured.count("run:") == 1
    assert "which has not run" not in measured
    # after the run, the second stage was planned on the real table
    assert "tokens=10, mean_document_tokens=3.3" in measured


def test_cte_results_have_the_schema_the_next_stage_was_compiled_for(session):
    # every kind of result column: a passed-through column, a score, a
    # label with probabilities, an extraction, an aggregate, and a window
    queries = {
        "score": "SELECT c.id, AI.SCORE(PROMPT('s {0}', c.text)) AS s "
                 "FROM contracts c",
        "label": "SELECT c.id, AI.CLASSIFY(PROMPT('k {0}', c.text), "
                 "ARRAY['lease', 'deed'], {'probabilities': true}) AS kind "
                 "FROM contracts c",
        "extract": "SELECT c.id, LENGTH(c.text) AS n, AI.EXTRACT(c.text, "
                   "{'terms': 'q?', 'when': 'w?'}) AS clause FROM contracts c",
        "aggregate": "SELECT AI.CLASSIFY(PROMPT('k {0}', c.text), "
                     "ARRAY['lease', 'deed']) AS kind, COUNT(*) AS n, "
                     "AVG(c.id) AS mean_id, MAX(c.id) AS top, SUM(c.id) AS total "
                     "FROM contracts c GROUP BY kind",
        "window": "SELECT c.id, AI.SCORE(PROMPT('s {0}', c.text)) AS s, "
                  "ROW_NUMBER() OVER (ORDER BY c.id) AS rank, "
                  "LAG(c.id) OVER (ORDER BY c.id) AS before, "
                  "SUM(c.id) OVER (ORDER BY c.id) AS running, "
                  "COUNT(c.id) OVER (ORDER BY c.id) AS seen "
                  "FROM contracts c",
    }
    oracle = Oracle(answers=lambda spec, document: CLAUSES[document],
                    labels=["lease", "deed", "lease"], probabilities=True)
    for name, text in queries.items():
        query = session.sql(text)
        schema = cte_schema(query.logical)
        result = execution.execute_query(query, physical_executor=oracle_executor(
            oracle))
        table = cte_table(result.collect(), schema)
        assert table.schema == schema, name
    kinds = cte_table(result.collect(), schema)
    with pytest.raises(ValueError, match="compiled for"):
        cte_table(kinds, cte_schema(session.sql(queries["score"]).logical))


def test_with_rejects_what_the_stages_cannot_express(session):
    extract = ("SELECT c.id, AI.EXTRACT(c.text, {'terms': 'q?'}) AS clause "
               "FROM contracts c")
    read = "SELECT x.id FROM clauses x WHERE AI_FILTER(PROMPT('k {0}', x.clause.terms))"
    for sql, message in (
        (f"WITH RECURSIVE clauses AS ({extract}) {read}", "RECURSIVE"),
        (f"WITH contracts AS ({extract}) SELECT x.id FROM contracts x "
         f"WHERE AI_FILTER(PROMPT('k {{0}}', x.clause.terms))", "another name"),
        ("WITH clauses AS (SELECT LENGTH(c.text), AI.EXTRACT(c.text, "
         "{'terms': 'q?'}) AS clause FROM contracts c) SELECT x.clause "
         "FROM clauses x WHERE AI_FILTER(PROMPT('k {0}', x.clause.terms))",
         "AS name"),
        (f"WITH clauses AS (SELECT c.id, d.id, AI.EXTRACT(c.text, {{'terms': 'q?'}}) "
         f"AS clause FROM contracts c JOIN contracts d "
         f"ON AI_FILTER(PROMPT('k {{0}} {{1}}', c.text, d.text))) {read}",
         "name twice"),
        (f"WITH later AS (SELECT x.id FROM clauses x WHERE "
         f"AI_FILTER(PROMPT('k {{0}}', x.clause.terms))), clauses AS ({extract}) "
         f"SELECT l.id FROM later l WHERE AI_FILTER(PROMPT('k {{0}}', l.id))",
         "clauses"),
    ):
        with pytest.raises(CompileError, match=message):
            session.sql(sql)


def test_a_session_closes_the_token_files_of_its_cte_results():
    session = quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
                            tokenizer=words)
    session.register("contracts", DocumentProvider.from_table(CONTRACTS, id_col="id"))
    query = session.sql(EXTRACT_THEN_SCORE)
    run(query, Oracle(answers=lambda spec, document: CLAUSES[document]))
    directory = Path(session.inputs._token_directory.name)
    assert list(directory.iterdir())
    assert "clauses" not in session.catalog
    session.close()
    assert not directory.exists()


@pytest.mark.parametrize("sql", [
    """WITH first AS (SELECT c.id, c.text FROM contracts c ORDER BY c.id LIMIT 2)
       SELECT f.id AS result_id FROM first f WHERE AI.IF(PROMPT('keep {0}', f.text))""",
    """WITH first AS (
           SELECT c.id FROM contracts c WHERE AI.IF(PROMPT('keep {0}', c.text))
           ORDER BY c.id LIMIT 2)
       SELECT f.id AS result_id FROM first f WHERE f.id > 0""",
    """WITH first AS (SELECT c.id, c.text FROM contracts c WHERE c.id < 3)
       SELECT a.id AS result_id FROM first a JOIN first b
       ON AI.IF(PROMPT('same? {0} {1}', a.text, b.text))""",
    """WITH first AS (SELECT c.id, c.text FROM contracts c WHERE c.id < 3),
       unused AS (SELECT u.id FROM first u)
       SELECT f.id AS result_id FROM first f WHERE AI.IF(PROMPT('keep {0}', f.text))""",
])
def test_relational_stages_before_and_after_ai_return_the_selected_rows(session, sql):
    result, recorder = run(session.sql(sql, dialect="bq"),
                           Oracle(filters=lambda *args: True,
                                  joins=lambda node, assignment:
                                  len(set(assignment.values())) == 1))
    assert result.collect().column("result_id").to_pylist() == [1, 2]
    assert "Project" in result.explain()
    assert len(recorder.requests) == 1


def test_a_dictionary_source_column_keeps_its_type_across_materialization(session):
    category = pa.array(["lease", "deed", "lease"]).dictionary_encode()
    session.register("categorized", DocumentProvider.from_table(
        CONTRACTS.append_column("category", category), id_col="id"))
    query = session.sql("""
        WITH kinds AS (
            SELECT c.id, c.category,
                   AI.CLASSIFY(PROMPT('kind {0}', c.text), ARRAY['lease', 'deed'])
                       AS kind
            FROM categorized c)
        SELECT k.category FROM kinds k WHERE AI.IF(PROMPT('keep {0}', k.kind))
    """, dialect="bq")
    result, _ = run(query, Oracle(labels=["lease", "deed", "lease"],
                                 filters=lambda *args: True))
    assert result.collect().column("k.category").combine_chunks() == category


def test_intermediate_files_are_released_after_success_and_failure(
        session, monkeypatch, tmp_path):
    from functools import partial
    from tempfile import TemporaryDirectory

    monkeypatch.setattr("quail.catalog.inputs.TemporaryDirectory",
                        partial(TemporaryDirectory, dir=tmp_path))
    query = session.sql(EXTRACT_THEN_SCORE)
    query.plan()
    source_directory = Path(session.inputs._token_directory.name)

    def fail_score(spec, ids):
        raise RuntimeError("score failed")

    with pytest.raises(RuntimeError, match="score failed"):
        run(query, Oracle(answers=lambda spec, document: CLAUSES[document],
                          scores=fail_score))
    assert list(tmp_path.iterdir()) == [source_directory]
    result, _ = run(query, Oracle(answers=lambda spec, document: CLAUSES[document]))
    assert result.count() == 3
    assert list(tmp_path.iterdir()) == [source_directory]


def test_repeated_execution_reads_the_newly_generated_text(session):
    query = session.sql(EXTRACT_THEN_SCORE)
    for clause in (CLAUSES[0], CLAUSES[2]):
        result, recorder = run(query, Oracle(answers=lambda *args: clause))
        assert result.count() == 3
        assert recorder.documents["x"] == [[words(clause)] * 3]


def test_stage_totals_include_cost_gpu_seconds_and_document_work(session):
    result = finish(session.sql(EXTRACT_THEN_SCORE), Oracle(
        answers=lambda spec, document: CLAUSES[document]),
        usd_per_query=0.1, gpu_s=2, evaluated_documents=3,
        evaluated_document_pairs=4)
    assert result.report["usd_per_query"] == pytest.approx(0.2)
    assert result.report["gpu_s"] == 4
    assert result.report["evaluated_documents"] == 6
    assert result.report["evaluated_document_pairs"] == 8


def test_a_filtered_cte_on_the_right_of_a_join_keeps_its_filter(session):
    query = session.sql("""
        WITH kept AS (
            SELECT c.id, c.text FROM contracts c
            WHERE AI.IF(PROMPT('keep {0}', c.text)))
        SELECT a.id, b.id AS other FROM contracts a JOIN kept b
        ON AI.IF(PROMPT('same? {0} {1}', a.text, b.text))
    """, dialect="bq")
    result, recorder = run(query, Oracle(
        filters=lambda alias, document, position: document != 1,
        joins=lambda node, assignment: len(set(assignment.values())) == 1))
    assert result.collect().to_pydict() == {"a.id": [1, 3], "other": [1, 3]}
    assert len(recorder.requests) == 1


def test_snowflake_qualification_preserves_an_implicit_table_alias(session):
    query = session.sql("""
        WITH kept AS (
            SELECT contracts.id, contracts.text FROM contracts
            WHERE AI_FILTER(PROMPT('keep {0}', contracts.text)))
        SELECT kept.id FROM kept
    """)
    result, _ = run(query, Oracle(filters=lambda *args: True))
    assert result.collect().to_pydict() == {"contracts.id": [1, 2, 3]}


def test_an_exists_prompt_reads_a_materialized_computed_column(session):
    query = session.sql("""
        WITH kinds AS (
            SELECT c.id,
                   AI.CLASSIFY(PROMPT('kind {0}', c.text), ARRAY['lease', 'deed'])
                       AS kind FROM contracts c)
        SELECT a.id FROM contracts a WHERE EXISTS (
            SELECT 1 FROM kinds k
            WHERE AI.IF(PROMPT('matches? {0} {1}', a.text, k.kind)))
    """, dialect="bq")
    result, recorder = run(query, Oracle(
        labels=["lease", "deed", "lease"], joins=lambda *args: True))
    assert result.collect().column("a.id").to_pylist() == [1, 2, 3]
    assert recorder.documents["k"] == [[["lease"], ["deed"], ["lease"]]]


def test_a_scalar_expression_over_a_label_reads_the_materialized_labels(session):
    query = session.sql("""
        WITH kinds AS (
            SELECT c.id,
                   AI.CLASSIFY(PROMPT('kind {0}', c.text), ARRAY['lease', 'deed'])
                       AS kind FROM contracts c),
        capitalized AS (SELECT k.id, UPPER(k.kind) AS upper_kind FROM kinds k)
        SELECT u.id AS result_id FROM capitalized u
        WHERE AI.IF(PROMPT('keep {0}', u.upper_kind))
    """, dialect="bq")
    result, recorder = run(query, Oracle(
        labels=["lease", "deed", "lease"], filters=lambda *args: True))
    assert result.collect().column("result_id").to_pylist() == [1, 2, 3]
    assert recorder.documents["k"] == [[["LEASE"], ["DEED"], ["LEASE"]]]


def test_an_unsupported_deep_field_path_raises_a_compile_error(session):
    session.register("nested", DocumentProvider.from_table(pa.table({
        "id": [1], "details": [{"a": {"b": {"text": "lease"}}}]}), id_col="id"))
    with pytest.raises(CompileError, match="field path"):
        session.sql("""
            WITH kept AS (SELECT n.id, n.details FROM nested n)
            SELECT k.id FROM kept k
            WHERE AI_FILTER(PROMPT('keep {0}', k.details.a.b.text))
        """)
