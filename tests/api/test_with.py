"""WITH: common table expressions whose results later AI operators read."""

from pathlib import Path

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, oracle_executor

import quail
from quail.api import execute as execution
from quail.api.query import ASSUMED_ANSWER_TOKENS, WithQuery
from quail.catalog import Catalog, DocumentProvider
from quail.execution.schema import cte_table
from quail.frontend.sql import compile_sql
from quail.logical import CompileError, WithPlan, cte_schema
from quail.planner.plan import EngineConfig

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
    assert isinstance(query, WithQuery)
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


def test_a_label_feeds_a_later_filter_and_dependent_ctes_run_in_order(session):
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
        SELECT l.id FROM leases l
        WHERE AI.IF(PROMPT('kept: {0}', l.kind))
    """, dialect="bq")
    assert [cte.name for cte in query.logical.ctes] == ["kinds", "leases"]
    assert query.logical.ctes[1].reads == ("kinds",)
    assert query.logical.reads == ("leases",)
    oracle = Oracle(
        labels=["lease", "deed", "lease"],
        filters=lambda alias, document, position: {
            "k": [True, False, True], "l": [True, False]}[alias][document])
    result, recorder = run(query, oracle)
    assert result.collect().column("l.id").to_pylist() == [1]
    assert [list(request.inputs) for request in recorder.requests] == [
        ["c"], ["k"], ["l"]]
    # the label column is text in the CTE's table, and the filter read it
    assert recorder.documents["k"] == [[["lease"], ["deed"], ["lease"]]]
    assert recorder.documents["l"] == [[["lease"], ["lease"]]]


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
    # a CTE nothing reads is compiled but does not run
    assert [cte.name for cte in query.logical.needed()] == ["clauses"]
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


def test_a_subquery_in_from_and_a_nested_with_are_stages_too(session):
    query = session.sql("""
        WITH scored AS (
            WITH clauses AS (
                SELECT c.id,
                       AI.EXTRACT(c.text, {'terms': 'What is the termination clause?'})
                           AS clause
                FROM contracts c
            )
            SELECT x.id, x.clause,
                   AI.SCORE(PROMPT('early? {0}', x.clause.terms)) AS score
            FROM clauses x
        )
        SELECT s.id, s.score
        FROM (SELECT y.id, y.score, y.clause FROM scored y
              WHERE AI.IF(PROMPT('high: {0}', y.clause.terms))) s
        WHERE AI.IF(PROMPT('kept: {0}', s.clause.terms))
    """, dialect="bq")
    assert [cte.name for cte in query.logical.ctes] == ["clauses", "scored", "s"]
    assert [cte.reads for cte in query.logical.ctes] == [(), ("clauses",), ("scored",)]
    assert query.logical.reads == ("s",)
    oracle = Oracle(answers=lambda spec, document: CLAUSES[document],
                    scores=lambda spec, ids: 0.9 if ids[0] == 0 else 0.1,
                    filters=lambda alias, document, position: {
                        "y": [True, False, True], "s": [True, False]}[alias][document])
    result, recorder = run(query, oracle)
    assert [list(request.inputs) for request in recorder.requests] == [
        ["c"], ["x"], ["y"], ["s"]]
    assert result.collect().to_pylist() == [{"s.id": 1, "s.score": 0.9}]


def test_an_empty_cte_result_gives_an_empty_result(session):
    query = session.sql("""
        WITH kept AS (
            SELECT c.id, c.text FROM contracts c
            WHERE AI.IF(PROMPT('keep: {0}', c.text))
        )
        SELECT k.id, AI.SCORE(PROMPT('score {0}', k.text)) AS score FROM kept k
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
    assert planned.index("with clauses:") < planned.index("query (reads clauses):")
    assert ("'x' reads the CTE 'clauses', which has not run; planned on 3 rows "
            f"of {ASSUMED_ANSWER_TOKENS} tokens each") in planned
    assert "run:" not in planned
    plan = query.plan()
    assert plan.gpus == 1 and plan.backend == "quail"
    assert plan.estimated_seconds == sum(
        stage.estimated_seconds for stage in plan.plans)
    assert set(plan.to_envelope(session.registry.codecs)) == {"ctes", "query"}
    # a passed-through document column keeps the source table's lengths
    passthrough = session.sql("""
        WITH kept AS (
            SELECT c.id, c.text FROM contracts c
            WHERE AI.IF(PROMPT('keep: {0}', c.text))
        )
        SELECT k.id FROM kept k WHERE AI.IF(PROMPT('again: {0}', k.text))
    """, dialect="bq")
    assert "planned on 1 rows of 8 tokens each" in passthrough.explain()

    oracle = Oracle(answers=lambda spec, document: CLAUSES[document],
                    scores=lambda spec, ids: 0.5)
    executor = oracle_executor(oracle)
    original = execution.execute_query
    monkeypatch.setattr(
        "quail.api.query.execute_query",
        lambda q, physical_executor=executor, **options: original(
            q, physical_executor=physical_executor, **options))
    measured = query.explain(analyze=True)
    assert measured.count("run:") == 2
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
    assert cte_schema(session.sql(queries["label"]).logical).names == [
        "id", "kind", "kind_probabilities"]
    assert cte_schema(session.sql(queries["extract"]).logical).names == [
        "id", "n", "clause", "clause_span"]


def test_with_rejects_what_the_stages_cannot_express(session):
    extract = ("SELECT c.id, AI.EXTRACT(c.text, {'terms': 'q?'}) AS clause "
               "FROM contracts c")
    read = "SELECT x.id FROM clauses x WHERE AI_FILTER(PROMPT('k {0}', x.clause.terms))"
    for sql, message in (
        (f"WITH RECURSIVE clauses AS ({extract}) {read}", "RECURSIVE"),
        (f"WITH clauses (a, b) AS ({extract}) {read}", "column names"),
        (f"WITH clauses AS ({extract}), clauses AS ({extract}) {read}",
         "defined twice"),
        (f"WITH contracts AS ({extract}) {read}", "registered table"),
        (f"SELECT x.id FROM ({extract}) WHERE "
         f"AI_FILTER(PROMPT('k {{0}}', x.clause.terms))", "needs an alias"),
        ("SELECT c.id FROM contracts c WHERE c.id IN (SELECT d.id FROM contracts d) "
         "AND AI_FILTER(PROMPT('k {0}', c.text))", "subqueries other than"),
        (f"WITH clauses AS (SELECT LENGTH(c.text), AI.EXTRACT(c.text, "
         f"{{'terms': 'q?'}}) AS clause FROM contracts c) {read}", "AS name"),
        (f"WITH clauses AS (SELECT c.id, d.id FROM contracts c JOIN contracts d "
         f"ON AI_FILTER(PROMPT('k {{0}} {{1}}', c.text, d.text))) {read}",
         "name twice"),
        (f"WITH later AS (SELECT x.id FROM clauses x WHERE "
         f"AI_FILTER(PROMPT('k {{0}}', x.clause.terms))), clauses AS ({extract}) "
         f"SELECT l.id FROM later l WHERE AI_FILTER(PROMPT('k {{0}}', l.id))",
         "unknown provider 'clauses'"),
        (f"WITH clauses AS ({extract}) SELECT x.id FROM clauses x",
         "no AI predicate"),
        (f"WITH clauses AS (SELECT c.id FROM contracts c) {read}", "no AI predicate"),
    ):
        with pytest.raises(CompileError, match=message):
            session.sql(sql)
    assert isinstance(compile_sql(extract, session.catalog), type(
        session.sql(extract).logical))
    with pytest.raises(RuntimeError, match="edited plans"):
        session.sql(f"WITH clauses AS ({extract}) {read}").run(plan=object())
    assert isinstance(compile_sql(
        f"WITH clauses AS ({extract}) {read}", session.catalog), WithPlan)


def test_a_session_closes_the_token_files_of_its_cte_results(tmp_path):
    session = quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
                            tokenizer=words)
    session.register("contracts", DocumentProvider.from_table(CONTRACTS, id_col="id"))
    query = session.sql(EXTRACT_THEN_SCORE)
    run(query, Oracle(answers=lambda spec, document: CLAUSES[document]))
    (own,) = session._cte_inputs
    directory = Path(own._token_directory.name)
    assert list(directory.iterdir())
    session.close()
    assert not directory.exists()
    assert isinstance(session.catalog, Catalog) and "clauses" not in session.catalog
