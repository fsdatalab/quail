"""WITH: common table expressions whose results later AI operators read."""


import pyarrow as pa
import pytest
from helpers.oracle import Oracle, oracle_executor
from helpers.sessions import TOKENIZER, encoder

import quail
from quail.api import execute as execution
from quail.api.config import EngineConfig
from quail.catalog import DocumentProvider
from quail.common.expressions import CompileError

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


@pytest.fixture()
def session():
    with quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
                       tokenizer=TOKENIZER) as value:
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
    # the CTE is its own model region, which runs before the one reading it
    explained = query.explain()
    assert explained.count("model region") == 2
    assert "AiExtract" in explained and "quail.ai_score" in explained
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
    assert [[int(token) for token in tokens] for tokens in scored] == [
        encoder()(CLAUSES[0]), [], encoder()(CLAUSES[2])]
    assert len(recorder.requests) == 2


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
    oracle = Oracle(answers=lambda spec, document: CLAUSES[document],
                    joins=lambda node, assignment: (
                        assignment["a"] == assignment["b"]))
    result, recorder = run(query, oracle)
    # the extraction once, though the join reads it twice, and the join;
    # the CTE nothing reads never runs
    assert len(recorder.requests) == 2
    rows = result.collect()
    assert rows.column("a.id").to_pylist() == [1, 2, 3]
    assert rows.column("other").to_pylist() == [1, 2, 3]
    # both aliases read the one result table
    assert recorder.documents["a"] == recorder.documents["b"]


@pytest.mark.parametrize("sql", [
    """WITH first AS (
           SELECT c.id FROM contracts c WHERE AI.IF(PROMPT('keep {0}', c.text))
           ORDER BY c.id LIMIT 2)
       SELECT f.id AS result_id FROM first f WHERE f.id > 0""",
])
def test_relational_stages_before_and_after_ai_return_the_selected_rows(session, sql):
    result, recorder = run(session.sql(sql, dialect="bq"),
                           Oracle(filters=lambda *args: True,
                                  joins=lambda node, assignment:
                                  len(set(assignment.values())) == 1))
    assert result.collect().column("result_id").to_pylist() == [1, 2]
    assert "Project" in result.explain()
    assert len(recorder.requests) == 1


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


def test_a_cte_that_classifies_returns_the_columns_its_reader_names(session):
    query = session.sql("""
        WITH k AS (
            SELECT c.id, AI.CLASSIFY(c.text, ['lease', 'note']) AS kind
            FROM contracts c)
        SELECT k.id, k.kind FROM k WHERE k.kind = 'lease'
    """)
    result, _ = run(query, Oracle(labels=lambda spec, ids: spec.labels[ids[0] % 2]))
    # the CTE is its own model region; DataFusion names the reader's columns
    assert result.collect().to_pydict() == {
        "k.id": [1, 3], "k.kind": ["lease", "lease"]}


def test_a_cte_read_through_a_join_that_reuses_the_readers_alias_is_refused(session):
    with pytest.raises(CompileError, match="duplicate table alias 'c'"):
        session.sql("""
            WITH kept AS (
                SELECT c.id, c.text FROM contracts c
                WHERE AI.IF(PROMPT('keep {0}', c.text)))
            SELECT k.id FROM kept k JOIN contracts c ON k.id = c.id
        """)
