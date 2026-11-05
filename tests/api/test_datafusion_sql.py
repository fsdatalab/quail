"""SQL that DataFusion runs around Quail's model regions.

Quail runs the operators that call the model and those that change
what the model reads; DataFusion runs the rest of the SQL, so any SQL
DataFusion runs works with AI functions in it.
"""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish
from helpers.sessions import CONFIG, fake_tok

import quail
from quail.logical import CompileError

AI_IF = "AI_FILTER(PROMPT('q: {0}', r.body))"
# the model keeps every review but r1
ORACLE = Oracle(filters=lambda alias, document, position: document != 1,
                labels=lambda spec, ids: "a")


@pytest.fixture
def session():
    with quail.Session(CONFIG, tokenizer=fake_tok) as session:
        session.register("reviews", quail.DocumentProvider.from_table(pa.table({
            "id": ["r0", "r1", "r2"], "body": ["a", "b", "c"],
            "stars": [5, 2, 4], "pid": ["p0", "p1", "p9"]}), id_col="id"))
        session.register("products", quail.DocumentProvider.from_table(pa.table({
            "id": ["p0", "p1"], "text": ["x", "y"]}), id_col="id"))
        yield session


@pytest.mark.parametrize("sql, rows", [
    ("SELECT r.id FROM reviews r WHERE r.stars > 3 ORDER BY r.id",
     [("r0",), ("r2",)]),
    (f"SELECT r.id FROM reviews r WHERE {AI_IF} AND r.stars BETWEEN 2 AND 4",
     [("r2",)]),
    (f"SELECT r.id, p.text FROM reviews r JOIN products p ON r.pid = p.id "
     f"WHERE {AI_IF} ORDER BY r.id", [("r0", "x")]),
    (f"SELECT r.id, p.id FROM reviews r CROSS JOIN products p WHERE {AI_IF} "
     "ORDER BY r.id, p.id",
     [("r0", "p0"), ("r0", "p1"), ("r2", "p0"), ("r2", "p1")]),
    (f"SELECT r.id, p.text FROM reviews r LEFT JOIN products p ON r.pid = p.id "
     f"WHERE {AI_IF} ORDER BY r.id", [("r0", "x"), ("r2", None)]),
    (f"SELECT r.id FROM reviews r WHERE {AI_IF} "
     "AND r.stars > (SELECT AVG(stars) FROM reviews) ORDER BY r.id",
     [("r0",), ("r2",)]),
    (f"SELECT r.id FROM reviews r WHERE {AI_IF} INTERSECT "
     "SELECT r.id FROM reviews r WHERE r.stars > 3 ORDER BY 1",
     [("r0",), ("r2",)]),
    ("SELECT r.id, CASE WHEN AI.CLASSIFY(PROMPT('t: {0}', r.body), ['a', 'b']) "
     "= 'a' THEN 1 ELSE 0 END AS f FROM reviews r ORDER BY r.id",
     [("r0", 1), ("r1", 1), ("r2", 1)]),
])
def test_sql_around_ai_functions_returns_what_datafusion_computes(session, sql, rows):
    query = session.sql(sql)
    # a query with no model call has no region for the oracle to answer
    result = finish(query, ORACLE) if hasattr(query, "logical") else query.run()
    table = result.collect()
    assert [tuple(row.values()) for row in table.to_pylist()] == rows


def test_an_or_of_a_model_call_and_a_column_test_is_refused_with_its_reason(session):
    with pytest.raises(CompileError, match="an OR of a model call and other tests"):
        session.sql(f"SELECT r.id FROM reviews r WHERE {AI_IF} OR r.stars > 3")


def test_a_limit_above_a_datafusion_projection_stops_at_its_rows(session):
    # DataFusion runs the CASE, and pushes the LIMIT into the region below
    query = session.sql(
        "SELECT r.id, CASE WHEN AI.CLASSIFY(PROMPT('t: {0}', r.body), ['a', 'b']) "
        "= 'a' THEN 1 ELSE 0 END AS f FROM reviews r LIMIT 2")
    assert finish(query, ORACLE).collect().num_rows == 2
