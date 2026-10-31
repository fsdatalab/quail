"""Column predicates: parsed in WHERE and ON, applied before any model call."""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, oracle_executor
from helpers.sessions import CONFIG, TRUTH, _run, fake_tok, make_executor

import quail
from quail.catalog.selection import selected_rows
from quail.logical import (
    And,
    ColumnRef,
    Not,
    Or,
    RegularPredicate,
    Scan,
)
from quail.logical import Filter as LogicalFilter
from quail.logical.expressions import normalize

REF = ColumnRef("r", "reviews", "stars")


def _predicate(comparison, value=None):
    return RegularPredicate(REF, comparison, value)


@pytest.fixture()
def session():
    session = quail.Session(CONFIG, tokenizer=fake_tok)
    session.register("reviews", quail.DocumentProvider.from_table(pa.table({
        "id": [f"r{i}" for i in range(6)],
        "review": [f"review {i} " + "pad " * 20 for i in range(6)],
        "stars": [5, 2, 4, None, 3, 5],
        "lang": ["en", "en", "fr", "en", "en", "de"],
    }), id_col="id"))
    yield session
    session.close()


FILTER_SQL = """
    SELECT r.id FROM reviews r
    WHERE AI_FILTER(PROMPT('q1: {0}', r.review), {'selectivity': 0.5})
      AND AI_FILTER(PROMPT('q2: {0}', r.review), {'selectivity': 0.5})
"""


def test_regular_predicates_select_documents_before_the_model(session):
    seen = []

    def execute(request):
        from quail.builtins import built_in_registry
        from quail.physical import Scan as PhysicalScan
        from quail.physical import decode_graph

        graph = decode_graph(request.plan["graph"], built_in_registry().codecs)
        for node in graph.nodes:
            if isinstance(node, PhysicalScan):
                seen.append(len(request.inputs[node.input_id].documents))
        return make_executor(TRUTH)(request)

    # the AI predicates keep r0 and r3 of all six; the regular predicates keep
    # r0, r2, and r4. The planner only estimates that from a sample; the
    # scan's runtime applies the tests, so the executor gets every
    # document and the model sees three
    query = session.sql(
        FILTER_SQL + " AND r.stars >= 3 AND r.lang IN ('en', 'fr') "
        "AND r.stars IS NOT NULL")
    plan = query.plan()
    scan = next(node for node in plan.nodes if type(node).__name__ == "Scan")
    assert (scan.num_documents, scan.expected_rows) == (6, 3.0)
    assert [str(predicate) for predicate in scan.predicates] == [
        "r.stars >= 3", "r.lang IN ('en', 'fr')", "r.stars IS NOT NULL"]
    assert "where r.stars >= 3 and r.lang IN ('en', 'fr')" in query.explain()
    result = _run(query, execute)
    assert seen == [6]
    assert result.to_rows() == [("r0",)]
    stage = next(s for s in result.report["stages"] if s["op"] == "filter")
    assert stage["evaluated"] == 3
    # answer tables and survivors hold positions in the registered table
    answers = result.answer_tables["filters"][("r", stage["written_position"])]
    assert answers.column("r").to_pylist() == [0, 2, 4]
    assert result.survivor_indices["r"].to_pylist() == [0]

    first = "WHERE AI_FILTER(PROMPT('q1: {0}', r.review))"
    query = session.sql(f"SELECT r.id, r.stars FROM reviews r {first} "
                        "AND r.stars < 4 AND r.lang = 'en'")
    assert _run(query, make_executor(TRUTH)).to_rows() == [("r1", 2), ("r4", 3)]
    # a function on a source column reaches the scan as written
    query = session.sql(f"SELECT r.id FROM reviews r {first} AND LENGTH(r.lang) > 2")
    assert _run(query, make_executor(TRUTH)).to_rows() == []


def test_not_moves_down_to_the_column_tests_and_keeps_sqls_nulls():
    stars = pa.table({"stars": [5, None, 2]})
    high, low = _predicate(">", 4), _predicate("<", 3)
    written = Not(And((high, Or((low, Not(_predicate("is null")))))))
    # De Morgan's laws, two NOTs cancel, and NOT IS NULL is IS NOT NULL
    assert normalize(written) == Or((Not(high), And((Not(low),
                                                     _predicate("is null")))))
    assert str(normalize(written)) == (
        "NOT (r.stars > 4) OR (NOT (r.stars < 3) AND r.stars IS NULL)")
    # the null row stays null through NOT, AND, and OR, so it is rejected
    assert written.evaluate(stars).to_pylist() == [False, None, True]
    assert normalize(written).evaluate(stars).to_pylist() == [False, None, True]
    assert selected_rows(stars, (written,)).to_pylist() == [2]


def test_not_and_or_of_one_tables_columns_run_at_its_scan(session):
    # the AI predicates keep r0 (en, 5 stars) and r3 (en, no stars)
    query = session.sql(
        FILTER_SQL + " AND NOT (LOWER(r.lang) LIKE '%e%' AND r.stars < 3)")
    scan = next(node for node in query.logical.walk() if isinstance(node, Scan))
    assert [str(predicate) for predicate in scan.predicates] == [
        "NOT (lower(r.lang) like '%e%') OR NOT (r.stars < 3)"]
    # r3's NULL < 3 makes the OR null, so the scan rejects r3
    assert _run(query, make_executor(TRUTH)).to_rows() == [("r0",)]
    query = session.sql(FILTER_SQL + " AND (r.stars IS NULL OR r.lang = 'de')")
    assert _run(query, make_executor(TRUTH)).to_rows() == [("r3",)]
    # NOT (a OR b) is two tests at the scan, as NOT a AND NOT b
    query = session.sql(
        "SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.review)) "
        "AND NOT (r.lang = 'fr' OR r.stars IS NULL)")
    scan = next(node for node in query.logical.walk() if isinstance(node, Scan))
    assert [str(predicate) for predicate in scan.predicates] == [
        "NOT (r.lang = 'fr')", "r.stars IS NOT NULL"]
    assert _run(query, make_executor(TRUTH)).to_rows() == [
        ("r0",), ("r1",), ("r4",)]


def test_an_or_over_two_tables_runs_on_the_result_rows(session):
    session.register("products", quail.DocumentProvider.from_table(pa.table({
        "asin": ["p0", "p1"], "description": ["d0", "d1"],
        "price": [10, 20]}), id_col="asin"))
    query = session.sql("""
        SELECT r.id, p.asin FROM reviews r
        JOIN products p ON AI_FILTER(PROMPT('x {0} {1}', r.review, p.description))
        WHERE r.stars > 4 OR p.price > 15
        ORDER BY r.id, p.asin
    """)
    filters = [node for node in query.logical.walk()
               if isinstance(node, LogicalFilter)]
    assert [str(node.condition) for node in filters] == [
        "r.stars > 4 OR p.price > 15"]
    result = _run(query, oracle_executor(Oracle(joins=lambda *_: True)))
    # every pair with p1, and r0 and r5 (5 stars) with p0; r3's null
    # stars with p1 is NULL OR TRUE, which is true
    assert result.to_rows() == [
        ("r0", "p0"), ("r0", "p1"), ("r1", "p1"), ("r2", "p1"), ("r3", "p1"),
        ("r4", "p1"), ("r5", "p0"), ("r5", "p1")]
    # the Project read the tested columns; the result does not return them
    assert result.collect().column_names == ["r.id", "p.asin"]
