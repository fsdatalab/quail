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
    CompileError,
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


def test_any_arrow_scalar_function_runs_in_where_and_select(session):
    # q1 keeps r0, r1, r3, and r4; their stars are 5, 2, None, and 3
    query = session.sql("""
        SELECT r.id, UTF8_REVERSE(UPPER(r.lang)) AS backwards,
               ABS(r.stars - 4) AS gap
        FROM reviews r
        WHERE AI_FILTER(PROMPT('q1: {0}', r.review)) AND ABS(r.stars - 4) <= 1
        ORDER BY r.id""")
    scan = next(node for node in query.logical.walk() if isinstance(node, Scan))
    assert [str(p) for p in scan.predicates] == ["abs(subtract(r.stars, 4)) <= 1"]
    assert _run(query, make_executor(TRUTH)).to_rows() == [
        ("r0", "NE", 1), ("r4", "NE", 1)]
    with pytest.raises(CompileError, match="unknown function 'nope'"):
        session.sql("SELECT r.id FROM reviews r WHERE NOPE(r.lang) = 'x' "
                    "AND AI_FILTER(PROMPT('q1: {0}', r.review))")


def test_where_takes_the_same_arrow_expressions_as_select(session):
    session.register("pairs", quail.DocumentProvider.from_table(pa.table({
        "id": [f"r{i}" for i in range(6)],
        "review": [f"review {i} " + "pad " * 20 for i in range(6)],
        "stars": [5, 2, 4, None, 3, 5],
        "floor": [4, 3, 4, 1, 1, 6],
        "lang": ["EN", "en", "Fr", "EN", "en", "DE"],
    }), id_col="id"))
    # q1 keeps r0, r1, r3, and r4
    first = "SELECT r.id FROM pairs r WHERE AI_FILTER(PROMPT('q1: {0}', r.review))"
    cases = {
        # a boolean function on its own, as SELECT returns it
        "UTF8_IS_UPPER(r.lang)": ["r0", "r3"],
        "NOT UTF8_IS_UPPER(r.lang)": ["r1", "r4"],
        # two columns of one table; r3's null stars is rejected
        "r.stars > r.floor": ["r0", "r4"],
        "r.stars - r.floor >= 1 OR r.lang = 'en'": ["r0", "r1", "r4"],
        "ABS(-r.stars) <= r.floor": ["r1"],
    }
    for test, kept in cases.items():
        query = session.sql(f"{first} AND ({test}) ORDER BY r.id")
        scan = next(node for node in query.logical.walk() if isinstance(node, Scan))
        assert scan.predicates, test
        assert [row[0] for row in _run(query, make_executor(TRUTH)).to_rows()] == (
            kept), test
    query = session.sql(f"{first} AND r.stars > r.floor")
    scan = next(node for node in query.logical.walk() if isinstance(node, Scan))
    assert [str(test) for test in scan.predicates] == ["greater(r.stars, r.floor)"]
    # SELECT returns the same comparison
    query = session.sql(
        "SELECT r.id, r.stars > r.floor AS above FROM pairs r "
        "WHERE AI_FILTER(PROMPT('q1: {0}', r.review)) ORDER BY r.id")
    assert _run(query, make_executor(TRUTH)).to_rows() == [
        ("r0", True), ("r1", False), ("r3", None), ("r4", True)]
    with pytest.raises(CompileError, match="true or false"):
        _run(session.sql(f"{first} AND UPPER(r.lang)"), make_executor(TRUTH))


def test_a_comparison_of_two_tables_in_where_runs_on_the_result_rows(session):
    session.register("products", quail.DocumentProvider.from_table(pa.table({
        "asin": ["p0", "p1"], "description": ["d0", "d1"],
        "price": [3, 4]}), id_col="asin"))
    query = session.sql("""
        SELECT r.id, p.asin FROM reviews r
        JOIN products p ON AI_FILTER(PROMPT('x {0} {1}', r.review, p.description))
        WHERE r.stars > p.price
        ORDER BY r.id, p.asin
    """)
    filters = [node for node in query.logical.walk()
               if isinstance(node, LogicalFilter)]
    assert [str(node.condition) for node in filters] == ["greater(r.stars, p.price)"]
    result = _run(query, oracle_executor(Oracle(joins=lambda *_: True)))
    assert result.to_rows() == [
        ("r0", "p0"), ("r0", "p1"), ("r2", "p0"), ("r5", "p0"), ("r5", "p1")]


SQL_CALLS = {
    # sqlglot parses each to a node of its own; Quail calls Arrow with
    # SQL's meaning. r0, r1, r3, and r4 pass q1
    "LOG10(r.stars * 20) > 1.7": [True, False, None, True],
    "SUBSTRING(r.lang, 2, 1)": ["n", "n", "n", "n"],
    "CONCAT(r.lang, '-', r.lang)": ["en-en", "en-en", "en-en", "en-en"],
    "CASE WHEN r.stars > 3 THEN 'high' WHEN r.stars > 0 THEN 'low' END": [
        "high", "low", None, "low"],
    "CAST(r.stars AS DOUBLE)": [5.0, 2.0, None, 3.0],
    "r.stars IN (2, 3)": [False, True, None, True],
    "ROUND(r.stars / 2)": [3.0, 1.0, None, 2.0],
    "GREATEST(r.stars, 3)": [5, 3, None, 3],
    "STRPOS(r.lang, 'n')": [2, 2, 2, 2],
    "r.stars BETWEEN 2 AND 3": [False, True, None, True],
}


@pytest.mark.parametrize("dialect", ["bq", "snowflake"])
def test_sql_functions_sqlglot_names_call_arrow_with_sqls_meaning(session, dialect):
    for expression, expected in SQL_CALLS.items():
        query = session.sql(
            f"SELECT r.id, {expression} AS v FROM reviews r "
            f"WHERE AI_FILTER(PROMPT('q1: {{0}}', r.review)) ORDER BY r.id",
            dialect=dialect)
        values = _run(query, make_executor(TRUTH)).collect().column("v").to_pylist()
        assert values == expected, expression
    with pytest.raises(CompileError, match="remainder"):
        session.sql("SELECT r.id, MOD(r.stars, 2) AS v FROM reviews r "
                    "WHERE AI_FILTER(PROMPT('q1: {0}', r.review))", dialect=dialect)


def test_order_by_group_by_on_and_prompt_take_expressions(session):
    # q1 keeps r0, r1, r3, and r4; their stars are 5, 2, None, and 3
    first = "FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.review))"
    query = session.sql(f"SELECT r.id {first} ORDER BY ABS(r.stars - 3), r.id")
    assert _run(query, make_executor(TRUTH)).to_rows() == [
        ("r4",), ("r1",), ("r0",), ("r3",)]
    query = session.sql(
        f"SELECT CASE WHEN r.stars > 2 THEN 'hi' ELSE 'lo' END AS band, "
        f"COUNT(*) AS n, SUM(r.stars * 2) AS s {first} "
        f"GROUP BY CASE WHEN r.stars > 2 THEN 'hi' ELSE 'lo' END ORDER BY band")
    assert _run(query, make_executor(TRUTH)).to_rows() == [
        ("hi", 2, 16), ("lo", 2, 4)]
    query = session.sql(
        "SELECT r.id FROM reviews r "
        "WHERE AI_FILTER(PROMPT('q1: {0}', LOWER(TRIM(r.review)))) ORDER BY r.id")
    assert _run(query, make_executor(TRUTH)).to_rows() == [
        ("r0",), ("r1",), ("r3",), ("r4",)]
    session.register("langs", quail.DocumentProvider.from_table(pa.table({
        "code": ["EN", "FR"], "notes": ["n0", "n1"]}), id_col="code"))
    query = session.sql("""
        SELECT r.id, l.code FROM reviews r
        JOIN langs l ON UPPER(TRIM(r.lang)) = l.code
            AND AI_FILTER(PROMPT('x {0} {1}', r.review, l.notes))
        ORDER BY r.id""")
    assert "on upper(utf8_trim_whitespace(r.lang)) = l.code" in query.explain()
    result = _run(query, oracle_executor(Oracle(joins=lambda *_: True)))
    assert result.to_rows() == [("r0", "EN"), ("r1", "EN"), ("r2", "FR"),
                                ("r3", "EN"), ("r4", "EN")]
