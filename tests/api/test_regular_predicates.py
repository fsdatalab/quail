"""Column predicates: parsed in WHERE and ON, applied before any model call."""

import pyarrow as pa
import pytest
from helpers.conditions import condition
from helpers.oracle import Oracle, oracle_executor
from helpers.planning import operators_of, row_tests
from helpers.sessions import CONFIG, TOKENIZER, TRUTH, _run, make_executor

import quail
from quail.common.expressions import CompileError
from quail.common.selection import selected_rows
from quail.physical import decode_plan_value
from quail.physical.plan import graph_from_envelope


@pytest.fixture()
def session():
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
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
    answer = make_executor(TRUTH)

    def execute(request):
        from quail.physical import Scan as PhysicalScan

        graph = graph_from_envelope(request.plan)
        for node in graph.nodes:
            if isinstance(node, PhysicalScan):
                seen.append(len(request.inputs[node.input_id].documents))
        return answer(request)

    execute.oracle = answer.oracle

    # the AI predicates keep r0 and r3 of all six; the regular predicates keep
    # r0, r2, and r4. DataFusion applies them before the region, so the
    # executor gets only those three documents
    query = session.sql(
        FILTER_SQL + " AND r.stars >= 3 AND r.lang IN ('en', 'fr') "
        "AND r.stars IS NOT NULL")
    plan = query.plan()
    scan = next(node for node in plan.nodes if type(node).__name__ == "Scan")
    assert scan.num_documents == 3
    explained = query.explain()
    assert "stars@2 >= 3" in explained
    result = _run(query, execute)
    assert seen == [3]
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
    assert sorted(_run(query, make_executor(TRUTH)).to_rows()) == [("r1", 2), ("r4", 3)]
    # a function on a source column reaches the scan as written
    query = session.sql(f"SELECT r.id FROM reviews r {first} AND LENGTH(r.lang) > 2")
    assert _run(query, make_executor(TRUTH)).to_rows() == []


def test_not_and_or_keep_sqls_nulls():
    stars = pa.table({"stars": [5, None, 2]})
    written = condition(
        "NOT (stars > 4 AND (stars < 3 OR NOT (stars IS NULL)))",
        {"stars": pa.int64()}, "r", "reviews")
    # the null row stays null through NOT, AND, and OR, so it is rejected
    assert written.evaluate(stars).to_pylist() == [False, None, True]
    assert selected_rows(stars, (written,)).to_pylist() == [2]


def test_a_condition_refuses_a_column_of_another_type_than_its_plan_gives():
    written = condition("stars > 4", {"stars": pa.int64()}, "r", "reviews")
    assert written.types == ("Int64",)
    with pytest.raises(CompileError, match="reads 'stars' as Int64"):
        written.evaluate(pa.table({"stars": ["5", "1"]}))


def test_a_condition_read_from_a_plan_evaluates_as_built():
    written = condition("stars > 4", {"stars": pa.int64()}, "r", "reviews")
    read = decode_plan_value(written.to_json())
    rows = pa.table({"stars": [5, 1, None]})
    assert read.evaluate(rows).to_pylist() == [True, False, None]


@pytest.mark.parametrize(("test", "expected"), [
    # Snowflake matches the whole value: 'e' is not 'en'
    ("REGEXP_LIKE(r.lang, 'e')", []),
    ("REGEXP_LIKE(r.lang, 'e.')", [("r0",), ("r3",)]),
    ("RLIKE(r.lang, 'e')", []),
    ("NOT RLIKE(r.lang, 'e')", [("r0",), ("r3",)]),
    ("REGEXP_LIKE(r.lang, r.lang || '?')", [("r0",), ("r3",)]),
])
def test_a_snowflake_regexp_matches_the_whole_value(session, test, expected):
    # the AI predicates keep r0 and r3, both in English
    query = session.sql(FILTER_SQL + f" AND {test}")
    assert _run(query, make_executor(TRUTH)).to_rows() == expected


@pytest.mark.parametrize("test", [
    "REGEXP_CONTAINS(r.lang, 'e')", "REGEXP_LIKE(r.lang, 'e')"])
def test_a_bigquery_regexp_matches_anywhere_in_the_value(session, test):
    query = session.sql(f"""
        SELECT r.id FROM reviews r
        WHERE AI.IF(PROMPT('q1: {{0}}', r.review))
          AND AI.IF(PROMPT('q2: {{0}}', r.review)) AND {test}
    """, dialect="bq")
    assert _run(query, make_executor(TRUTH)).to_rows() == [("r0",), ("r3",)]


def test_not_and_or_of_one_tables_columns_run_at_its_scan(session):
    # the AI predicates keep r0 (en, 5 stars) and r3 (en, no stars)
    query = session.sql(
        FILTER_SQL + " AND NOT (LOWER(r.lang) LIKE '%e%' AND r.stars < 3)")
    # the test runs in DataFusion, before the region
    (scan,) = operators_of(query).scans
    assert scan.predicates == ()
    # r3's NULL < 3 makes the OR null, so DataFusion rejects r3
    assert _run(query, make_executor(TRUTH)).to_rows() == [("r0",)]
    query = session.sql(FILTER_SQL + " AND (r.stars IS NULL OR r.lang = 'de')")
    assert _run(query, make_executor(TRUTH)).to_rows() == [("r3",)]
    # DataFusion runs NOT (a OR b) before the region
    query = session.sql(
        "SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.review)) "
        "AND NOT (r.lang = 'fr' OR r.stars IS NULL)")
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
    # DataFusion tests the OR on the region's rows
    assert row_tests(query) == ["stars > 4 OR price > 15"]
    result = _run(query, oracle_executor(Oracle(joins=lambda *_: True)))
    # every pair with p1, and r0 and r5 (5 stars) with p0; r3's null
    # stars with p1 is NULL OR TRUE, which is true
    assert result.to_rows() == [
        ("r0", "p0"), ("r0", "p1"), ("r1", "p1"), ("r2", "p1"), ("r3", "p1"),
        ("r4", "p1"), ("r5", "p0"), ("r5", "p1")]
    # the Project read the tested columns; the result does not return them
    assert result.collect().column_names == ["r.id", "p.asin"]


def test_datafusion_functions_run_in_where_and_select(session):
    # q1 keeps r0, r1, r3, and r4; their stars are 5, 2, None, and 3
    query = session.sql("""
        SELECT r.id, REVERSE(UPPER(r.lang)) AS backwards,
               ABS(r.stars - 4) AS gap
        FROM reviews r
        WHERE AI_FILTER(PROMPT('q1: {0}', r.review)) AND ABS(r.stars - 4) <= 1
        ORDER BY r.id""")
    # DataFusion's ABS runs the WHERE test before the region
    (scan,) = operators_of(query).scans
    assert scan.predicates == ()
    assert _run(query, make_executor(TRUTH)).to_rows() == [
        ("r0", "NE", 1), ("r4", "NE", 1)]
    with pytest.raises(CompileError, match="unknown function 'nope'"):
        session.sql("SELECT r.id FROM reviews r WHERE NOPE(r.lang) = 'x' "
                    "AND AI_FILTER(PROMPT('q1: {0}', r.review))")


def test_where_takes_the_same_expressions_as_select(session):
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
        # a test of a function's value
        "UPPER(r.lang) = r.lang": ["r0", "r3"],
        "NOT (UPPER(r.lang) = r.lang)": ["r1", "r4"],
        # two columns of one table; r3's null stars is rejected
        "r.stars > r.floor": ["r0", "r4"],
        "r.stars - r.floor >= 1 OR r.lang = 'en'": ["r0", "r1", "r4"],
        "ABS(-r.stars) <= r.floor": ["r1"],
    }
    for test, kept in cases.items():
        query = session.sql(f"{first} AND ({test}) ORDER BY r.id")
        (scan,) = operators_of(query).scans
        # DataFusion runs every plain test before the region
        assert scan.predicates == (), test
        assert [row[0] for row in _run(query, make_executor(TRUTH)).to_rows()] == (
            kept), test
    # SELECT returns the same comparison
    query = session.sql(
        "SELECT r.id, r.stars > r.floor AS above FROM pairs r "
        "WHERE AI_FILTER(PROMPT('q1: {0}', r.review)) ORDER BY r.id")
    assert _run(query, make_executor(TRUTH)).to_rows() == [
        ("r0", True), ("r1", False), ("r3", None), ("r4", True)]
    with pytest.raises(CompileError, match="a test is true or false, got Utf8"):
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
    assert row_tests(query) == ["stars > price"]
    result = _run(query, oracle_executor(Oracle(joins=lambda *_: True)))
    assert result.to_rows() == [
        ("r0", "p0"), ("r0", "p1"), ("r2", "p0"), ("r5", "p0"), ("r5", "p1")]


def test_an_equality_of_two_tables_in_where_pairs_their_rows_before_the_ai_join(
        session):
    session.register("products", quail.DocumentProvider.from_table(pa.table({
        "asin": ["p0", "p1"], "description": ["d0", "d1"],
        "stars": [5, 2]}), id_col="asin"))
    asked = []

    def same(node, assignment):
        asked.append((assignment["r"], assignment["p"]))
        return True

    for where in ("FROM reviews r JOIN products p ON {ai} WHERE r.stars = p.stars",
                  "FROM reviews r, products p WHERE r.stars = p.stars AND {ai}"):
        asked.clear()
        query = session.sql("SELECT r.id, p.asin " + where.format(
            ai="AI_FILTER(PROMPT('x {0} {1}', r.review, p.description))")
            + " ORDER BY r.id")
        # the equality is the AI join's key, as an ON equality would be
        assert [str(condition) for condition in operators_of(query).joins[0].on] == [
            "r.stars = p.stars"]
        assert any(node.type_name == "quail.hash_join" for node in query.plan().nodes)
        result = _run(query, oracle_executor(Oracle(joins=same)))
        assert result.to_rows() == [("r0", "p0"), ("r1", "p1"), ("r5", "p0")]
        # the model is asked only about the pairs with equal stars
        assert sorted(asked) == [(0, 0), (1, 1), (5, 0)]


SQL_CALLS = {
    # DataFusion evaluates each with its SQL meaning. r0, r1, r3, and r4
    # pass q1; their stars are 5, 2, None, and 3
    "LOG10(r.stars * 20) > 1.7": [True, False, None, True],
    "SUBSTRING(r.lang, 2, 1)": ["n", "n", "n", "n"],
    "CONCAT(r.lang, '-', r.lang)": ["en-en", "en-en", "en-en", "en-en"],
    "CASE WHEN r.stars > 3 THEN 'high' WHEN r.stars > 0 THEN 'low' END": [
        "high", "low", None, "low"],
    "CAST(r.stars AS DOUBLE)": [5.0, 2.0, None, 3.0],
    "r.stars IN (2, 3)": [False, True, None, True],
    # an integer divided by an integer is an integer
    "ROUND(r.stars / 2)": [2, 1, None, 1],
    # GREATEST skips a NULL
    "GREATEST(r.stars, 3)": [5, 3, 3, 3],
    "r.stars % 2": [1, 0, None, 1],
    "STRPOS(r.lang, 'n')": [2, 2, 2, 2],
    "r.stars BETWEEN 2 AND 3": [False, True, None, True],
}


@pytest.mark.parametrize("dialect", ["bq", "snowflake"])
def test_sql_functions_have_datafusions_meaning(session, dialect):
    for expression, expected in SQL_CALLS.items():
        query = session.sql(
            f"SELECT r.id, {expression} AS v FROM reviews r "
            f"WHERE AI_FILTER(PROMPT('q1: {{0}}', r.review)) ORDER BY r.id",
            dialect=dialect)
        values = _run(query, make_executor(TRUTH)).collect().column("v").to_pylist()
        assert values == expected, expression
    with pytest.raises(CompileError, match="write r.stars % 2"):
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
    # DataFusion computes the key before the region; the region joins on it
    explained = query.explain()
    assert "upper(btrim(lang@2)) as __quail_expr_0" in explained
    result = _run(query, oracle_executor(Oracle(joins=lambda *_: True)))
    assert result.to_rows() == [("r0", "EN"), ("r1", "EN"), ("r2", "FR"),
                                ("r3", "EN"), ("r4", "EN")]
