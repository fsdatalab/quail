"""Column predicates: parsed in WHERE and ON, applied before any model call."""

import pyarrow as pa
import pytest
from test_session import CONFIG, TRUTH, _run, fake_tok, make_executor

import quail
from quail.execution.selection import evaluate_predicate, selected_rows
from quail.frontend.builder import col, prompt
from quail.logical import ColumnPredicate, ColumnRef, CompileError, Scan

REF = ColumnRef("r", "reviews", "stars")


def _predicate(comparison, value=None):
    return ColumnPredicate(REF, comparison, value)


def test_predicates_evaluate_with_nulls_rejected():
    stars = pa.chunked_array([[3, None, 4], [5, 1]])
    assert evaluate_predicate(stars, _predicate(">=", 4)).to_pylist() == [
        False, None, True, True, False]
    assert evaluate_predicate(stars, _predicate("in", (1, 3))).to_pylist() == [
        True, False, False, False, True]
    assert evaluate_predicate(stars, _predicate("is null")).to_pylist() == [
        False, True, False, False, False]
    assert selected_rows({"stars": stars}, (
        _predicate("is not null"), _predicate("<>", 4))).to_pylist() == [0, 3, 4]
    # a float literal compares with an integer column; a string does not
    assert selected_rows({"stars": stars}, (_predicate(">", 3.5),)
                         ).to_pylist() == [2, 3]
    with pytest.raises(CompileError, match="cannot be evaluated"):
        evaluate_predicate(stars, _predicate("=", "five"))


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


def test_column_predicates_select_documents_before_the_model(session):
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

    # the AI predicates keep r0 and r3 of all six; the column tests keep
    # r0, r2, and r4. The planner only estimates that from a sample; the
    # scan's runtime applies the tests, so the executor gets every
    # document and the model sees three
    query = session.sql(
        FILTER_SQL + " AND r.stars >= 3 AND r.lang IN ('en', 'fr') "
        "AND r.stars IS NOT NULL")
    plan = query.plan()
    scan = next(node for node in plan.nodes if type(node).__name__ == "Scan")
    assert (scan.n_docs, scan.expected_docs) == (6, 3.0)
    assert scan.predicates == (
        ("stars", ">=", 3), ("lang", "in", ("en", "fr")),
        ("stars", "is not null", None))
    assert "where r.stars >= 3 and r.lang IN ('en', 'fr')" in query.explain()
    result = _run(query, execute)
    assert seen == [6]
    assert result.to_rows() == [("r0",)]
    stage = next(s for s in result.report["stages"] if s["op"] == "filter")
    assert stage["evaluated"] == 3
    # answer tables and survivors hold positions in the registered table
    answers = result.answer_tables["filters"][("r", stage["written_pos"])]
    assert answers.column("r").to_pylist() == [0, 2, 4]
    assert result.survivor_indices["r"].to_pylist() == [0]

    built = (session.docs("reviews").alias("r")
             .ai_filter(prompt("q1: {0}", col("r.review")))
             .where(col("r.stars") < 4, col("r.lang") == "en")
             .select("r.id", "r.stars"))
    assert _run(built, make_executor(TRUTH)).to_rows() == [("r1", 2), ("r4", 3)]


def test_sql_and_builder_bind_column_tests(session):
    plan = session.sql("""
        SELECT r.id FROM reviews r
        WHERE 4 <= r.stars AND r.lang <> 'de' AND r.stars IS NULL
          AND AI_FILTER(PROMPT('q: {0}', r.review))
    """).logical
    scan = next(node for node in plan.walk() if isinstance(node, Scan))
    lang = ColumnRef("r", "reviews", "lang")
    assert scan.predicates == (
        ColumnPredicate(REF, ">=", 4), ColumnPredicate(lang, "<>", "de"),
        ColumnPredicate(REF, "is null"))
    assert [str(p) for p in scan.predicates] == [
        "r.stars >= 4", "r.lang <> 'de'", "r.stars IS NULL"]
    built = (session.docs("reviews").alias("r")
             .ai_filter(prompt("q: {0}", col("r.review")))
             .where(col("r.stars") >= 4, col("r.lang") != "de",
                    col("r.stars").is_null())
             .select("r.id")).logical
    assert built == plan
    for sql, fragment in (
        ("r.stars BETWEEN 1 AND 3", "not supported"),
        ("NOT r.stars = 3", "NOT is supported"),
        ("r.stars = r.stars", "only AI_FILTER"),
        ("r.stars > CURRENT_DATE", "literal"),
        ("r.nope = 1", "not in provider"),
    ):
        with pytest.raises(CompileError, match=fragment):
            session.sql(f"SELECT r.id FROM reviews r WHERE {sql} AND "
                        f"AI_FILTER(PROMPT('q: {{0}}', r.review))")
    with pytest.raises(CompileError, match="column tests"):
        session.docs("reviews").alias("r").where("r.stars > 3")
    # a plain test in ON restricts the joined table
    session.register("products", quail.DocumentProvider.from_table(pa.table({
        "asin": ["p0", "p1"], "description": ["d0", "d1"],
        "price": [10, 20]}), id_col="asin"))
    plan = session.sql("""
        SELECT r.id, p.asin FROM reviews r
        JOIN products p ON p.price > 15
         AND AI_FILTER(PROMPT('x {0} {1}', r.review, p.description))
    """).logical
    scans = {node.alias: node for node in plan.walk() if isinstance(node, Scan)}
    assert [str(p) for p in scans["p"].predicates] == ["p.price > 15"]
    assert scans["r"].predicates == ()


def test_two_joins_on_one_table_sample_each_key_column(session):
    from quail.physical import HashJoin
    from quail.planner.plan import Refusal

    session.register("products", quail.DocumentProvider.from_table(pa.table({
        "asin": ["p0", "p1"], "description": ["d0", "d1"],
        "price": [5, 2]}), id_col="asin"))
    session.register("authors", quail.DocumentProvider.from_table(pa.table({
        "name": ["n0", "n1"], "bio": ["b0", "b1"],
        "lang": ["en", "fr"]}), id_col="name"))
    query = session.sql("""
        SELECT r.id FROM reviews r
        JOIN products p ON r.stars = p.price
         AND AI_FILTER(PROMPT('x {0} {1}', r.review, p.description))
        JOIN authors a ON r.lang = a.lang
         AND AI_FILTER(PROMPT('y {0} {1}', r.review, a.bio))
        WHERE AI_FILTER(PROMPT('q: {0}', r.review))
    """)
    plan = query.plan()
    assert not isinstance(plan, Refusal)
    fractions = {node.written_pos: node.pair_fraction for node in plan.nodes
                 if isinstance(node, HashJoin)}
    # stars 5, 2, 4, null, 3, 5 against prices 5, 2: three of twelve;
    # langs en x4, fr, de against en, fr: five of twelve
    assert fractions == {0: 3 / 12, 1: 5 / 12}


def test_column_tests_reach_score_scans_and_may_read_the_text(session):
    from quail.physical import Scan as PhysicalScan

    # the score planner builds its own scans; they keep the tests
    scored = (session.docs("reviews").alias("r")
              .where(col("r.stars") >= 4)
              .ai_score(prompt("q: {0}", col("r.review")), name="s")
              .select("r.id", "s"))
    plan = scored.plan()
    scan = next(node for node in plan.nodes if isinstance(node, PhysicalScan))
    assert scan.predicates == (("stars", ">=", 4),)
    assert (scan.n_docs, scan.expected_docs) == (6, 3.0)
    # a test on the document column keeps the text as a value column
    text = "review 2 " + "pad " * 20
    query = (session.docs("reviews").alias("r")
             .ai_filter(prompt("q1: {0}", col("r.review")))
             .where(col("r.review") == text)
             .select("r.id"))
    query.plan()
    logical_scan = next(node for node in query.logical.walk()
                        if isinstance(node, Scan))
    assert "review" in logical_scan.columns
    assert _run(query, make_executor(TRUTH)).to_rows() == []
    query = (session.docs("reviews").alias("r")
             .ai_filter(prompt("q1: {0}", col("r.review")))
             .where(col("r.review") == "review 3 " + "pad " * 20)
             .select("r.id"))
    assert _run(query, make_executor(TRUTH)).to_rows() == [("r3",)]
