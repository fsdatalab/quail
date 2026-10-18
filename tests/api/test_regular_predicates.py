"""Column predicates: parsed in WHERE and ON, applied before any model call."""

import pyarrow as pa
import pytest
from helpers.sessions import CONFIG, TRUTH, _run, fake_tok, make_executor

import quail
from quail.catalog.selection import selected_rows
from quail.frontend.expressions import col, prompt
from quail.logical import (
    ColumnRef,
    CompileError,
    GetField,
    RegularPredicate,
    ScalarFunction,
    Scan,
)
from quail.logical.expressions import column_expression

REF = ColumnRef("r", "reviews", "stars")


def _predicate(comparison, value=None):
    return RegularPredicate(REF, comparison, value)


def test_predicates_evaluate_with_nulls_rejected():
    stars = pa.table({"stars": pa.chunked_array([[3, None, 4], [5, 1]])})
    assert _predicate(">=", 4).evaluate(stars).to_pylist() == [
        False, None, True, True, False]
    assert _predicate("in", (1, 3)).evaluate(stars).to_pylist() == [
        True, False, False, False, True]
    assert _predicate("is null").evaluate(stars).to_pylist() == [
        False, True, False, False, False]
    assert selected_rows(stars, (
        _predicate("is not null"), _predicate("<>", 4))).to_pylist() == [0, 3, 4]
    # a float literal compares with an integer column; a string does not
    assert selected_rows(stars, (_predicate(">", 3.5),)).to_pylist() == [2, 3]
    with pytest.raises(CompileError, match="cannot be evaluated"):
        _predicate("=", "five").evaluate(stars)


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

    built = (session.docs("reviews").alias("r")
             .ai_filter(prompt("q1: {0}", col("r.review")))
             .where(col("r.stars") < 4, col("r.lang") == "en")
             .select("r.id", "r.stars"))
    assert _run(built, make_executor(TRUTH)).to_rows() == [("r1", 2), ("r4", 3)]
    # a function on a source column reaches the scan as written
    built = (session.docs("reviews").alias("r")
             .ai_filter(prompt("q1: {0}", col("r.review")))
             .where(col("r.lang").length() > 2)
             .select("r.id"))
    assert _run(built, make_executor(TRUTH)).to_rows() == []


def test_sql_and_builder_bind_regular_predicates(session):
    plan = session.sql("""
        SELECT r.id FROM reviews r
        WHERE 4 <= r.stars AND r.lang <> 'de' AND r.stars IS NULL
          AND AI_FILTER(PROMPT('q: {0}', r.review))
    """).logical
    scan = next(node for node in plan.walk() if isinstance(node, Scan))
    lang = ColumnRef("r", "reviews", "lang")
    assert scan.predicates == (
        RegularPredicate(REF, ">=", 4), RegularPredicate(lang, "<>", "de"),
        RegularPredicate(REF, "is null"))
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
    with pytest.raises(CompileError, match="takes tests"):
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


def test_where_on_the_joined_query_reaches_its_scan(session):
    session.register("products", quail.DocumentProvider.from_table(pa.table({
        "asin": ["p0", "p1"], "description": ["d0", "d1"],
        "price": [10, 20]}), id_col="asin"))
    sql = session.sql("""
        SELECT r.id, p.asin FROM reviews r
        JOIN products p ON r.stars = p.price AND p.price > 15
         AND AI_FILTER(PROMPT('x {0} {1}', r.review, p.description))
        WHERE r.lang = 'en'
    """).logical
    products = session.docs("products").alias("p").where(col("p.price") > 15)
    built = (session.docs("reviews").alias("r").where(col("r.lang") == "en")
             .join(products, on=col("r.stars") == col("p.price"))
             .ai_filter(prompt("x {0} {1}", col("r.review"),
                               col("p.description")))
             .select("r.id", "p.asin")).logical
    assert built == sql

    def predicates(plan):
        return {node.alias: [str(p) for p in node.predicates]
                for node in plan.walk() if isinstance(node, Scan)}

    products = session.docs("products").alias("p").where(col("p.price") > 15)
    built = (session.docs("reviews").alias("r").where(col("r.lang") == "en")
             .ai_join(products, prompt("x {0} {1}", col("r.review"),
                                       col("p.description")))
             .select("r.id", "p.asin")).logical
    assert predicates(built) == {"r": ["r.lang = 'en'"], "p": ["p.price > 15"]}


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
    fractions = {node.written_position: node.pair_fraction for node in plan.nodes
                 if isinstance(node, HashJoin)}
    # stars 5, 2, 4, null, 3, 5 against prices 5, 2: three of twelve;
    # langs en x4, fr, de against en, fr: five of twelve
    assert fractions == {0: 3 / 12, 1: 5 / 12}


def test_regular_predicates_reach_score_scans_and_may_read_the_text(session):
    from quail.physical import Scan as PhysicalScan

    scored = (session.docs("reviews").alias("r")
              .where(col("r.stars") >= 4)
              .ai_score(prompt("q: {0}", col("r.review")), name="s")
              .select("r.id", "s"))
    plan = scored.plan()
    scan = next(node for node in plan.nodes if isinstance(node, PhysicalScan))
    assert [str(predicate) for predicate in scan.predicates] == ["r.stars >= 4"]
    assert (scan.num_documents, scan.expected_rows) == (6, 3.0)
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


def test_text_tests_paths_and_functions_evaluate_with_one_arrow_call_each():
    names = pa.table({"name": pa.chunked_array([["Acme Corp", None, "acme inc"],
                                                ["Beta"]])})
    name = ColumnRef("r", "reviews", "name")

    def passes(comparison, value=None, path=(), function=None):
        predicate = RegularPredicate(column_expression(name, path, function),
                                     comparison, value)
        return predicate.evaluate(names).to_pylist()

    assert passes("like", "Acme%") == [True, None, False, False]
    assert passes("ilike", "%ACME%") == [True, None, True, False]
    assert passes("not ilike", "%ACME%") == [False, None, False, True]
    assert passes("contains", "inc") == [False, None, True, False]
    assert passes("icontains", "CORP") == [True, None, False, False]
    assert passes("starts with", "B") == [False, None, False, True]
    assert passes("ends with", "c") == [False, None, True, False]
    assert passes("regex", r"\binc\b") == [False, None, True, False]
    assert passes("regex", "^(?:acme)$") == [False, None, False, False]
    assert passes("<", 5, function="length") == [False, None, False, True]
    assert passes("=", "Beta") == [False, None, False, True]
    spans = pa.table({"name": pa.chunked_array([pa.StructArray.from_arrays(
        [pa.array([14, 0]), pa.array([27, 3])], names=["start", "end"],
        mask=pa.array([False, True]))])})
    predicate = RegularPredicate(GetField(name, "start"), "<", 20)
    assert predicate.evaluate(spans).to_pylist() == [True, None]
    # a text test needs a string; a path needs a struct
    with pytest.raises(CompileError, match="compares with a string"):
        RegularPredicate(name, "like", 3).validate()
    with pytest.raises(CompileError, match="unsupported column function"):
        RegularPredicate(ScalarFunction("upper", (name,)), "<", 3).validate()
    with pytest.raises(CompileError, match="has no fields"):
        RegularPredicate(GetField(name, "start"), "<", 3).evaluate(names)
    # source columns take the same tests, at the scan
    reviews = ColumnRef("r", "reviews", "review")
    both = names.append_column("review", names.column("name"))
    assert selected_rows(both, (
        RegularPredicate(reviews, "ilike", "%acme%"),
        RegularPredicate(ScalarFunction("length", (name,)), ">", 5),
    )).to_pylist() == [0, 2]


def test_stop_key_groups_follow_numeric_equality():
    from quail.catalog.selection import group_ids

    values = pa.array([-0.0, 0.0, float("nan"), float("nan"), 1.0, None, None])
    groups = group_ids([values, values.dictionary_encode()]).tolist()
    assert groups[0] == groups[1] and groups[2] == groups[3]
    assert groups[5] == groups[6] and len(set(groups)) == 4
