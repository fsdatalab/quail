"""Column predicates: parsed in WHERE and ON, applied before any model call."""

import json
import operator
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pytest
from helpers.sessions import CONFIG, TRUTH, _run, fake_tok, make_executor

import quail
from quail.catalog.selection import selected_rows
from quail.frontend.expressions import col, prompt
from quail.logical import ColumnRef, CompileError, RegularPredicate, Scan

REF = ColumnRef("r", "reviews", "stars")


@pytest.mark.parametrize("values,test,expected", [
    ([0, 10, 20], col("r.value") >= np.int64(15), [2]),
    ([3, 4, None], col("r.value") > np.float32(3.5), [1]),
    ([float(2**53), float(2**53 + 2), None],
     col("r.value") > np.int64(2**53 + 1), [1]),
    ([0, 1, None], col("r.value") < np.uint64(2**64 - 1), [0, 1]),
    ([0, 10, 20], col("r.value").isin(np.array([10, 20])), [1, 2]),
    ([3, 4, None], col("r.value").isin(np.array([3.5, 4], np.float32)), [1]),
    ([True, False, None], col("r.value") == np.bool_(True), [0]),
    (["en", "fr", None], col("r.value") == np.str_("en"), [0]),
])
def test_numpy_literals_survive_planning_and_execution(values, test, expected):
    with quail.Session(CONFIG, tokenizer=fake_tok) as session:
        session.register("docs", quail.DocumentProvider.from_table(pa.table({
            "id": range(len(values)), "text": ["doc"] * len(values),
            "value": values,
        }), id_col="id"))
        query = (session.docs("docs").alias("r").where(test)
                 .ai_filter(prompt("keep: {0}", col("r.text"))).select("r.id"))
        execute = make_executor({"r": {"keep:": [True] * len(values)}})

        def serialized_execute(request):
            return execute(replace(request, plan=json.loads(json.dumps(request.plan))))

        assert _run(query, serialized_execute).to_rows() == [
            (index,) for index in expected]


def _predicate(comparison, value=None):
    return RegularPredicate(REF, comparison, value)


def test_predicates_evaluate_with_nulls_rejected():
    stars = pa.chunked_array([[3, None, 4], [5, 1]])
    assert _predicate(">=", 4).evaluate(stars).to_pylist() == [
        False, None, True, True, False]
    assert _predicate("in", (1, 3)).evaluate(stars).to_pylist() == [
        True, False, False, False, True]
    assert _predicate("is null").evaluate(stars).to_pylist() == [
        False, True, False, False, False]
    assert selected_rows({"stars": stars}, (
        _predicate("is not null"), _predicate("<>", 4))).to_pylist() == [0, 3, 4]
    # a float literal compares with an integer column; a string does not
    assert selected_rows({"stars": stars}, (_predicate(">", 3.5),)
                         ).to_pylist() == [2, 3]
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
    assert (scan.n_docs, scan.expected_docs) == (6, 3.0)
    assert scan.predicates == (
        RegularPredicate(REF, ">=", 3),
        RegularPredicate(ColumnRef("r", "reviews", "lang"), "in", ("en", "fr")),
        RegularPredicate(REF, "is not null"))
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
    fractions = {node.written_pos: node.pair_fraction for node in plan.nodes
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
    assert scan.predicates == (RegularPredicate(REF, ">=", 4),)
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


def test_text_tests_paths_and_functions_evaluate_with_one_arrow_call_each():
    names = pa.chunked_array([["Acme Corp", None, "acme inc"], ["Beta"]])
    name = ColumnRef("r", "reviews", "name")

    def passes(comparison, value=None, path=(), function=None):
        predicate = RegularPredicate(name, comparison, value, path, function)
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
    spans = pa.chunked_array([pa.StructArray.from_arrays(
        [pa.array([14, 0]), pa.array([27, 3])], names=["start", "end"],
        mask=pa.array([False, True]))])
    predicate = RegularPredicate(name, "<", 20, ("start",))
    assert predicate.evaluate(spans).to_pylist() == [True, None]
    # a text test needs a string; a path needs a struct
    with pytest.raises(CompileError, match="compares with a string"):
        RegularPredicate(name, "like", 3).validate()
    with pytest.raises(CompileError, match="unsupported column function"):
        RegularPredicate(name, "<", 3, function="upper").validate()
    with pytest.raises(CompileError, match="cannot be evaluated"):
        RegularPredicate(name, "<", 3, ("start",)).evaluate(names)
    # source columns take the same tests, at the scan
    reviews = ColumnRef("r", "reviews", "review")
    assert selected_rows({"review": names, "name": names}, (
        RegularPredicate(reviews, "ilike", "%acme%"),
        RegularPredicate(name, ">", 5, function="length"))).to_pylist() == [0, 2]


@pytest.mark.parametrize("dialect", ["snowflake", "bq", "builder"])
def test_source_length_survives_planning_and_execution(session, dialect):
    session.register("lengths", quail.DocumentProvider.from_table(pa.table({
        "id": ["short", "long", "null"],
        "review": ["abc", "abcdefgh", "other"],
        "name": ["abc", "abcdefgh", None],
    }), id_col="id"))
    if dialect == "builder":
        query = (session.docs("lengths").alias("r")
                 .ai_filter(prompt("q1: {0}", col("r.review")))
                 .where(col("r.name").length() > 6)
                 .select("r.id"))
    else:
        query = session.sql(
            "SELECT r.id FROM lengths r WHERE LENGTH(r.name) > 6 "
            "AND AI_FILTER(PROMPT('q1: {0}', r.review))", dialect=dialect)
    result = _run(query, make_executor({"r": {"q1:": [1, 1, 1]}}))
    assert result.to_rows() == [("long",)]
    assert result.survivor_indices["r"].to_pylist() == [1]


@pytest.mark.parametrize("comparison,compare", [
    ("=", operator.eq), ("<>", operator.ne), ("<", operator.lt),
    ("<=", operator.le), (">", operator.gt), (">=", operator.ge),
])
@pytest.mark.parametrize("literal", [3.5, -3.5, 300])
@pytest.mark.parametrize("dtype", [pa.int8(), pa.int64(), pa.float32()])
def test_numeric_literals_keep_their_value(comparison, compare, literal, dtype):
    source = [-4, -3, None, 3, 4]
    values = pa.chunked_array([source[:2], source[2:]], type=dtype)
    assert _predicate(comparison, literal).evaluate(values).to_pylist() == [
        None if value is None else compare(value, literal) for value in source]


@pytest.mark.parametrize("comparison,compare", [
    ("=", operator.eq), ("<>", operator.ne), ("<", operator.lt),
    ("<=", operator.le), (">", operator.gt), (">=", operator.ge),
])
@pytest.mark.parametrize("literal", [
    -2**64, 2**53 + 1, 2**63, 2**64, float(2**53), 7.5,
    float("inf"), float("-inf"), float("nan"), 10**400, -10**400,
])
@pytest.mark.parametrize("dtype", [pa.int64(), pa.uint64(), pa.float64()])
def test_numeric_comparisons_preserve_large_values(comparison, compare, literal, dtype):
    source = [0, 7, 2**53, 2**53 + 1, 2**63 - 1, None]
    if dtype == pa.uint64():
        source += [2**63, 2**64 - 1]
    elif dtype == pa.int64():
        source += [-2**63]
    else:
        source = [float(value) if value is not None else None for value in source]
        source += [float("inf"), float("-inf"), float("nan")]
    values = pa.chunked_array([source[:2], source[2:]], type=dtype)
    assert _predicate(comparison, literal).evaluate(values).to_pylist() == [
        None if value is None else compare(value, literal) for value in source]


@pytest.mark.parametrize("dtype", [pa.int64(), pa.uint64(), pa.float64()])
def test_numeric_membership_preserves_large_and_fractional_literals(dtype):
    source = [0, 7, 2**53, 2**53 + 1, 2**63 - 1, None]
    if dtype == pa.uint64():
        source += [2**63, 2**64 - 1]
    elif dtype == pa.float64():
        source = [float(value) if value is not None else None for value in source]
    literals = (7.0, 7.5, 2**53 + 1, 2**63, 2**64 - 1, 10**400, None)
    values = pa.chunked_array([source[:2], source[2:]], type=dtype)
    assert _predicate("in", literals).evaluate(values).to_pylist() == [
        value is not None and value in literals for value in source]


@pytest.mark.parametrize("dtype", [pa.float32(), pa.float64()])
@pytest.mark.parametrize("literal", [0.0, -0.0, float("nan"), -float("nan")])
def test_numeric_membership_agrees_with_equality(dtype, literal):
    values = pa.chunked_array([[0.0, -0.0], [float("nan"), -float("nan"), None]],
                              type=dtype)
    assert _predicate("in", (literal,)).accepts(values).equals(
        _predicate("=", literal).accepts(values))


@pytest.mark.parametrize("dtype", [pa.int64(), pa.uint64(), pa.float64(), pa.null()])
@pytest.mark.parametrize("condition,accepts", [
    ("> 9007199254740992.0", lambda value: value > float(2**53)),
    ("< 18446744073709551616", lambda value: value < 2**64),
    ("IN (7.5, 9007199254740993, 18446744073709551615)",
     lambda value: value in (7.5, 2**53 + 1, 2**64 - 1)),
])
def test_large_numeric_sql_predicates_reach_execution(
        session, dtype, condition, accepts):
    values = [7, 2**53, 2**53 + 1, None]
    if dtype == pa.uint64():
        values += [2**64 - 1]
    elif dtype == pa.float64():
        values = [float(value) if value is not None else None for value in values]
    elif dtype == pa.null():
        values = [None] * len(values)
    session.register("numbers", quail.DocumentProvider.from_table(pa.table({
        "id": range(len(values)), "key": pa.array(values, dtype),
        "review": ["document"] * len(values),
    }), id_col="id"))
    query = session.sql(
        f"SELECT r.id FROM numbers r WHERE r.key {condition} "
        "AND AI_FILTER(PROMPT('q1: {0}', r.review))")
    result = _run(query, make_executor({"r": {"q1:": [True] * len(values)}}))
    assert result.to_rows() == [(index,) for index, value in enumerate(values)
                                if value is not None and accepts(value)]


@pytest.mark.parametrize("literals,expected", [
    ((3.5,), []), ((3.5, 4), [2]), ((3, 300), [0]),
])
def test_in_literals_are_not_truncated_to_the_column_type(literals, expected):
    values = pa.array([3, None, 4], type=pa.int8())
    assert selected_rows({"stars": values}, (_predicate("in", literals),)
                         ).to_pylist() == expected


@pytest.mark.parametrize("comparison,expected", [
    ("= 3.5", []), ("< 3.5", ["short"]), (">= 3.5", ["long"]),
    ("IN (3.5, 4)", ["long"]),
])
def test_fractional_length_predicates_reach_execution(session, comparison, expected):
    session.register("lengths", quail.DocumentProvider.from_table(pa.table({
        "id": ["short", "long", "null"],
        "review": ["abc", "abcd", "other"], "name": ["abc", "abcd", None],
    }), id_col="id"))
    query = session.sql(
        f"SELECT r.id FROM lengths r WHERE LENGTH(r.name) {comparison} "
        "AND AI_FILTER(PROMPT('q1: {0}', r.review))")
    result = _run(query, make_executor({"r": {"q1:": [1, 1, 1]}}))
    assert result.to_rows() == [(name,) for name in expected]


@pytest.mark.parametrize("function", ["REGEXP_LIKE", "RLIKE"])
@pytest.mark.parametrize("pattern,flags,expected", [
    ("acme", "i", [0, 1]), ("acme", "c", [1]),
    ("acme", "ic", [1]), ("acme", "ci", [0, 1]),
    ("acme", "im", [0, 1, 3]), ("a.b", "s", [2]),
    ("acme", "e", [1]), ("acme", "", [1]),
])
def test_regex_flags_reach_execution(session, function, pattern, flags, expected):
    session.register("names", quail.DocumentProvider.from_table(pa.table({
        "id": range(5), "review": ["doc"] * 5,
        "name": ["Acme", "acme", "a\nb", "x\nACME\ny", None],
    }), id_col="id"))
    query = session.sql(
        "SELECT r.id FROM names r "
        "WHERE AI_FILTER(PROMPT('q1: {0}', r.review)) "
        f"AND {function}(r.name, '{pattern}', '{flags}')", dialect="snowflake")
    result = _run(query, make_executor({"r": {"q1:": [1] * 5}}))
    assert result.to_rows() == [(i,) for i in expected]


@pytest.mark.parametrize("flags", ["'invalid'", "1", "NULL", "r.lang"])
def test_regex_flags_refuse_invalid_or_nonliteral_arguments(session, flags):
    with pytest.raises(CompileError, match="flags|literal"):
        session.sql(
            "SELECT r.id FROM reviews r "
            "WHERE AI_FILTER(PROMPT('q1: {0}', r.review)) "
            f"AND REGEXP_LIKE(r.lang, 'en', {flags})", dialect="snowflake")


@pytest.mark.parametrize("literals,expected", [
    ((None,), []), ((None, 5), ["r0", "r5"]),
])
def test_in_never_accepts_null_source_values(session, literals, expected):
    query = (session.docs("reviews").alias("r")
             .where(col("r.stars").isin(literals))
             .ai_filter(prompt("q1: {0}", col("r.review")))
             .select("r.id"))
    result = _run(query, make_executor({"r": {"q1:": [1] * 6}}))
    assert result.to_rows() == [(document,) for document in expected]
