"""DataFusion SQL shapes keep their semantics around model calls.

The model always answers the parametrized truth value, regardless of the
region's generated aliases or its materialized input's row numbers. Expected
rows come from the SQL and the small source tables, not another Quail plan.
Both truth values exercise retained and rejected rows, including outer joins.
"""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, oracle_executor
from helpers.sessions import CONFIG, TOKENIZER

import quail

pytest.importorskip("_quail_native")

R = pa.table({
    "id": ["r0", "r1", "r2", "r3"],
    "body": ["a b", "c d", "e f", "g h"],
    "stars": [1, 2, 3, 4],
})
U = pa.table({"id": ["r1", "r2", "u9"], "name": ["x", "y", "z"]})

PROMPT = "PROMPT('q1: {0}', r.body)"
PREDICATE = f"AI.IF({PROMPT})"
CLASSIFY = f"AI.CLASSIFY({PROMPT}, ['a', 'b'])"
SCORE = f"AI.SCORE({PROMPT})"
PAIR = "AI.IF(PROMPT('q1: {0} {1}', r.body, u.name))"

# Keep the same 32 shapes as the compatibility investigation. The score ORDER
# BY has a secondary key because the oracle assigns every row the same score.
QUERIES = {
    "where filter": f"SELECT r.id FROM r WHERE {PREDICATE}",
    "select AI.IF as column": f"SELECT r.id, {PREDICATE} AS k FROM r",
    "select classify no alias": f"SELECT r.id, {CLASSIFY} FROM r",
    "select expression of AI": f"SELECT r.id, UPPER({CLASSIFY}) AS c FROM r",
    "CASE WHEN AI.IF": (
        f"SELECT r.id, CASE WHEN {PREDICATE} THEN 'y' ELSE 'n' END AS k FROM r"
    ),
    "NOT AI.IF": f"SELECT r.id FROM r WHERE NOT {PREDICATE}",
    "OR with relational": f"SELECT r.id FROM r WHERE {PREDICATE} OR r.stars > 3",
    "ORDER BY AI.SCORE directly": (
        f"SELECT r.id FROM r ORDER BY {SCORE} DESC, r.id LIMIT 2"
    ),
    "GROUP BY classify": (
        f"SELECT {CLASSIFY} AS c, COUNT(*) AS n FROM r GROUP BY c"
    ),
    "COUNT FILTER (WHERE AI)": (
        f"SELECT COUNT(*) FILTER (WHERE {PREDICATE}) AS n FROM r"
    ),
    "SUM(CASE AI)": (
        f"SELECT SUM(CASE WHEN {PREDICATE} THEN 1 ELSE 0 END) AS n FROM r"
    ),
    "HAVING with AI agg": (
        "SELECT r.stars, COUNT(*) AS n FROM r GROUP BY r.stars "
        f"HAVING SUM(CASE WHEN {PREDICATE} THEN 1 ELSE 0 END) > 0"
    ),
    "window over AI score": (
        f"SELECT r.id, RANK() OVER (ORDER BY {SCORE} DESC) AS k FROM r"
    ),
    "inner join + AI filter": (
        f"SELECT r.id, u.name FROM r JOIN u ON r.id = u.id WHERE {PREDICATE}"
    ),
    "left join + AI filter on left": (
        f"SELECT r.id, u.name FROM r LEFT JOIN u ON r.id = u.id WHERE {PREDICATE}"
    ),
    "left join + AI on right in WHERE": (
        "SELECT r.id, u.name FROM r LEFT JOIN u ON r.id = u.id "
        "WHERE AI.IF(PROMPT('q1: {0}', u.name))"
    ),
    "full join + AI filter": (
        f"SELECT r.id, u.name FROM r FULL JOIN u ON r.id = u.id WHERE {PREDICATE}"
    ),
    "left join ON AI": f"SELECT r.id, u.name FROM r LEFT JOIN u ON {PAIR}",
    "full join ON AI": f"SELECT r.id, u.name FROM r FULL JOIN u ON {PAIR}",
    "cross join + AI pair filter": (
        f"SELECT r.id, u.name FROM r CROSS JOIN u WHERE {PAIR}"
    ),
    "IN subquery with AI": (
        f"SELECT u.id FROM u WHERE u.id IN (SELECT r.id FROM r WHERE {PREDICATE})"
    ),
    "EXISTS correlated AI": (
        "SELECT u.id FROM u WHERE EXISTS "
        f"(SELECT 1 FROM r WHERE r.id = u.id AND {PREDICATE})"
    ),
    "scalar subquery count AI": (
        f"SELECT (SELECT COUNT(*) FROM r WHERE {PREDICATE}) AS n"
    ),
    "CTE": (
        f"WITH k AS (SELECT r.id FROM r WHERE {PREDICATE}) "
        "SELECT * FROM k ORDER BY id"
    ),
    "UNION of two AI selects": (
        f"SELECT r.id FROM r WHERE {PREDICATE} "
        "UNION SELECT r.id FROM r WHERE r.stars > 3"
    ),
    "AI over subquery expression": (
        "SELECT s.id FROM (SELECT r.id, CONCAT(r.body, '!') AS b FROM r) s "
        "WHERE AI.IF(PROMPT('q1: {0}', s.b))"
    ),
    "AI over aggregated text": (
        "SELECT g.s FROM (SELECT r.stars AS s, STRING_AGG(r.body, ' ') AS t "
        "FROM r GROUP BY r.stars) g WHERE AI.IF(PROMPT('q1: {0}', g.t))"
    ),
    "AI over literal": "SELECT AI.IF(PROMPT('q1: {0}', 'hello')) AS k",
    "DISTINCT classify": f"SELECT DISTINCT {CLASSIFY} AS c FROM r ORDER BY 1",
    "LIMIT/OFFSET": (
        f"SELECT r.id FROM r WHERE {PREDICATE} ORDER BY r.id LIMIT 1 OFFSET 1"
    ),
    "EXPLAIN": f"EXPLAIN SELECT r.id FROM r WHERE {PREDICATE}",
    "CREATE TABLE AS": f"CREATE TABLE t2 AS SELECT r.id FROM r WHERE {PREDICATE}",
}

IDS = [("r0",), ("r1",), ("r2",), ("r3",)]
LEFT_JOIN = [("r0", None), ("r1", "x"), ("r2", "y"), ("r3", None)]
PAIRS = [(row[0], name) for row in IDS for name in ("x", "y", "z")]
LEFT_UNMATCHED = [(row[0], None) for row in IDS]
RIGHT_UNMATCHED = [(None, name) for name in ("x", "y", "z")]


def expected_rows(truth):
    """Evaluate the fixed relational examples with a constant model predicate."""
    kept = IDS if truth else []
    return {
        "where filter": kept,
        "select AI.IF as column": [(row[0], truth) for row in IDS],
        "select classify no alias": [(row[0], "a") for row in IDS],
        "select expression of AI": [(row[0], "A") for row in IDS],
        "CASE WHEN AI.IF": [(row[0], "y" if truth else "n") for row in IDS],
        "NOT AI.IF": [] if truth else IDS,
        "OR with relational": IDS if truth else [("r3",)],
        "ORDER BY AI.SCORE directly": [("r0",), ("r1",)],
        "GROUP BY classify": [("a", 4)],
        "COUNT FILTER (WHERE AI)": [(4 if truth else 0,)],
        "SUM(CASE AI)": [(4 if truth else 0,)],
        "HAVING with AI agg": [(stars, 1) for stars in range(1, 5)] if truth else [],
        "window over AI score": [(row[0], 1) for row in IDS],
        "inner join + AI filter": [("r1", "x"), ("r2", "y")] if truth else [],
        "left join + AI filter on left": LEFT_JOIN if truth else [],
        "left join + AI on right in WHERE": LEFT_JOIN if truth else [],
        "full join + AI filter": LEFT_JOIN + [(None, "z")] if truth else [],
        "left join ON AI": PAIRS if truth else LEFT_UNMATCHED,
        "full join ON AI": PAIRS if truth else LEFT_UNMATCHED + RIGHT_UNMATCHED,
        "cross join + AI pair filter": PAIRS if truth else [],
        "IN subquery with AI": [("r1",), ("r2",)] if truth else [],
        "EXISTS correlated AI": [("r1",), ("r2",)] if truth else [],
        "scalar subquery count AI": [(4 if truth else 0,)],
        "CTE": kept,
        "UNION of two AI selects": IDS if truth else [("r3",)],
        "AI over subquery expression": kept,
        "AI over aggregated text": [(stars,) for stars in range(1, 5)] if truth else [],
        "AI over literal": [(truth,)],
        "DISTINCT classify": [("a",)],
        "LIMIT/OFFSET": [("r1",)] if truth else [],
        "CREATE TABLE AS": kept,
    }


@pytest.fixture(params=[True, False], ids=["true", "false"])
def constant_model(request):
    truth = request.param
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
    oracle = Oracle(
        filters=lambda *_: truth,
        joins=lambda *_: truth,
        labels=lambda *_: "a",
        scores=lambda *_: 0.5,
    )
    oracle.catalog = session.catalog
    context = quail.SessionContext(session, physical_executor=oracle_executor(oracle))
    context.register("r", quail.DocumentProvider.from_table(R, id_col="id"))
    context.register("u", quail.DocumentProvider.from_table(U, id_col="id"))
    yield context, truth
    session.close()


@pytest.mark.parametrize("shape", QUERIES)
def test_datafusion_sql_shapes_with_model_calls(constant_model, shape):
    context, truth = constant_model
    table = context.sql(QUERIES[shape])
    if shape == "EXPLAIN":
        assert table.column_names == ["plan_type", "plan"]
        plans = "\n".join(table.column("plan").to_pylist())
        assert "ModelRegion" in plans
        assert "r" in plans
        return
    if shape == "CREATE TABLE AS":
        # Execution must install the resulting table, not merely plan its SELECT.
        table = context.sql("SELECT id FROM t2 ORDER BY id")
    expected = expected_rows(truth)[shape]
    rows = list(zip(*(column.to_pylist() for column in table.columns)))
    if shape in {
        "CTE", "LIMIT/OFFSET", "ORDER BY AI.SCORE directly", "CREATE TABLE AS",
    }:
        assert rows == expected
    else:
        assert sorted(rows, key=repr) == sorted(expected, key=repr)
    if shape == "select AI.IF as column":
        assert table.schema.field("k").type == pa.bool_()
    if shape == "select classify no alias":
        assert "classify" in table.column_names[1].lower()


NEGATED = {
    "alone": (f"SELECT r.id FROM r WHERE NOT {PREDICATE}", ["r1", "r3"]),
    "compared with FALSE": (
        f"SELECT r.id FROM r WHERE {PREDICATE} = FALSE", ["r1", "r3"]),
    "beside a second AI.IF in an OR": (
        f"SELECT r.id FROM r WHERE NOT {PREDICATE} "
        "OR AI.IF(PROMPT('q2: {0}', r.body))",
        ["r0", "r1", "r3"],
    ),
    "after another AI filter": (
        "SELECT r.id FROM r WHERE AI.IF(PROMPT('q2: {0}', r.body)) "
        f"AND NOT {PREDICATE}",
        ["r3"],
    ),
    "in an OR": (f"SELECT r.id FROM r WHERE NOT {PREDICATE} OR r.stars = 1",
                 ["r0", "r1", "r3"]),
    "under a LIMIT": (f"SELECT r.id FROM r WHERE NOT {PREDICATE} LIMIT 1", ["r1"]),
}


@pytest.mark.parametrize("shape", NEGATED)
def test_a_negated_ai_filter_runs_in_one_model_region(shape):
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
    oracle = Oracle(filters={"r": {"q1:": [1, 0, 1, 0], "q2:": [1, 0, 0, 1]}})
    oracle.catalog = session.catalog
    context = quail.SessionContext(session, physical_executor=oracle_executor(oracle))
    context.register("r", quail.DocumentProvider.from_table(R, id_col="id"))
    query, expected = NEGATED[shape]
    try:
        plans = context.sql("EXPLAIN " + query)
        logical = plans.column("plan").to_pylist()[
            plans.column("plan_type").to_pylist().index("logical_plan")]
        # DataFusion neither filters on the answer nor computes the input rows
        assert "Filter:" not in logical and "SharedInput" not in logical
        oracle.asked.clear()
        assert sorted(context.sql(query).column(0).to_pylist()) == expected
        if shape == "under a LIMIT":
            # the LIMIT is the filter's fetch: the model stops once a row passes
            plan = session.sql(query).explain()
            assert "filter: NOT __quail_answer_1, fetch: 1" in plan
        if shape == "in an OR":
            # r0 passes on its stars, so the model is not asked about it
            assert ("r", 0) not in oracle.asked
    finally:
        session.close()


@pytest.mark.parametrize("negated", [False, True])
def test_an_ai_filter_that_is_also_selected_asks_the_model_once(negated):
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
    oracle = Oracle(filters={"r": {"q1:": [1, 0, 1, 0]}})
    oracle.catalog = session.catalog
    context = quail.SessionContext(session, physical_executor=oracle_executor(oracle))
    context.register("r", quail.DocumentProvider.from_table(R, id_col="id"))
    test = f"NOT {PREDICATE}" if negated else PREDICATE
    try:
        table = context.sql(f"SELECT r.id, {PREDICATE} AS k FROM r WHERE {test}")
        ids = ["r1", "r3"] if negated else ["r0", "r2"]
        assert sorted(table.to_pylist(), key=repr) == [
            {"r.id": found, "k": not negated} for found in ids]
        assert sorted(oracle.asked) == [("r", row) for row in range(4)]
    finally:
        session.close()


REGION_BOUNDARY_QUERIES = {
    "score compared with a numeric expression": (
        f"SELECT r.id FROM r WHERE {SCORE} > r.stars / 4.0"
    ),
    "score threshold computed from classification": (
        f"SELECT r.id FROM r WHERE {SCORE} > "
        f"CASE WHEN {CLASSIFY} = 'a' THEN r.stars / 4.0 ELSE 0 END"
    ),
    "AND mixes score and classification": (
        f"SELECT r.id FROM r WHERE {SCORE} > 0.4 AND {CLASSIFY} = 'a'"
    ),
    "OR mixes score and classification": (
        f"SELECT r.id FROM r WHERE {SCORE} > 0.6 OR {CLASSIFY} = 'b'"
    ),
    "predicates read different document columns": (
        f"SELECT r.id FROM r WHERE {PREDICATE} "
        "AND AI.IF(PROMPT('q2: {0}', CAST(r.stars AS VARCHAR)))"
    ),
    "OR reads a classification result": (
        f"SELECT r.id FROM r WHERE {CLASSIFY} = 'b' OR r.stars > 3"
    ),
    "OR reads an extraction result": (
        "SELECT r.id FROM r WHERE "
        "AI.EXTRACT(r.body, {'answer': 'Copy the document.'})['answer'] "
        "IS NOT NULL OR r.stars > 3"
    ),
    "classification alias matches the table alias": (
        f"SELECT r.id, {CLASSIFY} AS r FROM r"
    ),
    "OR mixes classification and extraction results": (
        f"SELECT r.id FROM r WHERE {CLASSIFY} = 'b' OR "
        "AI.EXTRACT(r.body, {'answer': 'Copy the document.'})['answer'] IS NULL"
    ),
}


@pytest.mark.parametrize("shape", REGION_BOUNDARY_QUERIES)
def test_valid_expressions_shrink_model_regions(constant_model, shape):
    context, truth = constant_model
    table = context.sql(REGION_BOUNDARY_QUERIES[shape])
    expected = {
        "score compared with a numeric expression": [("r0",)],
        "score threshold computed from classification": [("r0",)],
        "AND mixes score and classification": IDS,
        "OR mixes score and classification": [],
        "predicates read different document columns": IDS if truth else [],
        "OR reads a classification result": [("r3",)],
        # The fixture's extraction oracle returns NULL for every document.
        "OR reads an extraction result": [("r3",)],
        "classification alias matches the table alias": [
            (row[0], "a") for row in IDS
        ],
        "OR mixes classification and extraction results": IDS,
    }[shape]
    rows = list(zip(*(column.to_pylist() for column in table.columns)))
    assert sorted(rows, key=repr) == sorted(expected, key=repr)


@pytest.mark.parametrize("expression, expected", [
    (f"{SCORE} AS score", 0.5),
    ("AI.EXTRACT_SPAN(r.body, {'answer': 'Copy the document.'})"
     "['answer']['start'] AS start", None),
    (f"{CLASSIFY} AS c, {CLASSIFY} AS d", ("a", "a")),
    (f"{SCORE} AS score, {CLASSIFY} AS c", (0.5, "a")),
    ("AI.CLASSIFY(PROMPT('Rank {0}', CAST(ROW_NUMBER() OVER (ORDER BY r.id) "
     "AS VARCHAR)), ['a', 'b']) AS c", "a"),
    ("AI.CLASSIFY(PROMPT('Compare {0} and {1}', r.body, "
     "CAST(r.stars AS VARCHAR)), ['a', 'b']) AS c", "a"),
    (f"AI.CLASSIFY(PROMPT('Read {{0}}', UPPER({CLASSIFY})), ['a', 'b']) AS c", "a"),
])
def test_fallback_columns_are_composable(constant_model, expression, expected):
    context, _ = constant_model
    table = context.sql(f"SELECT r.id, {expression} FROM r ORDER BY r.id")
    values = expected if isinstance(expected, tuple) else (expected,)
    assert list(zip(*(column.to_pylist() for column in table.columns))) == [
        (row[0], *values) for row in IDS
    ]


def test_fallback_preserves_hint_semantics_and_validates_options(constant_model):
    context, truth = constant_model
    table = context.sql(
        f"SELECT CASE WHEN HINT({PREDICATE}, {{'selectivity': 0.5}}) "
        "THEN 1 ELSE 0 END AS answer FROM r"
    )
    assert table.to_pylist() == [{"answer": int(truth)}] * 4
    with pytest.raises(RuntimeError, match="Unknown option"):
        context.sql(
            f"SELECT NOT HINT({PREDICATE}, {{'unknown': 1}}) AS answer FROM r"
        )


def test_fallback_input_runs_the_native_optimizer_before_early_claim(constant_model):
    context, truth = constant_model
    table = context.sql(
        f"SELECT NOT {PREDICATE} AS negative FROM r JOIN u ON {PAIR} AND {PAIR}"
    )
    assert table.to_pylist() == ([{"negative": False}] * 12 if truth else [])


def test_a_selected_ai_alias_can_match_an_input_column(constant_model):
    context, truth = constant_model
    table = context.sql(
        f"SELECT {PREDICATE} AS body, NOT {PREDICATE} AS negative FROM r"
    )
    assert table.to_pylist() == [{"body": truth, "negative": not truth}] * 4


def test_fallback_classification_omits_unrequested_probabilities(constant_model):
    context, _ = constant_model
    table = context.sql(
        "SELECT UPPER(AI.CLASSIFY(PROMPT('Tone {0}', r.body), ['a', 'b'], "
        "{'probabilities': true})) AS label FROM r"
    )
    assert table.to_pylist() == [{"label": "A"}] * 4


@pytest.mark.parametrize("negated", [False, True])
def test_uncorrelated_exists_keeps_sql_boolean_semantics(constant_model, negated):
    context, truth = constant_model
    keyword = "NOT EXISTS" if negated else "EXISTS"
    table = context.sql(
        f"SELECT u.id FROM u WHERE {keyword} "
        f"(SELECT 1 FROM r WHERE NOT {PREDICATE}) ORDER BY u.id"
    )
    assert table.column(0).to_pylist() == (
        ["r1", "r2", "u9"] if (truth == negated) else []
    )


@pytest.mark.parametrize("row_count", [1, 2])
def test_values_uses_the_same_model_column_fallback(constant_model, row_count):
    context, truth = constant_model
    row = "(AI.IF(PROMPT('Check {0}', 'hello')))"
    table = context.sql("VALUES " + ", ".join([row] * row_count))
    assert table.column(0).to_pylist() == [truth] * row_count


def test_user_row_column_is_not_a_generated_identity(constant_model):
    context, truth = constant_model
    source = pa.table({"id": ["a", "b"], "body": ["x", "y"], "__quail_row": [1, 1]})
    context.register("named_rows",
                     quail.DocumentProvider.from_table(source, id_col="id"))
    table = context.sql(
        "SELECT n.id, n.__quail_row FROM named_rows n "
        "WHERE AI.IF(PROMPT('Keep {0}', n.body)) ORDER BY n.id"
    )
    assert list(zip(*(column.to_pylist() for column in table.columns))) == (
        [("a", 1), ("b", 1)] if truth else []
    )
    table = context.sql(
        "SELECT g.__quail_row FROM (SELECT r.stars AS __quail_row, "
        "STRING_AGG(r.body, ' ') AS body FROM r GROUP BY r.stars) g "
        "WHERE AI.IF(PROMPT('Keep {0}', g.body)) ORDER BY g.__quail_row"
    )
    assert table.column(0).to_pylist() == ([1, 2, 3, 4] if truth else [])


def test_join_back_identity_does_not_shadow_a_selected_answer(constant_model):
    context, _ = constant_model
    table = context.sql(
        f"SELECT r.id, r.stars, {CLASSIFY} AS __quail_key_r FROM r ORDER BY r.id"
    )
    assert list(zip(*(column.to_pylist() for column in table.columns))) == [
        (f"r{i}", i + 1, "a") for i in range(4)
    ]


def test_generated_names_do_not_shadow_user_columns(constant_model):
    context, truth = constant_model
    table = context.sql(
        "SELECT s.__quail_call_1 AS call_value, "
        "s.__quail_argument_0 AS argument_value, "
        "s.__quail_row AS row_value, "
        "CASE WHEN AI.IF(PROMPT('Keep {0}', CONCAT(s.__quail_expr_0, '!'))) "
        "THEN 1 ELSE 0 END AS answer FROM "
        "(SELECT id AS __quail_call_1, body AS __quail_argument_0, "
        "stars AS __quail_row, body AS __quail_expr_0 FROM r) s "
        "ORDER BY s.__quail_call_1"
    )
    assert table.to_pylist() == [
        {"call_value": f"r{i}", "argument_value": body,
         "row_value": i + 1, "answer": int(truth)}
        for i, body in enumerate(R.column("body").to_pylist())
    ]


def test_datafusion_computes_the_length_of_a_model_feature_list():
    # The main fixture's model has no SAE; only this case needs the SAE model.
    from quail.api.config import EngineConfig

    session = quail.Session(
        EngineConfig(model="qwen3.5-2b-sae-bf16", device="h100-sxm"),
        tokenizer=TOKENIZER,
    )
    oracle = Oracle()

    def features(spec, document):
        # A staged relation can reorder rows; plant truth by document content,
        # not by the stage's local positional identity.
        text = str(oracle.texts[spec.aliases[0]][document])
        count = R.column("body").to_pylist().index(text) + 1
        return [{"feature": feature, "strength": 1.0} for feature in range(count)]

    oracle.features = features
    oracle.catalog = session.catalog
    context = quail.SessionContext(session, physical_executor=oracle_executor(oracle))
    context.register("r", quail.DocumentProvider.from_table(R, id_col="id"))
    try:
        table = context.sql(
            "SELECT r.id AS document_id, array_length(AI.SAE_FEATURES("
            "PROMPT('Topic? {0}', r.body), {'pooling': 'last'})) AS n "
            "FROM r ORDER BY r.id"
        )
        assert table.to_pylist() == [
            {"document_id": "r0", "n": 1}, {"document_id": "r1", "n": 2},
            {"document_id": "r2", "n": 3}, {"document_id": "r3", "n": 4},
        ]
    finally:
        session.close()
