"""DataFusion SQL shapes keep their semantics around model calls.

The model always answers one truth value, regardless of the
region's generated aliases or its materialized input's row numbers. Expected
rows come from the SQL and the small source tables, not another Quail plan.
Both truth values exercise retained and rejected rows, including outer joins.
"""

from contextlib import contextmanager

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
        # a row the join fills with NULL has a NULL answer, which is not TRUE
        "left join + AI on right in WHERE": [("r1", "x"), ("r2", "y")] if truth else [],
        "full join + AI filter": LEFT_JOIN if truth else [],
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


def constant_models():
    """Yield a context and the truth value its model answers, for each value."""
    for truth in (True, False):
        with quail.Session(CONFIG, tokenizer=TOKENIZER) as session:
            oracle = Oracle(
                filters=lambda *_, truth=truth: truth,
                joins=lambda *_, truth=truth: truth,
                labels=lambda *_: "a",
                scores=lambda *_: 0.5,
            )
            oracle.catalog = session.catalog
            context = quail.SessionContext(
                session, physical_executor=oracle_executor(oracle))
            context.register("r", quail.DocumentProvider.from_table(R, id_col="id"))
            context.register("u", quail.DocumentProvider.from_table(U, id_col="id"))
            yield context, truth


@contextmanager
def planted_filters(oracle, *tables):
    """Yield a session and a context over `tables`, answered by `oracle`."""
    with quail.Session(CONFIG, tokenizer=TOKENIZER) as session:
        oracle.catalog = session.catalog
        context = quail.SessionContext(
            session, physical_executor=oracle_executor(oracle))
        for name, table in zip("ru", tables):
            context.register(
                name, quail.DocumentProvider.from_table(table, id_col="id"))
        yield session, context


def _rows(table):
    """Return the table's rows as tuples, by column position."""
    return list(zip(*(column.to_pylist() for column in table.columns)))


def _plan(context, query, plan_type):
    """Return the `plan_type` text that EXPLAIN prints for `query`."""
    plans = context.sql("EXPLAIN " + query)
    return plans.column("plan").to_pylist()[
        plans.column("plan_type").to_pylist().index(plan_type)]


ORDERED_SHAPES = {
    "CTE", "LIMIT/OFFSET", "ORDER BY AI.SCORE directly", "CREATE TABLE AS",
}


def test_datafusion_sql_shapes_with_model_calls_return_the_rows_the_sql_states():
    for context, truth in constant_models():
        for shape, query in QUERIES.items():
            case = (shape, truth)
            table = context.sql(query)
            if shape == "EXPLAIN":
                assert table.column_names == ["plan_type", "plan"], case
                plans = "\n".join(table.column("plan").to_pylist())
                assert "ModelRegion" in plans, case
                assert "r" in plans, case
                continue
            if shape == "CREATE TABLE AS":
                # Execution must install the resulting table, not merely plan
                # its SELECT.
                table = context.sql("SELECT id FROM t2 ORDER BY id")
            expected = expected_rows(truth)[shape]
            rows = _rows(table)
            if shape in ORDERED_SHAPES:
                assert rows == expected, case
            else:
                assert sorted(rows, key=repr) == sorted(expected, key=repr), case
            if shape == "select AI.IF as column":
                assert table.schema.field("k").type == pa.bool_(), case
            if shape == "select classify no alias":
                assert "classify" in table.column_names[1].lower(), case


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


def test_a_negated_ai_filter_runs_in_one_model_region():
    for shape, (query, expected) in NEGATED.items():
        oracle = Oracle(filters={"r": {"q1:": [1, 0, 1, 0], "q2:": [1, 0, 0, 1]}})
        with planted_filters(oracle, R) as (session, context):
            logical = _plan(context, query, "logical_plan")
            # DataFusion neither filters on the answer nor computes the input rows
            assert "Filter:" not in logical and "SharedInput" not in logical, shape
            oracle.asked.clear()
            assert sorted(context.sql(query).column(0).to_pylist()) == expected, shape
            if shape == "under a LIMIT":
                # the LIMIT is the filter's fetch: the model stops once a row passes
                plan = session.sql(query).explain()
                assert "filter: NOT __quail_answer_1, fetch: 1" in plan, shape
            if shape == "in an OR":
                # r0 passes on its stars, so the model is not asked about it
                assert ("r", 0) not in oracle.asked, shape


def test_an_ai_filter_runs_below_an_outer_join_on_the_side_it_keeps():
    kept = [("r0", None), ("r2", "y")]
    for join in ("r LEFT JOIN u ON r.id = u.id", "u RIGHT JOIN r ON r.id = u.id"):
        oracle = Oracle(filters={"r": {"q1:": [1, 0, 1, 0]}})
        query = f"SELECT r.id, u.name FROM {join} WHERE {PREDICATE}"
        with planted_filters(oracle, R, U) as (_, context):
            physical = _plan(context, query, "physical_plan")
            # the model reads the table's rows, not the joined rows DataFusion
            # computes
            assert "ModelRegionExec" in physical, join
            assert "SharedInputExec" not in physical, join
            rows = _rows(context.sql(query))
            assert sorted(rows, key=repr) == sorted(kept, key=repr), join
            assert sorted(oracle.asked) == [("r", row) for row in range(4)], join


def test_a_test_of_two_calls_answers_runs_above_one_region():
    query = REGION_BOUNDARY_QUERIES["OR mixes classification and extraction results"]
    for context, truth in constant_models():
        physical = _plan(context, query, "physical_plan")
        # both calls run in one region; DataFusion tests their answers above it
        assert physical.count("ModelRegionExec") == 1, truth
        assert "SharedInputExec" not in physical and "FilterExec" in physical, truth
        ids = sorted(context.sql(query).column(0).to_pylist())
        assert ids == ["r0", "r1", "r2", "r3"], truth


def test_a_limit_over_an_or_filter_stays_above_the_region():
    oracle = Oracle(filters={"r": {"q1:": [1, 0, 1, 0]}})
    query = f"SELECT r.id FROM r WHERE {PREDICATE} OR r.stars > 3 LIMIT 10"
    with planted_filters(oracle, R) as (_, context):
        logical = _plan(context, query, "logical_plan")
        # DataFusion limits the rows; the region keeps the OR and its skipping
        assert "Limit:" in logical and "SharedInput" not in logical
        assert sorted(context.sql(query).column(0).to_pylist()) == ["r0", "r2", "r3"]
        assert ("r", 3) not in oracle.asked


def test_an_ai_filter_that_is_also_selected_asks_the_model_once():
    for negated in (False, True):
        oracle = Oracle(filters={"r": {"q1:": [1, 0, 1, 0]}})
        test = f"NOT {PREDICATE}" if negated else PREDICATE
        with planted_filters(oracle, R) as (_, context):
            table = context.sql(f"SELECT r.id, {PREDICATE} AS k FROM r WHERE {test}")
            ids = ["r1", "r3"] if negated else ["r0", "r2"]
            assert sorted(table.to_pylist(), key=repr) == [
                {"r.id": found, "k": not negated} for found in ids], negated
            assert sorted(oracle.asked) == [("r", row) for row in range(4)], negated


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


def test_valid_expressions_shrink_model_regions():
    for context, truth in constant_models():
        expected = {
            "score compared with a numeric expression": [("r0",)],
            "score threshold computed from classification": [("r0",)],
            "AND mixes score and classification": IDS,
            "OR mixes score and classification": [],
            "predicates read different document columns": IDS if truth else [],
            "OR reads a classification result": [("r3",)],
            # The oracle's extraction returns NULL for every document.
            "OR reads an extraction result": [("r3",)],
            "classification alias matches the table alias": [
                (row[0], "a") for row in IDS
            ],
            "OR mixes classification and extraction results": IDS,
        }
        for shape, query in REGION_BOUNDARY_QUERIES.items():
            rows = _rows(context.sql(query))
            assert sorted(rows, key=repr) == sorted(expected[shape], key=repr), (
                shape, truth)


FALLBACK_COLUMNS = [
    (f"{SCORE} AS score", (0.5,)),
    ("AI.EXTRACT_SPAN(r.body, {'answer': 'Copy the document.'})"
     "['answer']['start'] AS start", (None,)),
    (f"{CLASSIFY} AS c, {CLASSIFY} AS d", ("a", "a")),
    (f"{SCORE} AS score, {CLASSIFY} AS c", (0.5, "a")),
    ("AI.CLASSIFY(PROMPT('Rank {0}', CAST(ROW_NUMBER() OVER (ORDER BY r.id) "
     "AS VARCHAR)), ['a', 'b']) AS c", ("a",)),
    ("AI.CLASSIFY(PROMPT('Compare {0} and {1}', r.body, "
     "CAST(r.stars AS VARCHAR)), ['a', 'b']) AS c", ("a",)),
    (f"AI.CLASSIFY(PROMPT('Read {{0}}', UPPER({CLASSIFY})), ['a', 'b']) AS c",
     ("a",)),
]


def test_fallback_columns_are_composable():
    for context, truth in constant_models():
        for expression, values in FALLBACK_COLUMNS:
            table = context.sql(f"SELECT r.id, {expression} FROM r ORDER BY r.id")
            assert _rows(table) == [(row[0], *values) for row in IDS], (
                expression, truth)


def test_fallback_preserves_hint_semantics_and_validates_options():
    for context, truth in constant_models():
        table = context.sql(
            f"SELECT CASE WHEN HINT({PREDICATE}, {{'selectivity': 0.5}}) "
            "THEN 1 ELSE 0 END AS answer FROM r"
        )
        assert table.to_pylist() == [{"answer": int(truth)}] * 4, truth
        with pytest.raises(RuntimeError, match="Unknown option"):
            context.sql(
                f"SELECT NOT HINT({PREDICATE}, {{'unknown': 1}}) AS answer FROM r"
            )


def test_fallback_input_runs_the_native_optimizer_before_early_claim():
    for context, truth in constant_models():
        table = context.sql(
            f"SELECT NOT {PREDICATE} AS negative FROM r JOIN u ON {PAIR} AND {PAIR}"
        )
        expected = [{"negative": False}] * 12 if truth else []
        assert table.to_pylist() == expected, truth


def test_a_selected_ai_alias_can_match_an_input_column():
    for context, truth in constant_models():
        table = context.sql(
            f"SELECT {PREDICATE} AS body, NOT {PREDICATE} AS negative FROM r"
        )
        expected = [{"body": truth, "negative": not truth}] * 4
        assert table.to_pylist() == expected, truth


def test_fallback_classification_omits_unrequested_probabilities():
    for context, truth in constant_models():
        table = context.sql(
            "SELECT UPPER(AI.CLASSIFY(PROMPT('Tone {0}', r.body), ['a', 'b'], "
            "{'probabilities': true})) AS label FROM r"
        )
        assert table.to_pylist() == [{"label": "A"}] * 4, truth


def test_uncorrelated_exists_keeps_sql_boolean_semantics():
    for context, truth in constant_models():
        for negated in (False, True):
            keyword = "NOT EXISTS" if negated else "EXISTS"
            table = context.sql(
                f"SELECT u.id FROM u WHERE {keyword} "
                f"(SELECT 1 FROM r WHERE NOT {PREDICATE}) ORDER BY u.id"
            )
            expected = ["r1", "r2", "u9"] if (truth == negated) else []
            assert table.column(0).to_pylist() == expected, (keyword, truth)


def test_values_uses_the_same_model_column_fallback():
    row = "(AI.IF(PROMPT('Check {0}', 'hello')))"
    for context, truth in constant_models():
        for num_rows in (1, 2):
            table = context.sql("VALUES " + ", ".join([row] * num_rows))
            assert table.column(0).to_pylist() == [truth] * num_rows, (
                num_rows, truth)


def test_user_row_column_is_not_a_generated_identity():
    source = pa.table({"id": ["a", "b"], "body": ["x", "y"], "__quail_row": [1, 1]})
    for context, truth in constant_models():
        context.register("named_rows",
                         quail.DocumentProvider.from_table(source, id_col="id"))
        table = context.sql(
            "SELECT n.id, n.__quail_row FROM named_rows n "
            "WHERE AI.IF(PROMPT('Keep {0}', n.body)) ORDER BY n.id"
        )
        assert _rows(table) == ([("a", 1), ("b", 1)] if truth else []), truth
        table = context.sql(
            "SELECT g.__quail_row FROM (SELECT r.stars AS __quail_row, "
            "STRING_AGG(r.body, ' ') AS body FROM r GROUP BY r.stars) g "
            "WHERE AI.IF(PROMPT('Keep {0}', g.body)) ORDER BY g.__quail_row"
        )
        assert table.column(0).to_pylist() == ([1, 2, 3, 4] if truth else []), truth


def test_generated_names_do_not_shadow_user_columns():
    for context, truth in constant_models():
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
            {"call_value": f"r{index}", "argument_value": body,
             "row_value": index + 1, "answer": int(truth)}
            for index, body in enumerate(R.column("body").to_pylist())
        ], truth


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
