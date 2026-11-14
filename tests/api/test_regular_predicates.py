"""Column predicates: parsed in WHERE and ON, applied before any model call.

DataFusion runs every plain test before the model region, so the model
is asked only about the rows it passes. Each transcript test's
transcript shows the plan, the questions the model
answered, and the rows (helpers/transcript.py).
"""

import re

import pyarrow as pa
import pytest
from helpers.conditions import condition
from helpers.planning import operators_of, row_tests
from helpers.sessions import CONFIG, TOKENIZER, TRUTH, _run, make_executor
from helpers.transcript import Transcripts, assert_transcript

import quail
from quail.common.expressions import CompileError
from quail.common.selection import selected_rows
from quail.physical import decode_plan_value

# q1 passes r0, r1, r3, and r4; q2 passes r0, r2, r3, and r5
REVIEWS = pa.table({
    "id": [f"r{index}" for index in range(6)],
    "review": [f"b{index}" for index in range(6)],
    "stars": [5, 2, 4, None, 3, 5],
    "lang": ["en", "en", "fr", "en", "en", "de"],
})
PRODUCTS = pa.table({"asin": ["p0", "p1"], "description": ["d0", "d1"],
                     "price": [10, 20], "stars": [5, 2]})
ITEMS = pa.table({"asin": ["p0", "p1"], "description": ["d0", "d1"],
                  "price": [3, 4]})
PAIRS = pa.table({
    "id": [f"r{index}" for index in range(6)],
    "review": [f"b{index}" for index in range(6)],
    "stars": [5, 2, 4, None, 3, 5],
    "floor": [4, 3, 4, 1, 1, 6],
    "lang": ["EN", "en", "Fr", "EN", "en", "DE"],
})
LANGS = pa.table({"code": ["EN", "FR"], "notes": ["n0", "n1"]})
TABLES = {"reviews": REVIEWS, "products": (PRODUCTS, "asin"),
          "items": (ITEMS, "asin"), "pairs": PAIRS, "langs": (LANGS, "code")}

FILTER_SQL = (
    "SELECT r.id FROM reviews r "
    "WHERE AI_FILTER(PROMPT('q1: {0}', r.review), {'selectivity': 0.5}) "
    "AND AI_FILTER(PROMPT('q2: {0}', r.review), {'selectivity': 0.5})")
FIRST = "FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.review))"
PAIRS_FIRST = "FROM pairs r WHERE AI_FILTER(PROMPT('q1: {0}', r.review))"


def _answer(kind, prompt, rows):
    """Every pair joins; q1 and q2 pass the reviews above, by their number."""
    if not prompt.startswith("q"):
        # a join, which the run with every pushdown off asks as a filter
        return True
    index = int(re.sub(r"\D", "", rows[0]))
    return index in ((0, 1, 3, 4) if prompt.startswith("q1") else (0, 2, 3, 5))


def _transcript(*queries) -> str:
    """Transcripts of queries, each SQL or (SQL, keyword arguments)."""
    with Transcripts(TABLES, _answer) as runs:
        return "\n".join(runs.query(query[0], **query[1])
                         if isinstance(query, tuple) else runs.query(query)
                         for query in queries)


@pytest.fixture()
def session():
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
    for name, found in TABLES.items():
        table, id_column = found if isinstance(found, tuple) else (found, "id")
        session.register(
            name, quail.DocumentProvider.from_table(table, id_col=id_column))
    yield session
    session.close()


def test_column_tests_in_where_reject_rows_before_the_model_is_asked():
    assert_transcript(_transcript(
        f"{FILTER_SQL} AND r.stars >= 3 AND r.lang IN ('en', 'fr') "
        "AND r.stars IS NOT NULL",
        f"SELECT r.id, r.stars {FIRST} AND r.stars < 4 AND r.lang = 'en'",
        # a function on a source column
        f"SELECT r.id {FIRST} AND LENGTH(r.lang) > 2",
        # NOT, AND, and OR of one table's columns
        f"{FILTER_SQL} AND NOT (LOWER(r.lang) LIKE '%e%' AND r.stars < 3)",
        f"{FILTER_SQL} AND (r.stars IS NULL OR r.lang = 'de')",
        f"SELECT r.id {FIRST} AND NOT (r.lang = 'fr' OR r.stars IS NULL)"), """\
--- SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.review),
    {'selectivity': 0.5}) AND AI_FILTER(PROMPT('q2: {0}', r.review), {'selectivity':
    0.5}) AND r.stars >= 3 AND r.lang IN ('en', 'fr') AND r.stars IS NOT NULL ---
plan
  __quail_answer_2 = ai_filter('q2: {0}', r.review, named_struct('selectivity',
      Float64(0.5))), filter: __quail_answer_2, returns: []
  __quail_answer_1 = ai_filter('q1: {0}', r.review, named_struct('selectivity',
      Float64(0.5))), filter: __quail_answer_1, returns: []
asked 5 of 5
  'q1: {0}': r0 TRUE, r2 FALSE, r4 TRUE
  'q2: {0}': r0 TRUE, r4 FALSE
rows 1
  r0

--- SELECT r.id, r.stars FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.review)) AND
    r.stars < 4 AND r.lang = 'en' ---
plan
  __quail_answer_1 = ai_filter('q1: {0}', r.review), filter: __quail_answer_1, returns:
      []
asked 2 of 2
  'q1: {0}': r1 TRUE, r4 TRUE
rows 2
  r1, 2
  r4, 3

--- SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.review)) AND
    LENGTH(r.lang) > 2 ---
plan
  __quail_answer_1 = ai_filter('q1: {0}', r.review), filter: __quail_answer_1, returns:
      []
asked 0 of 0
rows 0

--- SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.review),
    {'selectivity': 0.5}) AND AI_FILTER(PROMPT('q2: {0}', r.review), {'selectivity':
    0.5}) AND NOT (LOWER(r.lang) LIKE '%e%' AND r.stars < 3) ---
plan
  __quail_answer_2 = ai_filter('q2: {0}', r.review, named_struct('selectivity',
      Float64(0.5))), filter: __quail_answer_2, returns: []
  __quail_answer_1 = ai_filter('q1: {0}', r.review, named_struct('selectivity',
      Float64(0.5))), filter: __quail_answer_1, returns: []
asked 6 of 6
  'q1: {0}': r0 TRUE, r2 FALSE, r4 TRUE, r5 FALSE
  'q2: {0}': r0 TRUE, r4 FALSE
rows 1
  r0

--- SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.review),
    {'selectivity': 0.5}) AND AI_FILTER(PROMPT('q2: {0}', r.review), {'selectivity':
    0.5}) AND (r.stars IS NULL OR r.lang = 'de') ---
plan
  __quail_answer_2 = ai_filter('q2: {0}', r.review, named_struct('selectivity',
      Float64(0.5))), filter: __quail_answer_2, returns: []
  __quail_answer_1 = ai_filter('q1: {0}', r.review, named_struct('selectivity',
      Float64(0.5))), filter: __quail_answer_1, returns: []
asked 3 of 3
  'q1: {0}': r3 TRUE, r5 FALSE
  'q2: {0}': r3 TRUE
rows 1
  r3

--- SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.review)) AND NOT
    (r.lang = 'fr' OR r.stars IS NULL) ---
plan
  __quail_answer_1 = ai_filter('q1: {0}', r.review), filter: __quail_answer_1, returns:
      []
asked 4 of 4
  'q1: {0}': r0 TRUE, r1 TRUE, r4 TRUE, r5 FALSE
rows 3
  r0
  r1
  r4
""")


def test_where_and_select_take_the_same_datafusion_expressions():
    assert_transcript(_transcript(
        (f"SELECT r.id, REVERSE(UPPER(r.lang)) AS backwards, "
         f"ABS(r.stars - 4) AS gap {FIRST} AND ABS(r.stars - 4) <= 1 "
         "ORDER BY r.id", {"ordered": True}),
        # a test of a function's value
        f"SELECT r.id {PAIRS_FIRST} AND UPPER(r.lang) = r.lang",
        f"SELECT r.id {PAIRS_FIRST} AND NOT (UPPER(r.lang) = r.lang)",
        # two columns of one table; r3's null stars is rejected
        f"SELECT r.id {PAIRS_FIRST} AND r.stars > r.floor",
        f"SELECT r.id {PAIRS_FIRST} AND (r.stars - r.floor >= 1 OR r.lang = 'en')",
        f"SELECT r.id {PAIRS_FIRST} AND ABS(-r.stars) <= r.floor",
        # SELECT returns the same comparison
        f"SELECT r.id, r.stars > r.floor AS above {PAIRS_FIRST}"), """\
--- SELECT r.id, REVERSE(UPPER(r.lang)) AS backwards, ABS(r.stars - 4) AS gap FROM
    reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.review)) AND ABS(r.stars - 4) <= 1
    ORDER BY r.id ---
plan
  __quail_answer_1 = ai_filter('q1: {0}', r.review), filter: __quail_answer_1, returns:
      []
asked 4 of 4
  'q1: {0}': r0 TRUE, r2 FALSE, r4 TRUE, r5 FALSE
rows 2, in order
  r0, 'NE', 1
  r4, 'NE', 1

--- SELECT r.id FROM pairs r WHERE AI_FILTER(PROMPT('q1: {0}', r.review)) AND
    UPPER(r.lang) = r.lang ---
plan
  __quail_answer_1 = ai_filter('q1: {0}', r.review), filter: __quail_answer_1, returns:
      []
asked 3 of 3
  'q1: {0}': r0 TRUE, r3 TRUE, r5 FALSE
rows 2
  r0
  r3

--- SELECT r.id FROM pairs r WHERE AI_FILTER(PROMPT('q1: {0}', r.review)) AND NOT
    (UPPER(r.lang) = r.lang) ---
plan
  __quail_answer_1 = ai_filter('q1: {0}', r.review), filter: __quail_answer_1, returns:
      []
asked 3 of 3
  'q1: {0}': r1 TRUE, r2 FALSE, r4 TRUE
rows 2
  r1
  r4

--- SELECT r.id FROM pairs r WHERE AI_FILTER(PROMPT('q1: {0}', r.review)) AND r.stars >
    r.floor ---
plan
  __quail_answer_1 = ai_filter('q1: {0}', r.review), filter: __quail_answer_1, returns:
      []
asked 2 of 2
  'q1: {0}': r0 TRUE, r4 TRUE
rows 2
  r0
  r4

--- SELECT r.id FROM pairs r WHERE AI_FILTER(PROMPT('q1: {0}', r.review)) AND (r.stars -
    r.floor >= 1 OR r.lang = 'en') ---
plan
  __quail_answer_1 = ai_filter('q1: {0}', r.review), filter: __quail_answer_1, returns:
      []
asked 3 of 3
  'q1: {0}': r0 TRUE, r1 TRUE, r4 TRUE
rows 3
  r0
  r1
  r4

--- SELECT r.id FROM pairs r WHERE AI_FILTER(PROMPT('q1: {0}', r.review)) AND
    ABS(-r.stars) <= r.floor ---
plan
  __quail_answer_1 = ai_filter('q1: {0}', r.review), filter: __quail_answer_1, returns:
      []
asked 3 of 3
  'q1: {0}': r1 TRUE, r2 FALSE, r5 FALSE
rows 1
  r1

--- SELECT r.id, r.stars > r.floor AS above FROM pairs r WHERE AI_FILTER(PROMPT('q1:
    {0}', r.review)) ---
plan
  __quail_answer_1 = ai_filter('q1: {0}', r.review), filter: __quail_answer_1, returns:
      []
asked 6 of 6
  'q1: {0}': r0 TRUE, r1 TRUE, r2 FALSE, r3 TRUE, r4 TRUE, r5 FALSE
rows 4
  r0, TRUE
  r1, FALSE
  r3, NULL
  r4, TRUE
""")


def test_a_regexp_matches_the_whole_value_in_snowflake_and_any_part_in_bigquery():
    bigquery = ("SELECT r.id FROM reviews r "
                "WHERE AI.IF(PROMPT('q1: {{0}}', r.review)) "
                "AND AI.IF(PROMPT('q2: {{0}}', r.review)) AND {}")
    assert_transcript(_transcript(
        # Snowflake matches the whole value: 'e' is not 'en'
        f"{FILTER_SQL} AND REGEXP_LIKE(r.lang, 'e')",
        f"{FILTER_SQL} AND REGEXP_LIKE(r.lang, 'e.')",
        f"{FILTER_SQL} AND RLIKE(r.lang, 'e')",
        f"{FILTER_SQL} AND NOT RLIKE(r.lang, 'e')",
        f"{FILTER_SQL} AND REGEXP_LIKE(r.lang, r.lang || '?')",
        (bigquery.format("REGEXP_CONTAINS(r.lang, 'e')"), {"dialect": "bq"}),
        (bigquery.format("REGEXP_LIKE(r.lang, 'e')"), {"dialect": "bq"})), """\
--- SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.review),
    {'selectivity': 0.5}) AND AI_FILTER(PROMPT('q2: {0}', r.review), {'selectivity':
    0.5}) AND REGEXP_LIKE(r.lang, 'e') ---
plan
  __quail_answer_2 = ai_filter('q2: {0}', r.review, named_struct('selectivity',
      Float64(0.5))), filter: __quail_answer_2, returns: []
  __quail_answer_1 = ai_filter('q1: {0}', r.review, named_struct('selectivity',
      Float64(0.5))), filter: __quail_answer_1, returns: []
asked 0 of 0
rows 0

--- SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.review),
    {'selectivity': 0.5}) AND AI_FILTER(PROMPT('q2: {0}', r.review), {'selectivity':
    0.5}) AND REGEXP_LIKE(r.lang, 'e.') ---
plan
  __quail_answer_2 = ai_filter('q2: {0}', r.review, named_struct('selectivity',
      Float64(0.5))), filter: __quail_answer_2, returns: []
  __quail_answer_1 = ai_filter('q1: {0}', r.review, named_struct('selectivity',
      Float64(0.5))), filter: __quail_answer_1, returns: []
asked 8 of 8
  'q1: {0}': r0 TRUE, r1 TRUE, r3 TRUE, r4 TRUE
  'q2: {0}': r0 TRUE, r1 FALSE, r3 TRUE, r4 FALSE
rows 2
  r0
  r3

--- SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.review),
    {'selectivity': 0.5}) AND AI_FILTER(PROMPT('q2: {0}', r.review), {'selectivity':
    0.5}) AND RLIKE(r.lang, 'e') ---
plan
  __quail_answer_2 = ai_filter('q2: {0}', r.review, named_struct('selectivity',
      Float64(0.5))), filter: __quail_answer_2, returns: []
  __quail_answer_1 = ai_filter('q1: {0}', r.review, named_struct('selectivity',
      Float64(0.5))), filter: __quail_answer_1, returns: []
asked 0 of 0
rows 0

--- SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.review),
    {'selectivity': 0.5}) AND AI_FILTER(PROMPT('q2: {0}', r.review), {'selectivity':
    0.5}) AND NOT RLIKE(r.lang, 'e') ---
plan
  __quail_answer_2 = ai_filter('q2: {0}', r.review, named_struct('selectivity',
      Float64(0.5))), filter: __quail_answer_2, returns: []
  __quail_answer_1 = ai_filter('q1: {0}', r.review, named_struct('selectivity',
      Float64(0.5))), filter: __quail_answer_1, returns: []
asked 10 of 10
  'q1: {0}': r0 TRUE, r1 TRUE, r2 FALSE, r3 TRUE, r4 TRUE, r5 FALSE
  'q2: {0}': r0 TRUE, r1 FALSE, r3 TRUE, r4 FALSE
rows 2
  r0
  r3

--- SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.review),
    {'selectivity': 0.5}) AND AI_FILTER(PROMPT('q2: {0}', r.review), {'selectivity':
    0.5}) AND REGEXP_LIKE(r.lang, r.lang || '?') ---
plan
  __quail_answer_2 = ai_filter('q2: {0}', r.review, named_struct('selectivity',
      Float64(0.5))), filter: __quail_answer_2, returns: []
  __quail_answer_1 = ai_filter('q1: {0}', r.review, named_struct('selectivity',
      Float64(0.5))), filter: __quail_answer_1, returns: []
asked 10 of 10
  'q1: {0}': r0 TRUE, r1 TRUE, r2 FALSE, r3 TRUE, r4 TRUE, r5 FALSE
  'q2: {0}': r0 TRUE, r1 FALSE, r3 TRUE, r4 FALSE
rows 2
  r0
  r3

--- SELECT r.id FROM reviews r WHERE AI.IF(PROMPT('q1: {0}', r.review)) AND
    AI.IF(PROMPT('q2: {0}', r.review)) AND REGEXP_CONTAINS(r.lang, 'e') ---
plan
  __quail_answer_2 = ai_filter('q2: {0}', r.review), filter: __quail_answer_2, returns:
      []
  __quail_answer_1 = ai_filter('q1: {0}', r.review), filter: __quail_answer_1, returns:
      []
asked 9 of 9
  'q1: {0}': r0 TRUE, r1 TRUE, r3 TRUE, r4 TRUE, r5 FALSE
  'q2: {0}': r0 TRUE, r1 FALSE, r3 TRUE, r4 FALSE
rows 2
  r0
  r3

--- SELECT r.id FROM reviews r WHERE AI.IF(PROMPT('q1: {0}', r.review)) AND
    AI.IF(PROMPT('q2: {0}', r.review)) AND REGEXP_LIKE(r.lang, 'e') ---
plan
  __quail_answer_2 = ai_filter('q2: {0}', r.review), filter: __quail_answer_2, returns:
      []
  __quail_answer_1 = ai_filter('q1: {0}', r.review), filter: __quail_answer_1, returns:
      []
asked 9 of 9
  'q1: {0}': r0 TRUE, r1 TRUE, r3 TRUE, r4 TRUE, r5 FALSE
  'q2: {0}': r0 TRUE, r1 FALSE, r3 TRUE, r4 FALSE
rows 2
  r0
  r3
""")


def test_a_where_test_over_two_tables_keeps_the_pairs_it_passes():
    join = "AI_FILTER(PROMPT('x {0} {1}', r.review, p.description))"
    assert_transcript(_transcript(
        # r3's null stars with p1 is NULL OR TRUE, which is true
        f"SELECT r.id, p.asin FROM reviews r JOIN products p ON {join} "
        "WHERE r.stars > 4 OR p.price > 15",
        f"SELECT r.id, p.asin FROM reviews r JOIN items p ON {join} "
        "WHERE r.stars > p.price"), """\
--- SELECT r.id, p.asin FROM reviews r JOIN products p ON AI_FILTER(PROMPT('x {0} {1}',
    r.review, p.description)) WHERE r.stars > 4 OR p.price > 15 ---
plan
  __quail_answer_1 = ai_filter('x {0} {1}', r.review, p.description), filter:
      __quail_answer_1, returns: []
asked 12 of 8
  'x {0} {1}': p0+r0 TRUE, p0+r1 TRUE, p0+r2 TRUE, p0+r3 TRUE, p0+r4 TRUE, p0+r5 TRUE,
      p1+r0 TRUE, p1+r1 TRUE, p1+r2 TRUE, p1+r3 TRUE, p1+r4 TRUE, p1+r5 TRUE
rows 8
  r0, p0
  r0, p1
  r1, p1
  r2, p1
  r3, p1
  r4, p1
  r5, p0
  r5, p1

--- SELECT r.id, p.asin FROM reviews r JOIN items p ON AI_FILTER(PROMPT('x {0} {1}',
    r.review, p.description)) WHERE r.stars > p.price ---
plan
  __quail_answer_1 = ai_filter('x {0} {1}', r.review, p.description), filter:
      __quail_answer_1, returns: []
asked 12 of 5
  'x {0} {1}': p0+r0 TRUE, p0+r1 TRUE, p0+r2 TRUE, p0+r3 TRUE, p0+r4 TRUE, p0+r5 TRUE,
      p1+r0 TRUE, p1+r1 TRUE, p1+r2 TRUE, p1+r3 TRUE, p1+r4 TRUE, p1+r5 TRUE
rows 5
  r0, p0
  r0, p1
  r2, p0
  r5, p0
  r5, p1
""")


def test_an_equality_of_two_tables_in_where_pairs_their_rows_before_the_ai_join():
    join = "AI_FILTER(PROMPT('x {0} {1}', r.review, p.description))"
    assert_transcript(_transcript(
        f"SELECT r.id, p.asin FROM reviews r JOIN products p ON {join} "
        "WHERE r.stars = p.stars",
        f"SELECT r.id, p.asin FROM reviews r, products p "
        f"WHERE r.stars = p.stars AND {join}"), """\
--- SELECT r.id, p.asin FROM reviews r JOIN products p ON AI_FILTER(PROMPT('x {0} {1}',
    r.review, p.description)) WHERE r.stars = p.stars ---
plan
  __quail_answer_1 = ai_filter('x {0} {1}', r.review, p.description), filter:
      __quail_answer_1, returns: []
asked 3 of 3
  'x {0} {1}': p0+r0 TRUE, p0+r5 TRUE, p1+r1 TRUE
rows 3
  r0, p0
  r1, p1
  r5, p0

--- SELECT r.id, p.asin FROM reviews r, products p WHERE r.stars = p.stars AND
    AI_FILTER(PROMPT('x {0} {1}', r.review, p.description)) ---
plan
  __quail_answer_1 = ai_filter('x {0} {1}', r.review, p.description), filter:
      __quail_answer_1, returns: []
asked 3 of 3
  'x {0} {1}': p0+r0 TRUE, p0+r5 TRUE, p1+r1 TRUE
rows 3
  r0, p0
  r1, p1
  r5, p0
""")


SQL_CALLS = (
    "LOG10(r.stars * 20) > 1.7", "SUBSTRING(r.lang, 2, 1)",
    "CONCAT(r.lang, '-', r.lang)",
    "CASE WHEN r.stars > 3 THEN 'high' WHEN r.stars > 0 THEN 'low' END",
    "CAST(r.stars AS DOUBLE)", "r.stars IN (2, 3)",
    # an integer divided by an integer is an integer
    "ROUND(r.stars / 2)",
    # GREATEST skips a NULL
    "GREATEST(r.stars, 3)",
    "r.stars % 2", "STRPOS(r.lang, 'n')", "r.stars BETWEEN 2 AND 3",
)


def test_sql_functions_have_datafusions_meaning_in_every_dialect():
    columns = ", ".join(f"{call} AS v{index}"
                        for index, call in enumerate(SQL_CALLS))
    sql = f"SELECT r.id, {columns} {FIRST}"
    assert_transcript(_transcript((sql, {"dialect": "bq"}), (sql, {})), """\
--- SELECT r.id, LOG10(r.stars * 20) > 1.7 AS v0, SUBSTRING(r.lang, 2, 1) AS v1,
    CONCAT(r.lang, '-', r.lang) AS v2, CASE WHEN r.stars > 3 THEN 'high' WHEN r.stars >
    0 THEN 'low' END AS v3, CAST(r.stars AS DOUBLE) AS v4, r.stars IN (2, 3) AS v5,
    ROUND(r.stars / 2) AS v6, GREATEST(r.stars, 3) AS v7, r.stars % 2 AS v8,
    STRPOS(r.lang, 'n') AS v9, r.stars BETWEEN 2 AND 3 AS v10 FROM reviews r WHERE
    AI_FILTER(PROMPT('q1: {0}', r.review)) ---
plan
  __quail_answer_1 = ai_filter('q1: {0}', r.review), filter: __quail_answer_1, returns:
      []
asked 6 of 6
  'q1: {0}': r0 TRUE, r1 TRUE, r2 FALSE, r3 TRUE, r4 TRUE, r5 FALSE
rows 4
  r0, TRUE, 'n', 'en-en', 'high', 5.0, FALSE, 2, 5, 1, 2, FALSE
  r1, FALSE, 'n', 'en-en', 'low', 2.0, TRUE, 1, 3, 0, 2, TRUE
  r3, NULL, 'n', 'en-en', NULL, NULL, NULL, NULL, 3, NULL, 2, NULL
  r4, TRUE, 'n', 'en-en', 'low', 3.0, TRUE, 1, 3, 1, 2, TRUE

--- SELECT r.id, LOG10(r.stars * 20) > 1.7 AS v0, SUBSTRING(r.lang, 2, 1) AS v1,
    CONCAT(r.lang, '-', r.lang) AS v2, CASE WHEN r.stars > 3 THEN 'high' WHEN r.stars >
    0 THEN 'low' END AS v3, CAST(r.stars AS DOUBLE) AS v4, r.stars IN (2, 3) AS v5,
    ROUND(r.stars / 2) AS v6, GREATEST(r.stars, 3) AS v7, r.stars % 2 AS v8,
    STRPOS(r.lang, 'n') AS v9, r.stars BETWEEN 2 AND 3 AS v10 FROM reviews r WHERE
    AI_FILTER(PROMPT('q1: {0}', r.review)) ---
plan
  __quail_answer_1 = ai_filter('q1: {0}', r.review), filter: __quail_answer_1, returns:
      []
asked 6 of 6
  'q1: {0}': r0 TRUE, r1 TRUE, r2 FALSE, r3 TRUE, r4 TRUE, r5 FALSE
rows 4
  r0, TRUE, 'n', 'en-en', 'high', 5.0, FALSE, 2, 5, 1, 2, FALSE
  r1, FALSE, 'n', 'en-en', 'low', 2.0, TRUE, 1, 3, 0, 2, TRUE
  r3, NULL, 'n', 'en-en', NULL, NULL, NULL, NULL, 3, NULL, 2, NULL
  r4, TRUE, 'n', 'en-en', 'low', 3.0, TRUE, 1, 3, 1, 2, TRUE
""")


def test_order_by_group_by_on_and_prompt_take_expressions():
    band = "CASE WHEN r.stars > 2 THEN 'hi' ELSE 'lo' END"
    assert_transcript(_transcript(
        (f"SELECT r.id {FIRST} ORDER BY ABS(r.stars - 3), r.id", {"ordered": True}),
        (f"SELECT {band} AS band, COUNT(*) AS n, SUM(r.stars * 2) AS s {FIRST} "
         f"GROUP BY {band} ORDER BY band", {"ordered": True}),
        # the model reads the computed text
        "SELECT r.id FROM reviews r "
        "WHERE AI_FILTER(PROMPT('q1: {0}', UPPER(TRIM(r.review))))",
        "SELECT r.id, l.code FROM reviews r "
        "JOIN langs l ON UPPER(TRIM(r.lang)) = l.code "
        "AND AI_FILTER(PROMPT('x {0} {1}', r.review, l.notes))"), """\
--- SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.review)) ORDER BY
    ABS(r.stars - 3), r.id ---
plan
  __quail_answer_1 = ai_filter('q1: {0}', r.review), filter: __quail_answer_1, returns:
      []
asked 6 of 6
  'q1: {0}': r0 TRUE, r1 TRUE, r2 FALSE, r3 TRUE, r4 TRUE, r5 FALSE
rows 4, in order
  r4
  r1
  r0
  r3

--- SELECT CASE WHEN r.stars > 2 THEN 'hi' ELSE 'lo' END AS band, COUNT(*) AS n,
    SUM(r.stars * 2) AS s FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.review))
    GROUP BY CASE WHEN r.stars > 2 THEN 'hi' ELSE 'lo' END ORDER BY band ---
plan
  __quail_answer_1 = ai_filter('q1: {0}', r.review), filter: __quail_answer_1, returns:
      []
asked 6 of 6
  'q1: {0}': r0 TRUE, r1 TRUE, r2 FALSE, r3 TRUE, r4 TRUE, r5 FALSE
rows 2, in order
  'hi', 2, 16
  'lo', 2, 4

--- SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', UPPER(TRIM(r.review))))
    ---
plan
  __quail_answer_1 = ai_filter('q1: {0}', r.__quail_expr_0), filter: __quail_answer_1,
      returns: []
asked 6 of 6
  'q1: {0}': 'B0' TRUE, 'B1' TRUE, 'B2' FALSE, 'B3' TRUE, 'B4' TRUE, 'B5' FALSE
rows 4
  r0
  r1
  r3
  r4

--- SELECT r.id, l.code FROM reviews r JOIN langs l ON UPPER(TRIM(r.lang)) = l.code AND
    AI_FILTER(PROMPT('x {0} {1}', r.review, l.notes)) ---
plan
  __quail_answer_1 = ai_filter('x {0} {1}', r.review, l.notes), filter:
      __quail_answer_1, returns: []
asked 5 of 5
  'x {0} {1}': EN+r0 TRUE, EN+r1 TRUE, EN+r3 TRUE, EN+r4 TRUE, FR+r2 TRUE
rows 5
  r0, EN
  r1, EN
  r2, FR
  r3, EN
  r4, EN
""")


def test_datafusion_runs_plain_tests_and_join_keys_outside_the_region(session):
    query = session.sql(f"{FILTER_SQL} AND r.stars >= 3 "
                        "AND r.lang IN ('en', 'fr') AND r.stars IS NOT NULL")
    (scan,) = [node for node in query.plan().nodes
               if type(node).__name__ == "Scan"]
    assert scan.num_documents == 3
    assert "stars@2 >= 3" in query.explain()
    # the region's scan holds no test of its own
    for test in ("NOT (LOWER(r.lang) LIKE '%e%' AND r.stars < 3)",
                 "ABS(r.stars - 4) <= 1"):
        (scan,) = operators_of(session.sql(f"{FILTER_SQL} AND {test}")).scans
        assert scan.predicates == (), test
    for test in ("UPPER(r.lang) = r.lang", "r.stars > r.floor",
                 "r.stars - r.floor >= 1 OR r.lang = 'en'"):
        (scan,) = operators_of(session.sql(
            f"SELECT r.id {PAIRS_FIRST} AND ({test})")).scans
        assert scan.predicates == (), test
    # a test of two tables runs on the region's rows; the result returns
    # only the selected columns
    join = "AI_FILTER(PROMPT('x {0} {1}', r.review, p.description))"
    query = session.sql(f"SELECT r.id, p.asin FROM reviews r JOIN products p "
                        f"ON {join} WHERE r.stars > 4 OR p.price > 15")
    assert row_tests(query) == ["stars > 4 OR price > 15"]
    assert _run(query, make_executor(TRUTH, lambda *_: True)).collect(
        ).column_names == ["r.id", "p.asin"]
    query = session.sql(f"SELECT r.id, p.asin FROM reviews r JOIN items p "
                        f"ON {join} WHERE r.stars > p.price")
    assert row_tests(query) == ["stars > price"]
    # an equality of two tables in WHERE is the AI join's key
    query = session.sql(f"SELECT r.id, p.asin FROM reviews r, products p "
                        f"WHERE r.stars = p.stars AND {join}")
    assert [str(test) for test in operators_of(query).joins[0].on] == [
        "r.stars = p.stars"]
    assert any(node.type_name == "quail.hash_join" for node in query.plan().nodes)
    # DataFusion computes an ON key before the region
    query = session.sql("SELECT r.id, l.code FROM reviews r "
                        "JOIN langs l ON UPPER(TRIM(r.lang)) = l.code "
                        "AND AI_FILTER(PROMPT('x {0} {1}', r.review, l.notes))")
    assert "upper(btrim(lang@2)) as __quail_expr_0" in query.explain()


def test_answer_tables_and_survivors_hold_positions_in_the_registered_table(
        session):
    query = session.sql(f"{FILTER_SQL} AND r.stars >= 3 "
                        "AND r.lang IN ('en', 'fr') AND r.stars IS NOT NULL")
    result = _run(query, make_executor(TRUTH))
    stage = next(s for s in result.report["stages"] if s["op"] == "filter")
    assert stage["evaluated"] == 3
    answers = result.answer_tables["filters"][("r", stage["written_position"])]
    assert answers.column("r").to_pylist() == [0, 2, 4]
    assert result.survivor_indices["r"].to_pylist() == [0]


def test_a_bad_expression_is_refused_with_its_reason(session):
    with pytest.raises(CompileError, match="unknown function 'nope'"):
        session.sql(f"SELECT r.id {FIRST} AND NOPE(r.lang) = 'x'")
    with pytest.raises(CompileError, match="a test is true or false, got Utf8"):
        _run(session.sql(f"SELECT r.id {PAIRS_FIRST} AND UPPER(r.lang)"),
             make_executor(TRUTH))
    for dialect in ("bq", "snowflake"):
        with pytest.raises(CompileError, match="write r.stars % 2"):
            session.sql(f"SELECT r.id, MOD(r.stars, 2) AS v {FIRST}",
                        dialect=dialect)


def test_a_condition_keeps_sqls_nulls_checks_its_types_and_reads_back_from_a_plan():
    stars = pa.table({"stars": [5, None, 2]})
    written = condition(
        "NOT (stars > 4 AND (stars < 3 OR NOT (stars IS NULL)))",
        {"stars": pa.int64()}, "r", "reviews")
    # the null row stays null through NOT, AND, and OR, so it is rejected
    assert written.evaluate(stars).to_pylist() == [False, None, True]
    assert selected_rows(stars, (written,)).to_pylist() == [2]
    written = condition("stars > 4", {"stars": pa.int64()}, "r", "reviews")
    assert written.types == ("Int64",)
    with pytest.raises(CompileError, match="reads 'stars' as Int64"):
        written.evaluate(pa.table({"stars": ["5", "1"]}))
    read = decode_plan_value(written.to_json())
    rows = pa.table({"stars": [5, 1, None]})
    assert read.evaluate(rows).to_pylist() == [True, False, None]
