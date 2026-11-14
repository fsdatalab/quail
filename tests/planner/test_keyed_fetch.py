"""An AI filter stops each key once the operator above has enough rows.

derive gives the AI filter on an IN subquery's side of a semi-join one
row per join key, and so to a correlated EXISTS whose AI filter reads
the subquery's table alone. It gives the AI filter under HAVING
COUNT(*) >= k k rows per group key. Each test's transcript shows the
plan, the questions the model answered, and the
rows (helpers/transcript.py).
"""

import pyarrow as pa
from helpers.transcript import Transcripts, assert_transcript

# two reviews of each product
REVIEWS = pa.table({
    "id": [f"r{index}" for index in range(6)],
    "body": [f"b{index}" for index in range(6)],
    "product": [1, 1, 2, 2, 3, 3],
})
# three reviews of each product
THREE_EACH = pa.table({
    "id": [f"r{index}" for index in range(9)],
    "body": [f"b{index}" for index in range(9)],
    "product": [1, 1, 1, 2, 2, 2, 3, 3, 3],
})
# texts repeat across products, and r0 and r1 share a text within product 1
REPEATED = pa.table({
    "id": [f"r{index}" for index in range(6)],
    "body": ["b0", "b0", "b1", "b0", "b2", "b2"],
    "product": [1, 1, 1, 2, 2, 2],
})
PRODUCTS = pa.table({"sku": ["p1", "p2", "p3", "p4"], "number": [1, 2, 3, 4]})
AI_IF = "AI_FILTER(PROMPT('kind: {0}', r.body))"
IN_SQL = (f"SELECT p.sku FROM products p WHERE p.number IN "
          f"(SELECT r.product FROM reviews r WHERE {AI_IF})")
COUNT_SQL = (f"SELECT r.product FROM reviews r WHERE {AI_IF} "
             "GROUP BY r.product HAVING COUNT(*) >= 2")


def _transcript(reviews, passing, *queries, tone=()) -> str:
    """Transcripts of queries whose model passes the reviews in ``passing``."""
    def answer(kind, prompt, rows):
        if prompt.startswith("tone:"):
            return rows[0] in tone if kind == "filter" else "a"
        return any(row in passing for row in rows[0].split("/"))

    tables = {"reviews": reviews, "products": (PRODUCTS, "sku")}
    with Transcripts(tables, answer, chunk_tokens=256) as runs:
        return "\n".join(runs.query(sql) for sql in queries)


def test_an_in_subquery_or_exists_of_an_ai_filter_stops_each_key_at_its_first_match():
    assert_transcript(_transcript(
        REVIEWS, ("r0", "r1", "r2", "r3"),
        IN_SQL,
        f"SELECT p.sku FROM products p WHERE EXISTS (SELECT 1 FROM "
        f"reviews r WHERE r.product = p.number AND {AI_IF})",
        # the subquery's other filters still run before the model
        f"SELECT p.sku FROM products p WHERE p.number IN "
        f"(SELECT r.product FROM reviews r WHERE r.product > 1 AND {AI_IF})"), """\
--- SELECT p.sku FROM products p WHERE p.number IN (SELECT r.product FROM reviews r
    WHERE AI_FILTER(PROMPT('kind: {0}', r.body))) ---
plan
  __quail_answer_1 = ai_filter('kind: {0}', r.body), filter: __quail_answer_1, fetch: 1
      per [product], returns: []
asked 4 of 6
  'kind: {0}': r0 TRUE, r2 TRUE, r4 FALSE, r5 FALSE
rows 2
  p1
  p2

--- SELECT p.sku FROM products p WHERE EXISTS (SELECT 1 FROM reviews r WHERE r.product =
    p.number AND AI_FILTER(PROMPT('kind: {0}', r.body))) ---
plan
  __quail_answer_1 = ai_filter('kind: {0}', r.body), filter: __quail_answer_1, fetch: 1
      per [product], returns: []
asked 4 of 6
  'kind: {0}': r0 TRUE, r2 TRUE, r4 FALSE, r5 FALSE
rows 2
  p1
  p2

--- SELECT p.sku FROM products p WHERE p.number IN (SELECT r.product FROM reviews r
    WHERE r.product > 1 AND AI_FILTER(PROMPT('kind: {0}', r.body))) ---
plan
  __quail_answer_1 = ai_filter('kind: {0}', r.body), filter: __quail_answer_1, fetch: 1
      per [product], returns: []
asked 3 of 4
  'kind: {0}': r2 TRUE, r4 FALSE, r5 FALSE
rows 1
  p2
""")


def test_an_in_subquery_of_a_classification_has_no_per_key_fetch():
    # a classification is not an AI.IF: the executor has no stop for it
    assert_transcript(_transcript(
        REVIEWS, (),
        "SELECT p.sku FROM products p WHERE p.number IN (SELECT r.product "
        "FROM reviews r WHERE AI_CLASSIFY(PROMPT('tone: {0}', r.body), "
        "['a', 'b']) = 'a')"), """\
--- SELECT p.sku FROM products p WHERE p.number IN (SELECT r.product FROM reviews r
    WHERE AI_CLASSIFY(PROMPT('tone: {0}', r.body), ['a', 'b']) = 'a') ---
plan
  __quail_column_1 = ai_classify('tone: {0}', r.body, make_array('a', 'b')), filter:
      __quail_column_1 = 'a'
asked 6 of 6
  'tone: {0}': r0 'a', r1 'a', r2 'a', r3 'a', r4 'a', r5 'a'
rows 3
  p1
  p2
  p3
""")


def test_a_count_threshold_stops_each_key_once_it_has_enough_matches():
    queries = [f"SELECT r.product FROM reviews r WHERE {AI_IF} "
               f"GROUP BY r.product HAVING {having}"
               for having in ("COUNT(*) >= 2", "COUNT(*) > 1", "2 <= COUNT(*)")]
    # a SELECT list that reads the count needs every match
    queries.append(f"SELECT r.product, COUNT(*) AS c FROM reviews r WHERE {AI_IF} "
                   "GROUP BY r.product HAVING COUNT(*) >= 2")
    assert_transcript(_transcript(THREE_EACH, ("r0", "r1", "r3", "r6"), *queries), """\
--- SELECT r.product FROM reviews r WHERE AI_FILTER(PROMPT('kind: {0}', r.body)) GROUP
    BY r.product HAVING COUNT(*) >= 2 ---
plan
  __quail_answer_1 = ai_filter('kind: {0}', r.body), filter: __quail_answer_1, fetch: 2
      per [product], returns: []
asked 9 of 9
  'kind: {0}': r0 TRUE, r1 TRUE, r2 FALSE, r3 TRUE, r4 FALSE, r5 FALSE, r6 TRUE, r7
      FALSE, r8 FALSE
rows 1
  1

--- SELECT r.product FROM reviews r WHERE AI_FILTER(PROMPT('kind: {0}', r.body)) GROUP
    BY r.product HAVING COUNT(*) > 1 ---
plan
  __quail_answer_1 = ai_filter('kind: {0}', r.body), filter: __quail_answer_1, fetch: 2
      per [product], returns: []
asked 9 of 9
  'kind: {0}': r0 TRUE, r1 TRUE, r2 FALSE, r3 TRUE, r4 FALSE, r5 FALSE, r6 TRUE, r7
      FALSE, r8 FALSE
rows 1
  1

--- SELECT r.product FROM reviews r WHERE AI_FILTER(PROMPT('kind: {0}', r.body)) GROUP
    BY r.product HAVING 2 <= COUNT(*) ---
plan
  __quail_answer_1 = ai_filter('kind: {0}', r.body), filter: __quail_answer_1, fetch: 2
      per [product], returns: []
asked 9 of 9
  'kind: {0}': r0 TRUE, r1 TRUE, r2 FALSE, r3 TRUE, r4 FALSE, r5 FALSE, r6 TRUE, r7
      FALSE, r8 FALSE
rows 1
  1

--- SELECT r.product, COUNT(*) AS c FROM reviews r WHERE AI_FILTER(PROMPT('kind: {0}',
    r.body)) GROUP BY r.product HAVING COUNT(*) >= 2 ---
plan
  __quail_answer_1 = ai_filter('kind: {0}', r.body), filter: __quail_answer_1, returns:
      []
asked 9 of 9
  'kind: {0}': r0 TRUE, r1 TRUE, r2 FALSE, r3 TRUE, r4 FALSE, r5 FALSE, r6 TRUE, r7
      FALSE, r8 FALSE
rows 1
  1, 2
""")


def test_a_per_key_stop_asks_each_repeated_text_at_most_once():
    assert_transcript(_transcript(REPEATED, ("r0", "r1", "r3"), IN_SQL, COUNT_SQL), """\
--- SELECT p.sku FROM products p WHERE p.number IN (SELECT r.product FROM reviews r
    WHERE AI_FILTER(PROMPT('kind: {0}', r.body))) ---
plan
  __quail_answer_1 = ai_filter('kind: {0}', r.body), filter: __quail_answer_1, fetch: 1
      per [product], returns: []
asked 1 of 6
  'kind: {0}': r0/r1/r3 TRUE
rows 2
  p1
  p2

--- SELECT r.product FROM reviews r WHERE AI_FILTER(PROMPT('kind: {0}', r.body)) GROUP
    BY r.product HAVING COUNT(*) >= 2 ---
plan
  __quail_answer_1 = ai_filter('kind: {0}', r.body), filter: __quail_answer_1, fetch: 2
      per [product], returns: []
asked 3 of 6
  'kind: {0}': r0/r1/r3 TRUE, r2 FALSE, r4/r5 FALSE
rows 1
  1
""")


def test_a_key_stops_only_once_a_document_passes_every_filter():
    # the first filter passes both reviews of a product, the second only the
    # later one, so the first must not stop the product at its first pass
    assert_transcript(_transcript(
        REVIEWS, ("r0", "r1", "r2", "r3"),
        f"SELECT p.sku FROM products p WHERE p.number IN (SELECT r.product "
        f"FROM reviews r WHERE {AI_IF} AND AI_FILTER(PROMPT('tone: {{0}}', r.body)))",
        tone=("r1", "r3")), """\
--- SELECT p.sku FROM products p WHERE p.number IN (SELECT r.product FROM reviews r
    WHERE AI_FILTER(PROMPT('kind: {0}', r.body)) AND AI_FILTER(PROMPT('tone: {0}',
    r.body))) ---
plan
  __quail_answer_2 = ai_filter('tone: {0}', r.body), filter: __quail_answer_2, fetch: 1
      per [product], returns: []
  __quail_answer_1 = ai_filter('kind: {0}', r.body), filter: __quail_answer_1, returns:
      []
asked 10 of 10
  'kind: {0}': r0 TRUE, r1 TRUE, r2 TRUE, r3 TRUE, r4 FALSE, r5 FALSE
  'tone: {0}': r0 FALSE, r1 TRUE, r2 FALSE, r3 TRUE
rows 2
  p1
  p2
""")
