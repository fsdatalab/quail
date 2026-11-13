"""An AI filter stops each key once the operator above has enough rows.

derive gives the AI filter on an IN subquery's side of a semi-join one
row per join key, and so to a correlated EXISTS whose AI filter reads
the subquery's table alone. It gives the AI filter under HAVING
COUNT(*) >= k k rows per group key. Each test's transcript, in
transcripts/, shows the plan, the questions the model answered, and the
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
        f"(SELECT r.product FROM reviews r WHERE r.product > 1 AND {AI_IF})"))


def test_an_in_subquery_of_a_classification_has_no_per_key_fetch():
    # a classification is not an AI.IF: the executor has no stop for it
    assert_transcript(_transcript(
        REVIEWS, (),
        "SELECT p.sku FROM products p WHERE p.number IN (SELECT r.product "
        "FROM reviews r WHERE AI_CLASSIFY(PROMPT('tone: {0}', r.body), "
        "['a', 'b']) = 'a')"))


def test_a_count_threshold_stops_each_key_once_it_has_enough_matches():
    queries = [f"SELECT r.product FROM reviews r WHERE {AI_IF} "
               f"GROUP BY r.product HAVING {having}"
               for having in ("COUNT(*) >= 2", "COUNT(*) > 1", "2 <= COUNT(*)")]
    # a SELECT list that reads the count needs every match
    queries.append(f"SELECT r.product, COUNT(*) AS c FROM reviews r WHERE {AI_IF} "
                   "GROUP BY r.product HAVING COUNT(*) >= 2")
    assert_transcript(_transcript(THREE_EACH, ("r0", "r1", "r3", "r6"), *queries))


def test_a_per_key_stop_asks_each_repeated_text_at_most_once():
    assert_transcript(_transcript(REPEATED, ("r0", "r1", "r3"), IN_SQL, COUNT_SQL))


def test_a_key_stops_only_once_a_document_passes_every_filter():
    # the first filter passes both reviews of a product, the second only the
    # later one, so the first must not stop the product at its first pass
    assert_transcript(_transcript(
        REVIEWS, ("r0", "r1", "r2", "r3"),
        f"SELECT p.sku FROM products p WHERE p.number IN (SELECT r.product "
        f"FROM reviews r WHERE {AI_IF} AND AI_FILTER(PROMPT('tone: {{0}}', r.body)))",
        tone=("r1", "r3")))
