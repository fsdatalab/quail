"""An AI filter stops each key once the operator above has enough rows.

push_down_semi_join_fetch gives the AI filter on an IN subquery's side
of a semi-join one row per join key (R13), and so to a correlated EXISTS
whose AI filter reads the subquery's table alone; push_down_count_threshold
gives the AI filter under HAVING COUNT(*) >= k k rows per group key
(R14). Each test runs a query with the rule on and off, checks that both
return the same rows, and counts the documents the model read.
"""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish
from helpers.sessions import CONFIG, TOKENIZER

import quail
from quail.builtins import LOGICAL_RULES, built_in_registry

SEMI_JOIN_RULE = "push_down_semi_join_fetch"
COUNT_RULE = "push_down_count_threshold"
# two reviews of each product; the model passes r0 to r3
REVIEWS = pa.table({
    "id": [f"r{index}" for index in range(6)],
    "body": [f"b{index}" for index in range(6)],
    "product": [1, 1, 2, 2, 3, 3],
})
PRODUCTS = pa.table({"sku": ["p1", "p2", "p3", "p4"], "number": [1, 2, 3, 4]})
AI_IF = "AI_FILTER(PROMPT('kind: {0}', r.body))"


def _session(rules, reviews=REVIEWS) -> quail.Session:
    session = quail.Session(CONFIG, tokenizer=TOKENIZER,
                            registry=built_in_registry(logical_rules=rules))
    session.register("reviews",
                     quail.DocumentProvider.from_table(reviews, id_col="id"))
    session.register("products",
                     quail.DocumentProvider.from_table(PRODUCTS, id_col="sku"))
    return session


def _rows_and_read(rule, sql, reviews=REVIEWS, passing=(0, 1, 2, 3)) -> list:
    """Return, with the rule on and off, the rows, sorted, and the documents read."""
    found = []
    for rules in (LOGICAL_RULES, [name for name in LOGICAL_RULES if name != rule]):
        session = _session(rules, reviews)
        read = []

        def passes(alias, document, position, read=read):
            read.append(document)
            return document in passing

        result = finish(session.sql(sql), Oracle(filters=passes)).collect()
        found.append((sorted(tuple(row.values()) for row in result.to_pylist()),
                      sorted(read)))
        session.close()
    return found


def test_an_in_subquery_of_an_ai_filter_stops_each_key_at_its_first_match():
    sql = (f"SELECT p.sku FROM products p WHERE p.number IN "
           f"(SELECT r.product FROM reviews r WHERE {AI_IF})")
    on, off = _rows_and_read(SEMI_JOIN_RULE, sql)
    assert on[0] == off[0] == [("p1",), ("p2",)]
    # r1 and r3 share a product with a review that already passed
    assert (on[1], off[1]) == ([0, 2, 4, 5], [0, 1, 2, 3, 4, 5])


def test_an_in_subquery_keeps_its_other_filters_before_the_model():
    sql = (f"SELECT p.sku FROM products p WHERE p.number IN "
           f"(SELECT r.product FROM reviews r WHERE r.product > 1 AND {AI_IF})")
    on, off = _rows_and_read(SEMI_JOIN_RULE, sql)
    assert on[0] == off[0] == [("p2",)]
    assert (len(on[1]), len(off[1])) == (3, 4)


def test_a_correlated_exists_of_an_ai_filter_stops_each_key_at_its_first_match():
    sql = (f"SELECT p.sku FROM products p WHERE EXISTS (SELECT 1 FROM reviews r "
           f"WHERE r.product = p.number AND {AI_IF})")
    on, off = _rows_and_read(SEMI_JOIN_RULE, sql)
    # the same rows as asking every review, as the IN subquery returns
    assert on[0] == off[0] == [("p1",), ("p2",)]
    # r1 and r3 share a product with a review that already passed
    assert (on[1], off[1]) == ([0, 2, 4, 5], [0, 1, 2, 3, 4, 5])


@pytest.mark.parametrize("subquery", [
    # a classification is not an AI.IF: the executor has no stop for it
    "SELECT r.product FROM reviews r WHERE "
    "AI_CLASSIFY(PROMPT('tone: {0}', r.body), ['a', 'b']) = 'a'",
])
def test_an_in_subquery_of_another_ai_operator_has_no_per_key_fetch(subquery):
    session = _session(LOGICAL_RULES)
    query = session.sql(f"SELECT p.sku FROM products p WHERE p.number IN ({subquery})")
    assert "per [" not in query.explain()
    session.close()


# three reviews of each product; the model passes r0, r1, r3, and r6
THREE_EACH = pa.table({
    "id": [f"r{index}" for index in range(9)],
    "body": [f"b{index}" for index in range(9)],
    "product": [1, 1, 1, 2, 2, 2, 3, 3, 3],
})


@pytest.mark.parametrize("having", ["COUNT(*) >= 2", "COUNT(*) > 1", "2 <= COUNT(*)"])
def test_a_count_threshold_stops_each_key_once_it_has_enough_matches(having):
    sql = (f"SELECT r.product FROM reviews r WHERE {AI_IF} "
           f"GROUP BY r.product HAVING {having}")
    on, off = _rows_and_read(COUNT_RULE, sql, THREE_EACH, passing=(0, 1, 3, 6))
    assert on[0] == off[0] == [(1,)]
    # product 1 has two matches after r0 and r1, so r2 is not read
    assert (on[1], off[1]) == ([0, 1, 3, 4, 5, 6, 7, 8], list(range(9)))


def test_a_count_threshold_the_select_list_reads_reads_every_document():
    sql = (f"SELECT r.product, COUNT(*) AS c FROM reviews r WHERE {AI_IF} "
           "GROUP BY r.product HAVING COUNT(*) >= 2")
    on, off = _rows_and_read(COUNT_RULE, sql, THREE_EACH, passing=(0, 1, 3, 6))
    assert on == off
    assert on[0] == [(1, 2)]


# texts repeat across products, and r0 and r1 share a text within product 1
REPEATED = pa.table({
    "id": [f"r{index}" for index in range(6)],
    "body": ["b0", "b0", "b1", "b0", "b2", "b2"],
    "product": [1, 1, 1, 2, 2, 2],
})


def test_a_per_key_stop_asks_each_repeated_text_at_most_once():
    in_sql = (f"SELECT p.sku FROM products p WHERE p.number IN "
              f"(SELECT r.product FROM reviews r WHERE {AI_IF})")
    count_sql = (f"SELECT r.product FROM reviews r WHERE {AI_IF} "
                 "GROUP BY r.product HAVING COUNT(*) >= 2")
    # the model passes every copy of b0
    on, off = _rows_and_read(SEMI_JOIN_RULE, in_sql, REPEATED, passing=(0, 1, 3))
    assert on[0] == off[0] == [("p1",), ("p2",)]
    # r0 passes for products 1 and 2, so no other text is asked
    assert (on[1], off[1]) == ([0], [0, 2, 4])
    on, off = _rows_and_read(COUNT_RULE, count_sql, REPEATED, passing=(0, 1, 3))
    assert on[0] == off[0] == [(1,)]
    # r0 counts twice for product 1, so b1 is not asked; product 2 has one
    # match and asks b2
    assert (on[1], off[1]) == ([0, 4], [0, 2, 4])
