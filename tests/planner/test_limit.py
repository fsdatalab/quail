"""A LIMIT as the fetch of the AI operator whose rows it reads.

derive makes a LIMIT, with its OFFSET, the fetch
of the AiNode below it. Each test runs a query with the rule on and off
and checks that both return rows the query allows: without ORDER BY, a
LIMIT of n returns any n of the qualifying rows.
"""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish
from helpers.sessions import CONFIG, TOKENIZER

import quail
from quail.builtins import LOGICAL_RULES, built_in_registry, logical_rules_without

RULE = "derive.fetch"
REVIEWS = pa.table({
    "id": ["r0", "r1", "r2", "r3", "r4", "r5"],
    "stars": [5, 2, 4, 1, 5, 3],
    "body": ["b0", "b1", "b2", "b3", "b4", "b5"],
})
AI_IF = "AI_FILTER(PROMPT('kind: {0}', r.body))"
# the model passes r1, r2, r4, and r5, and labels even rows 'a'
ORACLE = Oracle(filters=lambda alias, document, position: document in (1, 2, 4, 5),
                labels=lambda spec, ids: "a" if ids[0] % 2 == 0 else "b")
PASSING = [("r1",), ("r2",), ("r4",), ("r5",)]
NOTES = REVIEWS.set_column(0, "id", pa.array([f"n{index}" for index in range(6)]))


def _session(rules) -> quail.Session:
    session = quail.Session(CONFIG, tokenizer=TOKENIZER,
                            registry=built_in_registry(logical_rules=rules))
    session.register("reviews", quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    session.register("notes", quail.DocumentProvider.from_table(NOTES, id_col="id"))
    return session


@pytest.fixture
def sessions():
    on = _session(LOGICAL_RULES)
    off = _session(logical_rules_without(RULE))
    yield on, off
    on.close()
    off.close()


def _rows(session, sql) -> list:
    result = finish(session.sql(sql), ORACLE).collect()
    return [tuple(row.values()) for row in result.to_pylist()]


def _rows_on_and_off(sessions, sql) -> list:
    return [_rows(session, sql) for session in sessions]


def _assert_some_of(rows, allowed, count):
    assert len(rows) == count and len(set(rows)) == count, rows
    assert set(rows) <= set(allowed), rows


def test_a_limit_over_an_ai_filter_returns_that_many_passing_rows(sessions):
    sql = f"SELECT r.id FROM reviews r WHERE {AI_IF} LIMIT 2"
    assert "fetch: 2" in sessions[0].sql(sql).logical
    assert "fetch: 2" not in sessions[1].sql(sql).logical
    for rows in _rows_on_and_off(sessions, sql):
        _assert_some_of(rows, PASSING, 2)


def test_a_fetch_keeps_a_later_test_of_the_answer_above_the_ai_node(sessions):
    # testing the label before the fetch would return three reviews
    # labeled 'a'; the test runs on the three reviews the fetch keeps
    sql = ("SELECT d.id FROM (SELECT r.id, AI_CLASSIFY(PROMPT('tone: {0}', r.body), "
           "['a', 'b']) AS t FROM reviews r LIMIT 3) d WHERE d.t = 'a'")
    logical = sessions[0].sql(sql).logical
    assert "fetch: 3" in logical and "Filter" not in logical, logical
    for rows in _rows_on_and_off(sessions, sql):
        assert len(rows) <= 3 and set(rows) <= {("r0",), ("r2",), ("r4",)}, rows


def test_a_fetch_keeps_a_later_filter_of_the_input_above_the_ai_node(sessions):
    sql = (f"SELECT d.id FROM (SELECT r.id, r.stars FROM reviews r WHERE {AI_IF} "
           "LIMIT 3) d WHERE d.stars > 3")
    logical = sessions[0].sql(sql).logical
    assert "fetch: 3" in logical and "Filter" not in logical, logical
    for rows in _rows_on_and_off(sessions, sql):
        # three of the four passing reviews hold at least one of r2 and r4
        assert 1 <= len(rows) <= 2 and set(rows) <= {("r2",), ("r4",)}, rows


def test_a_limit_with_an_offset_skips_rows_after_the_ai_filter(sessions):
    sql = f"SELECT r.id FROM reviews r WHERE {AI_IF} LIMIT 2 OFFSET 1"
    assert "fetch: 3" in sessions[0].sql(sql).logical
    for rows in _rows_on_and_off(sessions, sql):
        _assert_some_of(rows, PASSING, 2)


def test_fewer_passing_rows_than_the_limit_returns_every_passing_row(sessions):
    sql = f"SELECT r.id FROM reviews r WHERE {AI_IF} LIMIT 10"
    for rows in _rows_on_and_off(sessions, sql):
        assert sorted(rows) == PASSING


def test_a_limit_over_a_projection_of_an_answer_becomes_the_fetch(sessions):
    # Limit_k(π(AI(R))) = π(Limit_k(AI(R))): the projection keeps one row per row
    sql = ("SELECT r.id, CONCAT(AI_CLASSIFY(PROMPT('tone: {0}', r.body), "
           "['a', 'b']), '!') AS t FROM reviews r LIMIT 2")
    assert "fetch: 2" in sessions[0].sql(sql).logical
    assert "fetch: 2" not in sessions[1].sql(sql).logical
    labeled = [(f"r{index}", "a!" if index % 2 == 0 else "b!") for index in range(6)]
    for rows in _rows_on_and_off(sessions, sql):
        _assert_some_of(rows, labeled, 2)


def test_a_limit_over_a_union_becomes_the_fetch_of_each_branch(sessions):
    # Limit_k(R ⊎ S) = Limit_k(Limit_k(R) ⊎ Limit_k(S))
    sql = (f"SELECT r.id FROM reviews r WHERE {AI_IF} UNION ALL "
           "SELECT n.id FROM notes n WHERE AI_FILTER(PROMPT('kind: {0}', n.body)) "
           "LIMIT 3")
    for session, fetched in zip(sessions, (True, False), strict=True):
        regions = session.sql(sql).regions()
        assert len(regions) == 2
        for region in regions:
            assert ("fetch: 3" in region.logical) == fetched, region.logical
    passing = PASSING + [(f"n{row[0][1:]}",) for row in PASSING]
    for rows in _rows_on_and_off(sessions, sql):
        _assert_some_of(rows, passing, 3)
