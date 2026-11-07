"""A LIMIT as the fetch of the AI operator whose rows it reads.

push_down_limit_into_ai_nodes makes a LIMIT, with its OFFSET, the fetch
of the AiNode below it. Each test runs a query with the rule on and off
and checks that both return the same rows.
"""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish
from helpers.sessions import CONFIG, TOKENIZER

import quail
from quail.builtins import LOGICAL_RULES, built_in_registry

RULE = "push_down_limit_into_ai_nodes"
REVIEWS = pa.table({
    "id": ["r0", "r1", "r2", "r3", "r4", "r5"],
    "stars": [5, 2, 4, 1, 5, 3],
    "body": ["b0", "b1", "b2", "b3", "b4", "b5"],
})
AI_IF = "AI_FILTER(PROMPT('kind: {0}', r.body))"
# the model passes r1, r2, r4, and r5, and labels even rows 'a'
ORACLE = Oracle(filters=lambda alias, document, position: document in (1, 2, 4, 5),
                labels=lambda spec, ids: "a" if ids[0] % 2 == 0 else "b")


def _session(rules) -> quail.Session:
    session = quail.Session(CONFIG, tokenizer=TOKENIZER,
                            registry=built_in_registry(logical_rules=rules))
    session.register("reviews", quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    return session


@pytest.fixture
def sessions():
    on = _session(LOGICAL_RULES)
    off = _session([name for name in LOGICAL_RULES if name != RULE])
    yield on, off
    on.close()
    off.close()


def _rows(session, sql) -> list:
    result = finish(session.sql(sql), ORACLE).collect()
    return [tuple(row.values()) for row in result.to_pylist()]


def _rows_on_and_off(sessions, sql) -> list:
    on, off = sessions
    rows = _rows(on, sql)
    assert rows == _rows(off, sql)
    return rows


def test_a_limit_over_an_ai_filter_returns_the_first_passing_rows_in_input_order(
        sessions):
    sql = f"SELECT r.id FROM reviews r WHERE {AI_IF} LIMIT 2"
    assert "fetch: 2" in sessions[0].sql(sql).logical
    assert "fetch: 2" not in sessions[1].sql(sql).logical
    assert _rows_on_and_off(sessions, sql) == [("r1",), ("r2",)]


def test_a_fetch_keeps_a_later_test_of_the_answer_above_the_ai_node(sessions):
    # the first three reviews, then those labeled 'a': testing the label
    # before the fetch would keep the first three labeled 'a'
    sql = ("SELECT d.id FROM (SELECT r.id, AI_CLASSIFY(PROMPT('tone: {0}', r.body), "
           "['a', 'b']) AS t FROM reviews r LIMIT 3) d WHERE d.t = 'a'")
    assert _rows_on_and_off(sessions, sql) == [("r0",), ("r2",)]


def test_a_fetch_keeps_a_later_filter_of_the_input_above_the_ai_node(sessions):
    sql = (f"SELECT d.id FROM (SELECT r.id, r.stars FROM reviews r WHERE {AI_IF} "
           "LIMIT 3) d WHERE d.stars > 3")
    assert _rows_on_and_off(sessions, sql) == [("r2",), ("r4",)]


def test_a_limit_with_an_offset_skips_rows_after_the_ai_filter(sessions):
    sql = f"SELECT r.id FROM reviews r WHERE {AI_IF} LIMIT 2 OFFSET 1"
    assert "fetch: 3" in sessions[0].sql(sql).logical
    assert _rows_on_and_off(sessions, sql) == [("r2",), ("r4",)]


def test_fewer_passing_rows_than_the_limit_returns_every_passing_row(sessions):
    sql = f"SELECT r.id FROM reviews r WHERE {AI_IF} LIMIT 10"
    assert _rows_on_and_off(sessions, sql) == [("r1",), ("r2",), ("r4",), ("r5",)]
