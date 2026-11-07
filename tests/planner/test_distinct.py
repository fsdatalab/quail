"""DataFusion's DISTINCT rules in a model region.

EliminateDistinct drops a DISTINCT that removes no row; PushDownDistinct
gives the AI operator under a DISTINCT a fetch of one row per value of
the selected columns. The tests of rows run each query with
push_down_distinct on and off and check that both return the same rows.
"""


import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish
from helpers.sessions import CONFIG, TOKENIZER

import quail
from quail.builtins import LOGICAL_RULES, built_in_registry

REVIEWS = pa.table({"id": ["r0", "r1"], "stars": [5, 2], "body": ["good", "slow"]})


@pytest.fixture()
def session():
    value = quail.Session(CONFIG, tokenizer=TOKENIZER)
    value.register("reviews", quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    yield value
    value.close()


@pytest.mark.parametrize("text", [
    "SELECT DISTINCT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body))",
    "SELECT DISTINCT r.stars AS k, COUNT(*) AS n FROM reviews r "
    "WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) GROUP BY r.stars",
])
def test_a_distinct_that_removes_no_row_is_eliminated(session, text):
    # DataFusion's plan of the region, as explain() shows it
    assert "Distinct" not in session.sql(text).logical


def test_a_distinct_on_one_tables_columns_stops_its_filter_at_the_first_match(session):
    query = session.sql("SELECT DISTINCT r.stars FROM reviews r "
                        "WHERE AI_FILTER(PROMPT('q1: {0}', r.body))")
    assert "Distinct" in query.logical
    assert "fetch: 1 per [stars]" in query.logical
    assert query.facts()["stop_keys"] == {"r": ["stars"]}


KEYED = pa.table({
    "id": ["k0", "k1", "k2", "k3", "k4", "k5"],
    "stars": [5, 5, 4, 4, 3, 3],
    "body": ["b0", "b1", "b2", "b3", "b4", "b5"],
})
AI_IF = "AI_FILTER(PROMPT('kind: {0}', r.body))"
# the model passes k1, k2, k3, and k4, and labels the first rows 'a', 'b',
# then 'c' for the rest
ORACLE = Oracle(filters=lambda alias, document, position: document in (1, 2, 3, 4),
                labels=lambda spec, ids: "abcccc"[ids[0]])


def _keyed_session(rules) -> quail.Session:
    session = quail.Session(CONFIG, tokenizer=TOKENIZER,
                            registry=built_in_registry(logical_rules=rules))
    session.register("reviews", quail.DocumentProvider.from_table(KEYED, id_col="id"))
    return session


@pytest.fixture()
def sessions():
    on = _keyed_session(LOGICAL_RULES)
    off = _keyed_session(
        [name for name in LOGICAL_RULES if name != "push_down_distinct"])
    yield on, off
    on.close()
    off.close()


def _rows_on_and_off(sessions, sql) -> list:
    found = []
    for session in sessions:
        result = finish(session.sql(sql), ORACLE).collect()
        found.append(sorted(tuple(row.values()) for row in result.to_pylist()))
    assert found[0] == found[1]
    return found[0]


def test_a_distinct_over_an_ai_filters_input_columns_returns_each_passing_key_once(
        sessions):
    sql = f"SELECT DISTINCT r.stars FROM reviews r WHERE {AI_IF}"
    assert "fetch: 1 per [stars]" in sessions[0].sql(sql).logical
    assert "fetch: 1 per [stars]" not in sessions[1].sql(sql).logical
    assert _rows_on_and_off(sessions, sql) == [(3,), (4,), (5,)]


def test_a_filter_on_a_distincts_columns_keeps_the_per_key_fetch(sessions):
    sql = (f"SELECT * FROM (SELECT DISTINCT r.stars FROM reviews r WHERE {AI_IF}) d "
           "WHERE d.stars > 3")
    query = sessions[0].sql(sql)
    assert "fetch: 1 per [stars]" in query.logical
    assert query.facts()["stop_keys"] == {"r": ["stars"]}
    assert _rows_on_and_off(sessions, sql) == [(4,), (5,)]


def test_a_distinct_over_a_classifys_label_fetches_one_row_per_label(sessions):
    sql = ("SELECT DISTINCT AI_CLASSIFY(PROMPT('tone: {0}', r.body), ['a', 'b', 'c']) "
           "AS t FROM reviews r")
    assert "fetch: 1 per [t]" in sessions[0].sql(sql).logical
    assert _rows_on_and_off(sessions, sql) == [("a",), ("b",), ("c",)]
