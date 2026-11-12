"""DataFusion's DISTINCT rules in a model region.

derive drops a DISTINCT that removes no row, and gives the AI operator
under any other DISTINCT a fetch of one row per value of the selected
columns. The tests of rows run each query with
derive's fetch on and off and check that both return the same rows.
"""


import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish
from helpers.sessions import CONFIG, TOKENIZER

import quail
from quail.builtins import LOGICAL_RULES, built_in_registry, logical_rules_without

REVIEWS = pa.table({"id": ["r0", "r1"], "stars": [5, 2], "body": ["good", "slow"]})


@pytest.fixture()
def session():
    value = quail.Session(CONFIG, tokenizer=TOKENIZER)
    value.register("reviews", quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    yield value
    value.close()


def test_a_distinct_that_removes_no_row_is_eliminated(session):
    for text in (
            "SELECT DISTINCT r.id FROM reviews r "
            "WHERE AI_FILTER(PROMPT('q1: {0}', r.body))",
            "SELECT DISTINCT r.stars AS k, COUNT(*) AS n FROM reviews r "
            "WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) GROUP BY r.stars"):
        # DataFusion's plan of the region, as explain() shows it
        assert "Distinct" not in session.sql(text).logical, text


def test_a_limit_above_a_distinct_of_unique_keys_stops_the_model(session):
    session.register("aspects", quail.DocumentProvider.from_table(
        pa.table({"id": ["a0", "a1"], "aspect": ["price", "speed"]}), id_col="id"))
    for text in (
            "SELECT DISTINCT r.id, r.body FROM reviews r "
            "WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) LIMIT 1",
            "SELECT DISTINCT r.id, a.id FROM reviews r JOIN aspects a "
            "ON AI_FILTER(PROMPT('q2: {0} {1}', r.body, a.aspect)) LIMIT 1"):
        (region,) = quail.SessionContext(session)._native.explain_regions(
            text, None, False)
        assert "Limit: 1" in region and "stop per key" not in region, text
    # rows of one table read twice repeat, so that DISTINCT stays
    text = ("SELECT DISTINCT u.id FROM (SELECT r.id FROM reviews r "
            "WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) UNION ALL "
            "SELECT r.id FROM reviews r) u")
    passing = Oracle(filters=lambda alias, document, position: True)
    found = finish(session.sql(text), passing).collect()
    assert sorted(found.column(0).to_pylist()) == ["r0", "r1"]


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
        logical_rules_without("derive.fetch"))
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
    query = sessions[0].sql(sql)
    # DataFusion runs the DISTINCT above the region
    assert "AggregateExec" in query.explain()
    assert "fetch: 1 per [stars]" in query.logical
    assert query.facts()["stop_keys"] == {"r": ["stars"]}
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
