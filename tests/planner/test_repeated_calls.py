"""A repeated AI call asks the model once per row.

AI_{f→b}(AI_{f→a}(R)) = π_{*, a→b}(AI_{f→a}(R)). The lowering gives
calls written the same way in one SELECT list and WHERE one AiNode;
merge_repeated_ai_calls merges an AiNode into the same call below it,
such as an AI.IF written twice. Each test runs a query with the rule on
and off and checks that both return the same rows.
"""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish
from helpers.sessions import CONFIG, TOKENIZER

import quail
from quail.builtins import LOGICAL_RULES, built_in_registry

RULE = "merge_repeated_ai_calls"
REVIEWS = pa.table({
    "id": ["r0", "r1", "r2", "r3", "r4", "r5"],
    "body": ["b0", "b1", "b2", "b3", "b4", "b5"],
})
AI_IF = "AI_FILTER(PROMPT('kind: {0}', r.body))"
TONE = "AI_CLASSIFY(PROMPT('tone: {0}', r.body), ['a', 'b'])"


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


def _rows_and_questions(session, sql) -> tuple[list, int]:
    """Return the query's rows, sorted, and how many questions the model answered."""
    asked = []

    def passes(alias, document, position):
        asked.append(document)
        return document in (1, 2, 4, 5)

    oracle = Oracle(filters=passes,
                    labels=lambda spec, ids: "a" if ids[0] % 2 == 0 else "b")
    result = finish(session.sql(sql), oracle).collect()
    labeled = sum(len(rows) for _, rows in oracle.calls)
    rows = sorted(tuple(row.values()) for row in result.to_pylist())
    return rows, len(asked) + labeled


def test_an_ai_filter_written_twice_asks_the_model_once_per_row(sessions):
    sql = f"SELECT r.id FROM reviews r WHERE {AI_IF} AND {AI_IF}"
    assert "predicate 2" not in sessions[0].sql(sql).explain()
    assert "predicate 2" in sessions[1].sql(sql).explain()
    on, off = (_rows_and_questions(session, sql) for session in sessions)
    assert on[0] == off[0] == [("r1",), ("r2",), ("r4",), ("r5",)]
    # without the rule, the second filter asks again about the four that pass
    assert (on[1], off[1]) == (6, 10)


def test_a_classification_in_select_and_where_labels_each_row_once(sessions):
    sql = f"SELECT r.id, {TONE} AS t FROM reviews r WHERE {TONE} = 'a'"
    for session in sessions:
        assert session.sql(sql).logical.count("AiNode") == 1
        rows, num_questions = _rows_and_questions(session, sql)
        assert rows == [("r0", "a"), ("r2", "a"), ("r4", "a")]
        assert num_questions == 6
