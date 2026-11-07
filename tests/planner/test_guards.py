"""A guarded AI call asks the model only where its guard is TRUE.

An AI operator's guard g says which rows the model is asked about; every
other row's answer is NULL. push_down_case_guards makes the test of a
CASE that reads the answer the guard (R6). Each test runs a query with
the rule on and off, checks that both return the same rows, and counts
the questions the model answered.
"""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish
from helpers.sessions import CONFIG, TOKENIZER

import quail
from quail.api.config import EngineConfig
from quail.builtins import LOGICAL_RULES, built_in_registry
from quail.physical.plan import Refusal

CASE_RULE = "push_down_case_guards"
REVIEWS = pa.table({
    "id": ["r0", "r1", "r2", "r3", "r4", "r5"],
    "body": ["b0", "b1", "b2", "b3", "b4", "b5"],
    "long": [True, False, None, True, False, True],
})
TONE = "AI_CLASSIFY(PROMPT('tone: {0}', r.body), ['a', 'b'])"


def _session(rules) -> quail.Session:
    session = quail.Session(CONFIG, tokenizer=TOKENIZER,
                            registry=built_in_registry(logical_rules=rules))
    session.register("reviews",
                     quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    return session


@pytest.fixture
def sessions():
    on = _session(LOGICAL_RULES)
    off = _session([name for name in LOGICAL_RULES if name != CASE_RULE])
    yield on, off
    on.close()
    off.close()


def _rows_and_questions(session, sql) -> tuple[list, int]:
    """Return the query's rows, sorted, and how many questions the model answered."""
    written = []

    def summary(spec, document):
        written.append(document)
        return f"summary {document}"

    oracle = Oracle(labels=lambda spec, ids: "a" if ids[0] % 2 == 0 else "b",
                    maps=summary)
    result = finish(session.sql(sql), oracle).collect()
    labeled = sum(len(rows) for _, rows in oracle.calls)
    rows = sorted(tuple(row.values()) for row in result.to_pylist())
    return rows, labeled + len(written)


def test_a_case_over_a_classification_labels_only_the_rows_it_reads(sessions):
    sql = f"SELECT r.id, CASE WHEN r.long THEN {TONE} END AS t FROM reviews r"
    assert "guard: r.long" in sessions[0].sql(sql).logical
    on, off = (_rows_and_questions(session, sql) for session in sessions)
    assert on[0] == off[0] == [("r0", "a"), ("r1", None), ("r2", None),
                               ("r3", "b"), ("r4", None), ("r5", "b")]
    # three rows are long; a NULL test asks nothing, as CASE reads it
    assert (on[1], off[1]) == (3, 6)


def test_a_case_over_a_map_writes_only_the_answers_it_reads(sessions):
    sql = ("SELECT r.id, CASE WHEN r.id <> 'r0' AND r.long "
           "THEN AI_MAP(r.body, 'Summarize.') END AS s FROM reviews r")
    on, off = (_rows_and_questions(session, sql) for session in sessions)
    assert on[0] == off[0] == [("r0", None), ("r1", None), ("r2", None),
                               ("r3", "summary 3"), ("r4", None), ("r5", "summary 5")]
    assert (on[1], off[1]) == (2, 6)


def test_a_case_and_its_answer_read_together_ask_every_row(sessions):
    sql = (f"SELECT r.id, CASE WHEN r.long THEN {TONE} END AS t, "
           f"{TONE} AS u FROM reviews r")
    assert "guard:" not in sessions[0].sql(sql).logical
    on, off = (_rows_and_questions(session, sql) for session in sessions)
    assert on == off
    assert on[1] == 6


def test_the_physical_plan_shows_a_classifications_guard(sessions):
    sql = f"SELECT r.id, CASE WHEN r.long THEN {TONE} END AS t FROM reviews r"
    physical = sessions[0].sql(sql).explain()
    assert "asks only where r.long" in physical


def test_stock_vllm_refuses_a_guarded_classification():
    session = quail.Session(
        EngineConfig(gpus=1, model="qwen3-4b-fp8", backend="stock_vllm",
                     device="h100-sxm"),
        tokenizer=TOKENIZER)
    session.register("reviews",
                     quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    plan = session.sql(
        f"SELECT r.id, CASE WHEN r.long THEN {TONE} END AS t FROM reviews r").plan()
    session.close()
    assert isinstance(plan, Refusal)
    assert plan.constraint == "guard_needs_quail_backend"
