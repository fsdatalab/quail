"""A guarded AI call asks the model only where its guard is TRUE.

An AI operator's guard g says which rows the model is asked about; every
other row's answer is NULL. derive makes the test of a
CASE that reads the answer the guard (R6). Each test runs a query with
the rule on and off, checks that both return the same rows, and counts
the questions the model answered. derive also guards an AI
filter in an OR with the OR's other terms (R7); in an OR of several AI
filters, each is asked only where the terms before it are not TRUE.
"""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish
from helpers.sessions import CONFIG, TOKENIZER

import quail
from quail.api.config import EngineConfig
from quail.builtins import LOGICAL_RULES, built_in_registry, logical_rules_without
from quail.physical import AiClassify, AiFilter
from quail.physical.plan import Refusal

CASE_RULE = "derive.guard"
REVIEWS = pa.table({
    "id": ["r0", "r1", "r2", "r3", "r4", "r5"],
    "body": ["b0", "b1", "b2", "b3", "b4", "b5"],
    "long": [True, False, None, True, False, True],
})
TONE = "AI_CLASSIFY(PROMPT('tone: {0}', r.body), ['a', 'b'])"
KIND = "AI_FILTER(PROMPT('kind: {0}', r.body))"


def _session(rules) -> quail.Session:
    session = quail.Session(CONFIG, tokenizer=TOKENIZER,
                            registry=built_in_registry(logical_rules=rules))
    session.register("reviews",
                     quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    return session


@pytest.fixture
def sessions():
    on = _session(LOGICAL_RULES)
    off = _session(logical_rules_without(CASE_RULE))
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
    assert "asks only where r.long" in sessions[0].sql(sql).explain()

    above_a_cte = (f"WITH t AS (SELECT r.id, r.long, {TONE} AS tone FROM reviews r) "
                   "SELECT t.id, CASE WHEN t.long THEN t.tone END AS tone FROM t")
    assert "guard: r.long" in sessions[0].sql(above_a_cte).logical
    assert [_rows_and_questions(session, above_a_cte)
            for session in sessions] == [on, off]


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


def test_stock_vllm_refuses_a_guarded_classification_and_a_guarded_ai_filter():
    session = quail.Session(
        EngineConfig(gpus=1, model="qwen3-4b-fp8", backend="stock_vllm",
                     device="h100-sxm"),
        tokenizer=TOKENIZER)
    session.register("reviews",
                     quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    plans = {
        "guard_needs_quail_backend": session.sql(
            f"SELECT r.id, CASE WHEN r.long THEN {TONE} END AS t "
            "FROM reviews r").plan(),
        "or_filter_needs_quail_backend": session.sql(
            f"SELECT r.id FROM reviews r WHERE r.long OR {KIND}").plan(),
    }
    session.close()
    for constraint, plan in plans.items():
        assert isinstance(plan, Refusal), constraint
        assert plan.constraint == constraint


OR_RULE = "derive.guard"


def _or_rows_and_questions(rules, sql) -> tuple[list, list]:
    """Return the query's ids, sorted, and the rows the model was asked about."""
    asked = []

    def kind(alias, document, written_position):
        asked.append(document)
        return document % 2 == 0

    session = _session(rules)
    try:
        result = finish(session.sql(sql), Oracle(filters=kind)).collect()
    finally:
        session.close()
    return sorted(row["r.id"] for row in result.to_pylist()), sorted(asked)


def test_an_or_of_a_column_and_an_ai_filter_asks_only_where_the_column_is_not_true(
        sessions):
    sql = f"SELECT r.id FROM reviews r WHERE r.long OR {KIND}"
    off = logical_rules_without(OR_RULE)
    on_rows, on_asked = _or_rows_and_questions(LOGICAL_RULES, sql)
    off_rows, off_asked = _or_rows_and_questions(off, sql)
    # r0, r3, r5 are long; of the rest, r2 and r4 are kind
    assert on_rows == off_rows == ["r0", "r2", "r3", "r4", "r5"]
    assert (on_asked, off_asked) == ([1, 2, 4], [0, 1, 2, 3, 4, 5])
    query = sessions[0].sql(sql)
    assert "guard: r.long IS NOT TRUE" in query.logical
    physical = query.explain()
    assert "asks only where" in physical
    assert "passes each row where r.long" in physical


def test_an_or_with_a_null_column_asks_the_model_about_the_null_row():
    sql = f"SELECT r.id FROM reviews r WHERE {KIND} OR (r.long AND r.id <> 'r0')"
    on_rows, on_asked = _or_rows_and_questions(LOGICAL_RULES, sql)
    off_rows, _ = _or_rows_and_questions(
        logical_rules_without(OR_RULE), sql)
    assert on_rows == off_rows == ["r0", "r2", "r3", "r4", "r5"]
    assert on_asked == [0, 1, 2, 4]


LONG = "AI_FILTER(PROMPT('long: {0}', r.body))"


def _ors_rows_and_questions(rules, sql, kind, long) -> tuple[list, list, list]:
    """Return the query's ids, sorted, and the rows each AI.IF was asked about.

    Args:
        rules: The logical rules the session runs.
        sql: The query.
        kind: The documents the first AI.IF accepts.
        long: The documents the second AI.IF accepts.
    """
    asked = ([], [])

    def answer(alias, document, written_position):
        asked[written_position].append(document)
        return document in (kind, long)[written_position]

    session = _session(rules)
    try:
        result = finish(session.sql(sql), Oracle(filters=answer)).collect()
    finally:
        session.close()
    return (sorted(row["r.id"] for row in result.to_pylist()),
            sorted(asked[0]), sorted(asked[1]))


def test_an_or_of_two_ai_filters_asks_the_second_only_where_the_first_is_not_true():
    sql = f"SELECT r.id FROM reviews r WHERE {KIND} OR {LONG}"
    off = logical_rules_without(OR_RULE)
    on = _ors_rows_and_questions(LOGICAL_RULES, sql, {0, 3}, {2, 3, 4})
    everything = _ors_rows_and_questions(off, sql, {0, 3}, {2, 3, 4})
    assert on[0] == everything[0] == ["r0", "r2", "r3", "r4"]
    # the first AI.IF is asked about every row, the second where it said no
    assert on[1:] == ([0, 1, 2, 3, 4, 5], [1, 2, 4, 5])
    assert everything[1:] == ([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])


def test_the_planner_runs_the_term_of_an_or_that_passes_more_rows_first():
    often = "AI_FILTER(PROMPT('long: {0}', r.body), {'selectivity': 0.9})"
    sql = f"SELECT r.id FROM reviews r WHERE {KIND} OR {often}"
    rows, kind, long = _ors_rows_and_questions(LOGICAL_RULES, sql, {0, 3}, {2, 3, 4})
    assert rows == ["r0", "r2", "r3", "r4"]
    # the second call written runs first; the first is asked where it said no
    assert (kind, long) == ([0, 1, 5], [0, 1, 2, 3, 4, 5])


def test_an_or_ranks_as_one_filter_among_the_filters_beside_it(sessions):
    kind = "AI_FILTER(PROMPT('kind: {0}', r.body), {'selectivity': 0.2})"
    long = "AI_FILTER(PROMPT('long: {0}', r.body), {'selectivity': 0.6})"
    for third_passes, printed in (
            # the third filter drops half the rows, more than the OR's 32%: it
            # runs first, then the OR's term that passes more rows
            (0.5, ["kind:", "long:", "third:"]),
            # it drops a tenth of the rows, so the OR runs first, its terms
            # together
            (0.9, ["third:", "kind:", "long:"])):
        third = (f"AI_FILTER(PROMPT('third: {{0}}', r.body), "
                 f"{{'selectivity': {third_passes}}})")
        query = sessions[0].sql(
            f"SELECT r.id FROM reviews r WHERE ({kind} OR {long}) AND {third}")
        # explain() prints the last filter to run first
        physical = query.explain().split("physical:")[1]
        assert sorted(printed, key=physical.index) == printed, third_passes


def test_an_or_of_a_column_and_two_ai_filters_asks_each_where_no_earlier_term_is_true(
        sessions):
    sql = f"SELECT r.id FROM reviews r WHERE r.long OR {KIND} OR {LONG}"
    off = logical_rules_without(OR_RULE)
    on = _ors_rows_and_questions(LOGICAL_RULES, sql, {4}, {1})
    everything = _ors_rows_and_questions(off, sql, {4}, {1})
    # r0, r3, r5 are long: TRUE OR NULL is TRUE. r2's long is NULL and both
    # AI.IFs say no: NULL OR FALSE OR FALSE is NULL, and the row is dropped
    assert on[0] == everything[0] == ["r0", "r1", "r3", "r4", "r5"]
    # the planner picks which AI.IF runs first; the other is asked only
    # where the first said no
    assert on[1:] in (([1, 2, 4], [1, 2]), ([2, 4], [1, 2, 4]))
    assert everything[1:] == ([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])
    query = sessions[0].sql(sql)
    assert "guard: r.long IS NOT TRUE AND __quail_answer_1 IS NOT TRUE" in query.logical
    physical = query.explain()
    assert "keeps every row for the later terms of its OR" in physical
    assert "asks only the rows no earlier term of its OR passed" in physical
    assert "returns the rows a term of its OR passed" in physical
    assert "passes each row where r.long" in physical


def test_a_guarded_ai_call_is_priced_on_the_rows_its_guard_passes(sessions):
    def classified_rows(sql):
        plan = sessions[0].sql(sql).plan()
        (node,) = [node for node in plan.nodes if isinstance(node, AiClassify)]
        return node.spec.expected_rows, plan.estimated_seconds

    guarded = classified_rows(
        f"SELECT r.id, CASE WHEN r.long THEN {TONE} END AS t FROM reviews r")
    everything = classified_rows(f"SELECT r.id, {TONE} AS t FROM reviews r")
    # r.long passes the default 20% of rows
    assert guarded[0] == pytest.approx(0.2 * everything[0])
    assert guarded[1] < everything[1]

    sql = f"SELECT r.id FROM reviews r WHERE r.long OR {KIND}"
    (guarded,) = [node for node in sessions[0].sql(sql).plan().nodes
                  if isinstance(node, AiFilter)]
    unguarded = _session(logical_rules_without(OR_RULE))
    (everything,) = [node for node in unguarded.sql(sql).plan().nodes
                     if isinstance(node, AiFilter)]
    unguarded.close()
    # r.long IS NOT TRUE passes 1 - 20% of rows
    assert guarded.expected_rows == pytest.approx(0.8 * everything.expected_rows)


EXTRACT = "AI.EXTRACT(r.body, {'w': 'Which word?'})"


def test_a_case_over_an_extraction_is_null_where_its_test_is_not_true(sessions):
    sql = f"SELECT r.id, CASE WHEN r.long THEN {EXTRACT} END AS e FROM reviews r"
    assert "guard: r.long" in sessions[0].sql(sql).logical
    found = []
    for session in sessions:
        asked = []

        def answer(spec, document, asked=asked):
            asked.append(document)
            return f"b{document}"

        result = finish(session.sql(sql), Oracle(answers=answer)).collect()
        found.append((sorted(tuple(row.values()) for row in result.to_pylist()),
                      sorted(asked)))
    on, off = found
    # an unasked row is NULL as a whole, not a struct of NULL answers
    assert on[0] == off[0] == [
        ("r0", {"w": "b0"}), ("r1", None), ("r2", None), ("r3", {"w": "b3"}),
        ("r4", None), ("r5", {"w": "b5"})]
    assert (on[1], off[1]) == ([0, 3, 5], [0, 1, 2, 3, 4, 5])
