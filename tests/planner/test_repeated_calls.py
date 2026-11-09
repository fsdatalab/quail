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
from quail.common.expressions import CompileError, SimilarCallWarning

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


def test_a_classification_in_a_derived_table_and_outside_it_labels_each_row_once(
        sessions):
    sql = (f"SELECT d.id, d.t, {TONE.replace('r.body', 'd.body')} AS u FROM "
           f"(SELECT r.id, r.body, {TONE} AS t FROM reviews r) d")
    assert sessions[0].sql(sql).logical.count("AiNode") == 1
    # without the rule, the derived table is a model region of its own
    assert len(sessions[1].sql(sql).regions()) == 2
    on, off = (_rows_and_questions(session, sql) for session in sessions)
    assert on[0] == off[0] == [(f"r{index}", label, label) for index, label in
                               enumerate("ababab")]
    # without the rule, the outer call labels every row again
    assert (on[1], off[1]) == (6, 12)


def test_a_query_of_two_model_regions_reports_each_region(sessions):
    sql = (f"SELECT d.id, d.t, {TONE.replace('r.body', 'd.body')} AS u FROM "
           f"(SELECT r.id, r.body, {TONE} AS t FROM reviews r) d")
    oracle = Oracle(labels=lambda spec, ids: "a")
    report = finish(sessions[1].sql(sql), oracle).report
    regions = report["regions"]
    assert len(regions) == 2
    # each region labels the six reviews
    assert [sum(node.get("evaluated_documents", 0)
                for node in region["node_metrics"].values())
            for region in regions] == [6, 6]


def test_an_extraction_in_select_and_its_field_in_where_asks_each_row_once(sessions):
    call = "AI_EXTRACT(r.body, {'w': 'When?'})"
    sql = f"SELECT r.id, {call} AS e FROM reviews r WHERE {call}['w'] IS NOT NULL"
    for session in sessions:
        assert session.sql(sql).logical.count("AiNode") == 1
        asked = []

        def when(spec, document):
            asked.append(document)
            return None if document % 2 else f"day {document}"

        result = finish(session.sql(sql), Oracle(answers=when)).collect()
        assert sorted(row["r.id"] for row in result.to_pylist()) == ["r0", "r2", "r4"]
        assert sorted(asked) == list(range(6))


def test_calls_that_differ_only_in_spelling_are_refused(sessions):
    sql = ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('kind: {0}', r.body)) "
           "AND AI_FILTER(PROMPT('Kind {0}', r.body))")
    with pytest.raises(CompileError, match="differ only in case, whitespace"):
        sessions[0].sql(sql).explain()


def test_calls_a_typo_apart_warn_and_both_run(sessions):
    sql = ("SELECT r.id FROM reviews r "
           "WHERE AI_FILTER(PROMPT('Is the review kind? {0}', r.body)) "
           "AND AI_FILTER(PROMPT('Is the reveiw kind? {0}', r.body))")
    with pytest.warns(SimilarCallWarning, match="differ in a few letters"):
        query = sessions[0].sql(sql)
        query.explain()
