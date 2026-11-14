"""A repeated AI call asks the model once per row.

AI_{f→b}(AI_{f→a}(R)) = π_{*, a→b}(AI_{f→a}(R)). The lowering gives
calls written the same way in one SELECT list and WHERE one AiNode;
merge_repeated_ai_calls merges an AiNode into the same call below it,
such as an AI.IF written twice. It runs before derive, so derive gives
the one node what every read of its answer needs. The transcript shows
the plan, the questions the model answered, and the rows
(helpers/transcript.py).
"""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish
from helpers.sessions import CONFIG, TOKENIZER
from helpers.transcript import Transcripts, assert_transcript

import quail
from quail.builtins import LOGICAL_RULES, built_in_registry, logical_rules_without
from quail.common.expressions import CompileError, SimilarCallWarning

RULE = "merge_repeated_ai_calls"
REVIEWS = pa.table({
    "id": ["r0", "r1", "r2", "r3", "r4", "r5"],
    "body": ["b0", "b1", "b2", "b3", "b4", "b5"],
})
AI_IF = "AI_FILTER(PROMPT('kind: {0}', r.body))"
TONE = "AI_CLASSIFY(PROMPT('tone: {0}', r.body), ['a', 'b'])"
WHEN = "AI_EXTRACT(r.body, {'w': 'When?'})"


def _session(rules) -> quail.Session:
    session = quail.Session(CONFIG, tokenizer=TOKENIZER,
                            registry=built_in_registry(logical_rules=rules))
    session.register("reviews", quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    return session


@pytest.fixture
def sessions():
    on = _session(LOGICAL_RULES)
    off = _session(logical_rules_without(RULE))
    yield on, off
    on.close()
    off.close()


def _answer(kind, prompt, rows):
    """Pass rows 1, 2, 4, and 5, label even rows 'a', and copy even rows' text."""
    index = int(rows[0][1:])
    if kind == "label":
        return "a" if index % 2 == 0 else "b"
    if kind == "extract":
        return None if index % 2 else f"b{index}"
    return index in (1, 2, 4, 5)


def test_a_call_written_twice_asks_the_model_once_per_row():
    with Transcripts({"reviews": REVIEWS}, _answer) as runs:
        assert_transcript("\n".join(runs.query(sql) for sql in (
            f"SELECT r.id FROM reviews r WHERE {AI_IF} AND {AI_IF}",
            f"SELECT r.id, {TONE} AS t FROM reviews r WHERE {TONE} = 'a'",
            # the derived table and the outer query are one model region
            f"SELECT d.id, d.t, {TONE.replace('r.body', 'd.body')} AS u FROM "
            f"(SELECT r.id, r.body, {TONE} AS t FROM reviews r) d",
            f"SELECT r.id, {WHEN} AS e FROM reviews r "
            f"WHERE {WHEN}['w'] IS NOT NULL")), """\
--- SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('kind: {0}', r.body)) AND
    AI_FILTER(PROMPT('kind: {0}', r.body)) ---
plan
  __quail_answer_1 = ai_filter('kind: {0}', r.body), filter: __quail_answer_1, returns:
      []
asked 6 of 10
  'kind: {0}': r0 FALSE, r1 TRUE, r2 TRUE, r3 FALSE, r4 TRUE, r5 TRUE
rows 4
  r1
  r2
  r4
  r5

--- SELECT r.id, AI_CLASSIFY(PROMPT('tone: {0}', r.body), ['a', 'b']) AS t FROM reviews
    r WHERE AI_CLASSIFY(PROMPT('tone: {0}', r.body), ['a', 'b']) = 'a' ---
plan
  t = ai_classify('tone: {0}', r.body, make_array('a', 'b')), filter: t = 'a'
asked 6 of 6
  'tone: {0}': r0 'a', r1 'b', r2 'a', r3 'b', r4 'a', r5 'b'
rows 3
  r0, 'a'
  r2, 'a'
  r4, 'a'

--- SELECT d.id, d.t, AI_CLASSIFY(PROMPT('tone: {0}', d.body), ['a', 'b']) AS u FROM
    (SELECT r.id, r.body, AI_CLASSIFY(PROMPT('tone: {0}', r.body), ['a', 'b']) AS t FROM
    reviews r) d ---
plan
  t = ai_classify('tone: {0}', r.body, make_array('a', 'b'))
asked 6 of 12
  'tone: {0}': r0 'a', r1 'b', r2 'a', r3 'b', r4 'a', r5 'b'
rows 6
  r0, 'a', 'a'
  r1, 'b', 'b'
  r2, 'a', 'a'
  r3, 'b', 'b'
  r4, 'a', 'a'
  r5, 'b', 'b'

--- SELECT r.id, AI_EXTRACT(r.body, {'w': 'When?'}) AS e FROM reviews r WHERE
    AI_EXTRACT(r.body, {'w': 'When?'})['w'] IS NOT NULL ---
plan
  e = ai_extract(r.body, named_struct('w', 'When?')), filter: get_field(e, 'w') IS NOT
      NULL, returns: [e]
asked 6 of 6
  'When?': r0 'b0', r1 NULL, r2 'b2', r3 NULL, r4 'b4', r5 NULL
rows 3
  r0, {'w': 'b0'}
  r2, {'w': 'b2'}
  r4, {'w': 'b4'}
""")


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


def test_calls_a_spelling_apart_are_refused_and_calls_a_typo_apart_warn(sessions):
    sql = ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('kind: {0}', r.body)) "
           "AND AI_FILTER(PROMPT('Kind {0}', r.body))")
    with pytest.raises(CompileError, match="differ only in case, whitespace"):
        sessions[0].sql(sql).explain()

    sql = ("SELECT r.id FROM reviews r "
           "WHERE AI_FILTER(PROMPT('Is the review kind? {0}', r.body)) "
           "AND AI_FILTER(PROMPT('Is the reveiw kind? {0}', r.body))")
    with pytest.warns(SimilarCallWarning, match="differ in a few letters"):
        query = sessions[0].sql(sql)
        query.explain()
