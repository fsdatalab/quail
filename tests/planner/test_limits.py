"""A LIMIT above an AI operator stops the model once enough rows pass.

derive makes a LIMIT of k rows after an OFFSET of m the fetch k + m of
the AI operator whose rows it reads. A LIMIT with no offset then leaves
the plan; one with an offset stays. A filter of the operator's input
moves below it, as the fetch counts the rows that pass the filter; a
later test of the answer, or of the input after the LIMIT, stays above
it. A LIMIT over a projection of an answer, or over a CTE, is the fetch
of the operator below; a LIMIT over a UNION ALL is the fetch of each
branch. A HINT's selectivity is an estimate, so a node with one takes no
fetch. The transcripts show the plan, the questions
the model answered, and the rows (helpers/transcript.py).
"""

import pyarrow as pa
from helpers.transcript import Transcripts, assert_transcript

REVIEWS = pa.table({
    "id": [f"r{index}" for index in range(6)],
    "body": [f"b{index}" for index in range(6)],
    "stars": [1, 4, 5, 2, 5, 3],
})
NOTES = pa.table({
    "id": [f"n{index}" for index in range(6)],
    "body": [f"c{index}" for index in range(6)],
})
KIND = "AI_FILTER(PROMPT('kind: {0}', r.body))"
TONE = "AI_CLASSIFY(PROMPT('tone: {0}', r.body), ['a', 'b'])"


def _answer(kind, prompt, rows):
    """Pass rows 1, 2, 4, and 5 of each table, and label even rows 'a'.

    The 'long' question passes rows 4 and 5.
    """
    index = int(rows[0][1:])
    if kind == "label":
        return "a" if index % 2 == 0 else "b"
    if prompt.startswith("long"):
        return index >= 4
    return index in (1, 2, 4, 5)


def _transcript(*queries) -> str:
    with Transcripts({"reviews": REVIEWS, "notes": NOTES}, _answer,
                     chunk_tokens=256) as runs:
        return "\n".join(runs.query(sql) for sql in queries)


def test_a_limit_over_an_ai_filter_stops_the_model_once_enough_rows_pass():
    assert_transcript(_transcript(
        f"SELECT r.id FROM reviews r WHERE {KIND} LIMIT 2",
        f"SELECT r.id FROM reviews r WHERE {KIND} LIMIT 2 OFFSET 1",
        # more rows than pass
        f"SELECT r.id FROM reviews r WHERE {KIND} LIMIT 10",
        f"WITH kind AS (SELECT r.id FROM reviews r WHERE {KIND}) "
        "SELECT kind.id FROM kind LIMIT 2",
        f"SELECT r.id FROM reviews r WHERE r.stars > 3 AND {KIND} LIMIT 2",
        f"SELECT r.id FROM reviews r WHERE HINT({KIND}, {{'selectivity': 0.5}}) "
        "LIMIT 2",
        # the first rows the first filter passes do not pass the second
        f"SELECT r.id FROM reviews r WHERE {KIND} "
        "AND AI_FILTER(PROMPT('long: {0}', r.body)) LIMIT 1"), """\
--- SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('kind: {0}', r.body)) LIMIT 2 ---
plan
  __quail_answer_1 = ai_filter('kind: {0}', r.body), filter: __quail_answer_1, fetch: 2,
      returns: []
asked 4 of 6
  'kind: {0}': r0 FALSE, r1 TRUE, r2 TRUE, r3 FALSE
rows 2
  r1
  r2

--- SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('kind: {0}', r.body)) LIMIT 2
    OFFSET 1 ---
plan
  __quail_answer_1 = ai_filter('kind: {0}', r.body), filter: __quail_answer_1, fetch: 3,
      returns: []
asked 6 of 6
  'kind: {0}': r0 FALSE, r1 TRUE, r2 TRUE, r3 FALSE, r4 TRUE, r5 TRUE
rows 2
  r2
  r4

--- SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('kind: {0}', r.body)) LIMIT 10 ---
plan
  __quail_answer_1 = ai_filter('kind: {0}', r.body), filter: __quail_answer_1, fetch:
      10, returns: []
asked 6 of 6
  'kind: {0}': r0 FALSE, r1 TRUE, r2 TRUE, r3 FALSE, r4 TRUE, r5 TRUE
rows 4
  r1
  r2
  r4
  r5

--- WITH kind AS (SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('kind: {0}',
    r.body))) SELECT kind.id FROM kind LIMIT 2 ---
plan
  __quail_answer_1 = ai_filter('kind: {0}', r.body), filter: __quail_answer_1, fetch: 2,
      returns: []
asked 4 of 6
  'kind: {0}': r0 FALSE, r1 TRUE, r2 TRUE, r3 FALSE
rows 2
  r1
  r2

--- SELECT r.id FROM reviews r WHERE r.stars > 3 AND AI_FILTER(PROMPT('kind: {0}',
    r.body)) LIMIT 2 ---
plan
  __quail_answer_1 = ai_filter('kind: {0}', r.body), filter: __quail_answer_1, fetch: 2,
      returns: []
asked 3 of 3
  'kind: {0}': r1 TRUE, r2 TRUE, r4 TRUE
rows 2
  r1
  r2

--- SELECT r.id FROM reviews r WHERE HINT(AI_FILTER(PROMPT('kind: {0}', r.body)),
    {'selectivity': 0.5}) LIMIT 2 ---
plan
  __quail_answer_1 = ai_filter('kind: {0}', r.body), filter: __quail_answer_1, hint:
      named_struct('selectivity', Float64(0.5)), returns: []
asked 4 of 6
  'kind: {0}': r0 FALSE, r1 TRUE, r2 TRUE, r3 FALSE
rows 2
  r1
  r2

--- SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('kind: {0}', r.body)) AND
    AI_FILTER(PROMPT('long: {0}', r.body)) LIMIT 1 ---
plan
  __quail_answer_2 = ai_filter('long: {0}', r.body), filter: __quail_answer_2, fetch: 1,
      returns: []
  __quail_answer_1 = ai_filter('kind: {0}', r.body), filter: __quail_answer_1, returns:
      []
asked 10 of 10
  'kind: {0}': r0 FALSE, r1 TRUE, r2 TRUE, r3 FALSE, r4 TRUE, r5 TRUE
  'long: {0}': r1 FALSE, r2 FALSE, r4 TRUE, r5 TRUE
rows 1
  r4
""")


def test_a_fetch_keeps_a_later_test_of_the_answer_or_the_input_above_it():
    assert_transcript(_transcript(
        f"SELECT d.id FROM (SELECT r.id, {TONE} AS t FROM reviews r LIMIT 3) d "
        "WHERE d.t = 'a'",
        f"SELECT d.id FROM (SELECT r.id, r.stars FROM reviews r WHERE {KIND} "
        "LIMIT 3) d WHERE d.stars > 3"), """\
--- SELECT d.id FROM (SELECT r.id, AI_CLASSIFY(PROMPT('tone: {0}', r.body), ['a', 'b'])
    AS t FROM reviews r LIMIT 3) d WHERE d.t = 'a' ---
plan
  t = ai_classify('tone: {0}', r.body, make_array('a', 'b')), fetch: 3
asked 3 of 6
  'tone: {0}': r0 'a', r1 'b', r2 'a'
rows 2
  r0
  r2

--- SELECT d.id FROM (SELECT r.id, r.stars FROM reviews r WHERE AI_FILTER(PROMPT('kind:
    {0}', r.body)) LIMIT 3) d WHERE d.stars > 3 ---
plan
  __quail_answer_1 = ai_filter('kind: {0}', r.body), filter: __quail_answer_1, fetch: 3,
      returns: []
asked 6 of 6
  'kind: {0}': r0 FALSE, r1 TRUE, r2 TRUE, r3 FALSE, r4 TRUE, r5 TRUE
rows 3
  r1
  r2
  r4
""")


def test_a_limit_over_a_projection_or_a_union_is_the_fetch_below_it():
    assert_transcript(_transcript(
        f"SELECT r.id, CONCAT({TONE}, '!') AS t FROM reviews r LIMIT 2",
        # three rows pass, so which branch runs first does not change the rows
        f"SELECT r.id FROM reviews r WHERE r.stars > 4 AND {KIND} UNION ALL "
        "SELECT n.id FROM notes n WHERE n.id = 'n1' "
        "AND AI_FILTER(PROMPT('kind: {0}', n.body)) LIMIT 3"), """\
--- SELECT r.id, CONCAT(AI_CLASSIFY(PROMPT('tone: {0}', r.body), ['a', 'b']), '!') AS t
    FROM reviews r LIMIT 2 ---
plan
  __quail_call_1 = ai_classify('tone: {0}', r.body, make_array('a', 'b')), fetch: 2
asked 2 of 6
  'tone: {0}': r0 'a', r1 'b'
rows 2
  r0, 'a!'
  r1, 'b!'

--- SELECT r.id FROM reviews r WHERE r.stars > 4 AND AI_FILTER(PROMPT('kind: {0}',
    r.body)) UNION ALL SELECT n.id FROM notes n WHERE n.id = 'n1' AND
    AI_FILTER(PROMPT('kind: {0}', n.body)) LIMIT 3 ---
plan
  __quail_answer_1 = ai_filter('kind: {0}', r.body), filter: __quail_answer_1, fetch: 3,
      returns: []
  __quail_answer_2 = ai_filter('kind: {0}', n.body), filter: __quail_answer_2, fetch: 3,
      returns: []
asked 3 of 3
  'kind: {0}': n1 TRUE, r2 TRUE, r4 TRUE
rows 3
  n1
  r2
  r4
""")
