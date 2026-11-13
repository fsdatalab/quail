"""A LIMIT above an AI operator stops the model once enough rows pass.

derive makes a LIMIT of k rows after an OFFSET of m the fetch k + m of
the AI operator whose rows it reads. A LIMIT with no offset then leaves
the plan; one with an offset stays. A filter of the operator's input
moves below it, as the fetch counts the rows that pass the filter; a
later test of the answer, or of the input after the LIMIT, stays above
it. A LIMIT over a projection of an answer, or over a CTE, is the fetch
of the operator below; a LIMIT over a UNION ALL is the fetch of each
branch. A HINT's selectivity is an estimate, so a node with one takes no
fetch. The transcripts, in transcripts/, show the plan, the questions
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
        "AND AI_FILTER(PROMPT('long: {0}', r.body)) LIMIT 1"))


def test_a_fetch_keeps_a_later_test_of_the_answer_or_the_input_above_it():
    assert_transcript(_transcript(
        f"SELECT d.id FROM (SELECT r.id, {TONE} AS t FROM reviews r LIMIT 3) d "
        "WHERE d.t = 'a'",
        f"SELECT d.id FROM (SELECT r.id, r.stars FROM reviews r WHERE {KIND} "
        "LIMIT 3) d WHERE d.stars > 3"))


def test_a_limit_over_a_projection_or_a_union_is_the_fetch_below_it():
    assert_transcript(_transcript(
        f"SELECT r.id, CONCAT({TONE}, '!') AS t FROM reviews r LIMIT 2",
        # three rows pass, so which branch runs first does not change the rows
        f"SELECT r.id FROM reviews r WHERE r.stars > 4 AND {KIND} UNION ALL "
        "SELECT n.id FROM notes n WHERE n.id = 'n1' "
        "AND AI_FILTER(PROMPT('kind: {0}', n.body)) LIMIT 3"))
