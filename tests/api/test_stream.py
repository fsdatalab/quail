"""A streamed query hands out each row as the model finishes it.

``Query.execute_stream`` reads DataFusion's rows as the operators
produce them, while the query's model regions run on the workers. Each
transcript shows the questions the model answered and, between them,
each batch of rows the reader received (helpers/transcript.py).
"""

import pyarrow as pa
import pytest
from helpers.sessions import CONFIG, TOKENIZER
from helpers.transcript import Transcripts, assert_transcript

import quail

pytest.importorskip("_quail_native")

REVIEWS = pa.table({
    "id": ["r0", "r1", "r2", "r3", "r4", "r5"],
    "body": [f"review number {index} " * 4 for index in range(6)],
    "long": [True, False, True, True, False, True],
})
# r3 repeats r0's review and r5 repeats r2's
REPEATS = REVIEWS.set_column(1, "body", pa.array(
    [f"review number {index} " * 4 for index in (0, 1, 2, 0, 4, 2)]))
# r1 has no review: the model is not asked about it
MISSING = REVIEWS.set_column(1, "body", pa.array(
    [None if index == 1 else f"review number {index} " * 4 for index in range(6)]))
# agent turns and user replies: t2 and t3 have no agent turn, t4 repeats
# t0's pair, and t5 repeats t1's reply
REPLIES = pa.table({
    "turn_id": ["t0", "t1", "t2", "t3", "t4", "t5"],
    "session_id": ["s0", "s8", "s0", "s1", "s8", "s0"],
    "previous_message": [f"agent turn {index} " * 2 if index is not None else None
                         for index in (0, 1, None, None, 0, 5)],
    "content": [f"user reply {index} " * 2 for index in (0, 1, 2, 3, 0, 1)],
})
KIND = "AI_FILTER(PROMPT('kind: {0}', r.body))"
# a forward pass that holds one review: the model answers one at a time
SMALL_PASS = 128


def _answer(asked, prompt, rows):
    """The model keeps r0, r2, and r5, and every copy of their text."""
    (row,) = rows
    return any(name in ("r0", "r2", "r5") for name in row.split("/"))


def _transcript(*queries, reviews=REVIEWS) -> str:
    with Transcripts({"reviews": reviews}, _answer,
                     chunk_tokens=SMALL_PASS) as runs:
        return "\n".join(runs.stream(sql) for sql in queries)


def test_an_ai_filter_streams_each_kept_row_before_the_model_asks_the_next():
    assert_transcript(_transcript(
        f"SELECT r.id FROM reviews r WHERE r.long AND {KIND}"), """\
--- SELECT r.id FROM reviews r WHERE r.long AND AI_FILTER(PROMPT('kind: {0}', r.body))
    ---
streamed
  asked 'kind: {0}': r0 TRUE
  asked 'kind: {0}': r2 TRUE
  received r0
  received r2
  asked 'kind: {0}': r3 FALSE
  asked 'kind: {0}': r5 TRUE
  received r5
rows 3
""")


def test_an_ai_filter_over_repeated_reviews_streams_every_copy_it_keeps():
    assert_transcript(_transcript(
        f"SELECT r.id FROM reviews r WHERE r.long AND {KIND}", reviews=REPEATS), """\
--- SELECT r.id FROM reviews r WHERE r.long AND AI_FILTER(PROMPT('kind: {0}', r.body))
    ---
streamed
  asked 'kind: {0}': r0/r3 TRUE
  received r0, r3
  asked 'kind: {0}': r2/r5 TRUE
  received r2, r5
rows 4
""")


def test_an_ai_filter_over_a_missing_review_streams_each_kept_row():
    assert_transcript(_transcript(
        f"SELECT r.id FROM reviews r WHERE {KIND}", reviews=MISSING), """\
--- SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('kind: {0}', r.body)) ---
streamed
  asked 'kind: {0}': r0 TRUE
  asked 'kind: {0}': r2 TRUE
  received r0
  received r2
  asked 'kind: {0}': r3 FALSE
  asked 'kind: {0}': r4 FALSE
  asked 'kind: {0}': r5 TRUE
  received r5
rows 3
""")


def _angry(asked, prompt, rows):
    """The model finds the user angry after t0's and t5's agent turns."""
    return any(name in ("t0", "t5") for name in rows[0].split("/"))


def test_an_ai_filter_over_a_pair_of_columns_streams_each_kept_row():
    # a pass that holds one agent turn and its reply
    with Transcripts({"replies": (REPLIES, "turn_id")}, _angry,
                     chunk_tokens=2 * SMALL_PASS) as runs:
        assert_transcript(runs.stream(
            "SELECT r.turn_id, r.session_id, r.content FROM replies r "
            "WHERE right(r.session_id, 1) IN ('0', '8') AND AI.IF(PROMPT("
            "'Agent: {0} User: {1} Is the user angry?', r.previous_message, "
            "r.content))", dialect="bq"), """\
--- SELECT r.turn_id, r.session_id, r.content FROM replies r WHERE right(r.session_id,
    1) IN ('0', '8') AND AI.IF(PROMPT('Agent: {0} User: {1} Is the user angry?',
    r.previous_message, r.content)) ---
streamed
  asked 'Agent: {0} User: {1} Is the user angry?': t0/t4+t0/t4 TRUE
  asked 'Agent: {0} User: {1} Is the user angry?': t1+t1/t5 FALSE
  received t0, t4
  asked 'Agent: {0} User: {1} Is the user angry?': t5+t1/t5 TRUE
  received t5
rows 3
""")


def test_a_stream_without_a_model_region_cuts_batches_and_stops_at_its_limit():
    with quail.Session(CONFIG, tokenizer=TOKENIZER) as session:
        session.register("reviews", quail.DocumentProvider.from_table(
            REVIEWS, id_col="id"))
        reader = session.sql("SELECT r.id FROM reviews r WHERE r.long "
                             "ORDER BY r.id").execute_stream(batch_rows=2, limit=3)
        assert [batch.num_rows for batch in reader] == [2, 1]
