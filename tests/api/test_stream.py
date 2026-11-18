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
KIND = "AI_FILTER(PROMPT('kind: {0}', r.body))"
# a forward pass that holds one review: the model answers one at a time
SMALL_PASS = 128


def _answer(asked, prompt, rows):
    """The model keeps r0, r2, and r5."""
    (row,) = rows
    return row in ("r0", "r2", "r5")


def _transcript(*queries) -> str:
    with Transcripts({"reviews": REVIEWS}, _answer,
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


def test_a_stream_without_a_model_region_cuts_batches_and_stops_at_its_limit():
    with quail.Session(CONFIG, tokenizer=TOKENIZER) as session:
        session.register("reviews", quail.DocumentProvider.from_table(
            REVIEWS, id_col="id"))
        reader = session.sql("SELECT r.id FROM reviews r WHERE r.long "
                             "ORDER BY r.id").execute_stream(batch_rows=2, limit=3)
        assert [batch.num_rows for batch in reader] == [2, 1]
