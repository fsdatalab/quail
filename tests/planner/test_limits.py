"""A LIMIT above an AI filter stops the model once enough rows pass.

derive makes a LIMIT of k rows after an OFFSET of m the fetch k + m of
the AI operator whose rows it reads. A LIMIT with no offset then leaves
the plan; one with an offset stays. A filter of the operator's input
moves below it, as the fetch counts the rows that pass the filter. A
HINT's selectivity is an estimate, so a node with one takes no fetch.
The transcripts, in transcripts/, show the plan, the questions the
model answered, and the rows (helpers/transcript.py).
"""

import pyarrow as pa
from helpers.transcript import Transcripts, assert_transcript

REVIEWS = pa.table({
    "id": [f"r{index}" for index in range(6)],
    "body": [f"b{index}" for index in range(6)],
    "stars": [1, 4, 5, 2, 5, 3],
})
KIND = "AI_FILTER(PROMPT('kind: {0}', r.body))"


def test_a_limit_over_an_ai_filter_stops_the_model_once_enough_rows_pass():
    def answer(kind, prompt, rows):
        return rows[0] in ("r1", "r2", "r4", "r5")

    with Transcripts({"reviews": REVIEWS}, answer) as runs:
        assert_transcript("\n".join(runs.query(sql) for sql in (
            f"SELECT r.id FROM reviews r WHERE {KIND} LIMIT 2",
            f"SELECT r.id FROM reviews r WHERE {KIND} LIMIT 2 OFFSET 1",
            f"SELECT r.id FROM reviews r WHERE r.stars > 3 AND {KIND} LIMIT 2",
            f"SELECT r.id FROM reviews r WHERE HINT({KIND}, {{'selectivity': 0.5}}) "
            "LIMIT 2")))
