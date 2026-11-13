"""Model operators over copied documents are asked once per distinct text.

The transcript, in transcripts/, names a text several reviews hold by
all of them, such as ``r0/r2/r4``, and compares the questions asked
with those of the run without deduplication (helpers/transcript.py).
"""

import pyarrow as pa
from helpers.transcript import Transcripts, assert_transcript

BODIES = ["noisy room", "cold room", "noisy room", "slow lift", "noisy room",
          "cold room"]
REVIEWS = pa.table({
    "id": [f"r{index}" for index in range(len(BODIES))],
    "body": BODIES,
    # a copy of a review can sit in another hotel than its first
    "hotel": ["h1", "h2", "h2", "h1", "h3", "h1"],
})


def _answer(kind, prompt, rows):
    """A review is short unless slow; two reviews agree when both are of rooms."""
    bodies = [BODIES[int(row.split("/")[0][1:])] for row in rows]
    if len(bodies) == 1:
        return not bodies[0].startswith("slow")
    return all("room" in body for body in bodies)


def test_a_filter_asks_once_per_text_and_a_join_once_per_pair_of_texts():
    short = ("SELECT r.id FROM reviews r "
             "WHERE AI_FILTER(PROMPT('Is {0} short?', r.body))")
    self_join = ("SELECT a.id, b.id FROM reviews a JOIN reviews b "
                 "ON a.hotel = b.hotel "
                 "AND AI_FILTER(PROMPT('Do {0} and {1} agree?', a.body, b.body))")
    with Transcripts({"reviews": REVIEWS}, _answer, pipelines=True) as runs:
        filtered = runs.query(short)
    with Transcripts({"reviews": REVIEWS}, _answer) as runs:
        joined = runs.query(self_join)
    assert_transcript(filtered + "\n" + joined)
