"""A filter of a table's columns runs before the AI operator above it.

σ_q(AI(R)) = AI(σ_q(R)) when q reads only R's columns and the operator
has no fetch. Quail moves such a filter to its table while it finds the
model regions, before DataFusion's optimizer runs, and DataFusion's
push_down_filter moves it further down. derive gives an extraction whose
answer must contain a literal the filter that its document contains it.
The transcripts, in transcripts/, show the plan, the questions the model
answered, and the rows (helpers/transcript.py).
"""

import pyarrow as pa
from helpers.transcript import Transcripts, assert_transcript

REVIEWS = pa.table({
    "id": ["r0", "r1", "r2", "r3", "r4", "r5"],
    "stars": [5, 2, 4, 1, 5, 3],
    "body": ["b0", "b1", "b2", "b3", "b4", "b5"],
})
KIND = "AI_FILTER(PROMPT('kind: {0}', r.body))"
TONE = "AI_CLASSIFY(PROMPT('tone: {0}', r.body), ['a', 'b'])"
LEASES = pa.table({
    "id": ["c0", "c1", "c2", "c3", "c4"],
    "text": ["Ends in 2027.", "Ends in 2026.", "2027 start", "No date.",
             "ENDS 2027"],
})


def test_a_filter_above_an_ai_filter_or_classification_runs_before_the_model():
    def answer(kind, prompt, rows):
        index = int(rows[0][1:])
        if kind == "label":
            return "a" if index % 2 == 0 else "b"
        return index in (1, 2, 4, 5)

    with Transcripts({"reviews": REVIEWS}, answer) as runs:
        assert_transcript("\n".join(runs.query(sql) for sql in (
            f"SELECT d.id FROM (SELECT r.id, r.stars FROM reviews r WHERE {KIND}) d "
            "WHERE d.stars > 3",
            f"SELECT d.id, d.t FROM (SELECT r.id, r.stars, {TONE} AS t "
            "FROM reviews r) d WHERE d.stars > 3")))


def test_an_extraction_tested_with_like_reads_only_documents_with_its_literal():
    def copy_year(kind, prompt, rows):
        (text,) = [text for row, text in zip(LEASES["id"].to_pylist(),
                                             LEASES["text"].to_pylist(),
                                             strict=True) if row == rows[0]]
        return next((word for word in text.rstrip(".").split() if word.isdigit()),
                    None)

    call = "AI_EXTRACT(l.text, {'year': 'When does it end?'})"
    year = call + "['year']"
    with Transcripts({"leases": LEASES}, copy_year) as runs:
        assert_transcript("\n".join(runs.query(
            f"SELECT l.id, {call} AS e FROM leases l WHERE {test.format(year)}")
            # LIKE '%2027%' needs "2027", and ILIKE '20_7' needs "20"
            for test in ("{} LIKE '%2027%'", "{} ILIKE '20_7'")))
