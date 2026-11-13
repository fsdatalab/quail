"""DataFusion's DISTINCT rules in a model region.

derive drops a DISTINCT that removes no row, and gives the AI operator
under any other DISTINCT a fetch of one row per value of the selected
columns. The transcripts, in transcripts/, show the plan, the questions
the model answered, and the rows (helpers/transcript.py).
"""


import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish
from helpers.sessions import CONFIG, TOKENIZER
from helpers.transcript import Transcripts, assert_transcript

import quail

REVIEWS = pa.table({"id": ["r0", "r1"], "stars": [5, 2], "body": ["good", "slow"]})


@pytest.fixture()
def session():
    value = quail.Session(CONFIG, tokenizer=TOKENIZER)
    value.register("reviews", quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    yield value
    value.close()


def test_a_distinct_that_removes_no_row_is_eliminated(session):
    for text in (
            "SELECT DISTINCT r.id FROM reviews r "
            "WHERE AI_FILTER(PROMPT('q1: {0}', r.body))",
            "SELECT DISTINCT r.stars AS k, COUNT(*) AS n FROM reviews r "
            "WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) GROUP BY r.stars"):
        # DataFusion's plan of the region, as explain() shows it
        assert "Distinct" not in session.sql(text).logical, text


def test_a_limit_above_a_distinct_of_unique_keys_stops_the_model(session):
    session.register("aspects", quail.DocumentProvider.from_table(
        pa.table({"id": ["a0", "a1"], "aspect": ["price", "speed"]}), id_col="id"))
    for text in (
            "SELECT DISTINCT r.id, r.body FROM reviews r "
            "WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) LIMIT 1",
            "SELECT DISTINCT r.id, a.id FROM reviews r JOIN aspects a "
            "ON AI_FILTER(PROMPT('q2: {0} {1}', r.body, a.aspect)) LIMIT 1"):
        (region,) = quail.SessionContext(session)._native.explain_regions(
            text, None, False)
        assert "Limit: 1" in region and "stop per key" not in region, text
    # rows of one table read twice repeat, so that DISTINCT stays
    text = ("SELECT DISTINCT u.id FROM (SELECT r.id FROM reviews r "
            "WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) UNION ALL "
            "SELECT r.id FROM reviews r) u")
    passing = Oracle(filters=lambda alias, document, position: True)
    found = finish(session.sql(text), passing).collect()
    assert sorted(found.column(0).to_pylist()) == ["r0", "r1"]


KEYED = pa.table({
    "id": ["k0", "k1", "k2", "k3", "k4", "k5"],
    "stars": [5, 5, 4, 4, 3, 3],
    "body": ["b0", "b1", "b2", "b3", "b4", "b5"],
})
AI_IF = "AI_FILTER(PROMPT('kind: {0}', r.body))"


def _answer(kind, prompt, rows):
    """The model passes k1 to k4, and labels k0 'a', k1 'b', and the rest 'c'."""
    (row,) = rows
    if kind == "label":
        return {"k0": "a", "k1": "b"}.get(row, "c")
    return row in ("k1", "k2", "k3", "k4")


def test_a_distinct_fetches_one_row_of_each_selected_value():
    with Transcripts({"reviews": KEYED}, _answer) as runs:
        assert_transcript("\n".join(runs.query(sql) for sql in (
            f"SELECT DISTINCT r.stars FROM reviews r WHERE {AI_IF}",
            # a filter on the DISTINCT's columns keeps the per-key fetch
            f"SELECT * FROM (SELECT DISTINCT r.stars FROM reviews r WHERE {AI_IF}) d "
            "WHERE d.stars > 3",
            # one row of each label
            "SELECT DISTINCT AI_CLASSIFY(PROMPT('tone: {0}', r.body), "
            "['a', 'b', 'c']) AS t FROM reviews r")))
