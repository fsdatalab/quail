"""ORDER BY, LIMIT, OFFSET, and DISTINCT run in DataFusion above the model region."""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish
from helpers.sessions import TOKENIZER

import quail
from quail.api.config import EngineConfig
from quail.catalog import DocumentProvider
from quail.physical import Limit

# the fourth review has no body, so it has no score
BODIES = ["refund now", "fine", "refund please", None, "refund", "ok"]
# the score of each document, by its position; the executor reads a
# score as a float32, which holds each of these exactly
SCORES = [0.875, 0.125, 0.625, None, 0.75, 0.25]


@pytest.fixture()
def session():
    value = quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
                          tokenizer=TOKENIZER)
    value.register("reviews", DocumentProvider.from_table(
        pa.table({"id": [10, 11, 12, 13, 14, 15], "body": BODIES,
                  "stars": [1, 2, 1, 2, 1, 2]}), id_col="id"))
    yield value
    value.close()


def _scores():
    return Oracle(scores=lambda spec, ids: SCORES[ids[0]])


SCORE = "AI.SCORE(PROMPT('Asks for a refund: {0}', r.body))"


def test_order_by_limit_and_offset_over_an_ai_score_return_the_sorted_rows(session):
    query = session.sql(f"SELECT r.id, {SCORE} AS score FROM reviews r "
                        "ORDER BY score DESC NULLS LAST LIMIT 2")
    rows = finish(query, _scores()).collect().to_pylist()
    assert rows == [{"r.id": 10, "score": 0.875}, {"r.id": 14, "score": 0.75}]
    # the region scores every review; DataFusion keeps the top two
    assert not any(isinstance(node, Limit) for node in query.plan().nodes)
    assert "filter_limit" not in query.plan().settings
    assert "SortExec: TopK(fetch=2)" in query.explain()
    # an offset skips the sorted rows, with nulls highest
    query = session.sql(f"SELECT r.id FROM reviews r ORDER BY {SCORE} DESC "
                        "LIMIT 2 OFFSET 3")
    rows = finish(query, _scores()).collect().to_pylist()
    assert rows == [{"r.id": 12}, {"r.id": 15}]
    last = session.sql(f"SELECT r.id FROM reviews r ORDER BY {SCORE} LIMIT 1 OFFSET 5")
    assert finish(last, _scores()).collect().to_pylist() == [{"r.id": 13}]


def test_a_limit_stops_an_ai_filter_and_a_distinct_returns_each_value_once(session):
    query = session.sql("SELECT r.id FROM reviews r "
                        "WHERE AI.IF(PROMPT('Asks for a refund: {0}', r.body)) LIMIT 2")
    assert query.plan().settings["filter_limit"] == 2
    rows = finish(query, Oracle(filters=lambda alias, document, position:
                                "refund" in (BODIES[document] or ""))).collect()
    assert rows.column("r.id").to_pylist() == [10, 12]
    # DISTINCT returns each value of the kept rows once
    query = session.sql("SELECT DISTINCT r.stars FROM reviews r "
                        "WHERE AI.IF(PROMPT('Asks for a refund: {0}', r.body)) "
                        "ORDER BY r.stars")
    rows = finish(query, Oracle(filters=lambda alias, document, position:
                                "refund" in (BODIES[document] or "") or document == 5))
    assert rows.collect().to_pylist() == [{"r.stars": 1}, {"r.stars": 2}]
    assert "AggregateExec" in query.explain()
