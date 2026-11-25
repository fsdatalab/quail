"""The builder composes model calls and explains invalid query arguments."""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish
from helpers.sessions import CONFIG, TOKENIZER

import quail
from quail.common.expressions import CompileError

TABLES = {
    "reviews": pa.table({"id": ["r0", "r1"], "stars": [5, 2],
                         "body": ["good film", "slow film"]}),
    "aspects": pa.table({"id": ["a0", "a1"], "aspect": ["ending", "pacing"]}),
    "contracts": pa.table({"id": ["c0"], "text": ["Lease ends 2027."]}),
}

SENTIMENT = "AI.CLASSIFY(PROMPT('q5: {0}', r.body), ['positive', 'negative'])"
CLASSIFY = f"{SENTIMENT} AS sentiment"

def test_a_label_filter_on_an_unknown_label_is_refused_with_its_reason():
    sql = f"SELECT r.id, {CLASSIFY} FROM reviews r WHERE {SENTIMENT} = 'mixed'"
    reason = r"lists \['mixed'\], which are not among its labels"
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
    for table_name, table in TABLES.items():
        session.register(table_name,
                         quail.DocumentProvider.from_table(table, id_col="id"))
    with pytest.raises(CompileError, match=reason):
        session.sql(sql)
    session.close()


def test_a_score_and_a_generative_predicate_both_filter_rows():
    with quail.Session(CONFIG, tokenizer=TOKENIZER) as session:
        session.register("reviews", quail.DocumentProvider.from_table(
            TABLES["reviews"], id_col="id"))
        for accepted, score in ((True, 0.25), (True, 0.75), (False, 0.25),
                                (False, 0.75)):
            query = session.sql(
                "SELECT r.id AS review_id FROM reviews r "
                "WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) "
                "AND AI.SCORE(PROMPT('q2: {0}', r.body)) > 0.5 ORDER BY r.id")
            oracle = Oracle(filters=lambda *_, accepted=accepted: accepted,
                            scores=lambda *_, score=score: score)
            rows = finish(query, oracle).collect()
            expected = ([{"review_id": "r0"}, {"review_id": "r1"}]
                        if accepted and score > 0.5 else [])
            assert rows.to_pylist() == expected, (accepted, score)

