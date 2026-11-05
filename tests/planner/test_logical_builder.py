"""The logical plan builder refuses each query it cannot plan, saying why."""

import pyarrow as pa
import pytest
from helpers.sessions import CONFIG, TOKENIZER

import quail
from quail.common.expressions import CompileError

TABLES = {
    "reviews": pa.table({"id": ["r0", "r1"], "stars": [5, 2],
                         "body": ["good film", "slow film"]}),
    "aspects": pa.table({"id": ["a0", "a1"], "aspect": ["ending", "pacing"]}),
    "contracts": pa.table({"id": ["c0"], "text": ["Lease ends 2027."]}),
}

CLASSIFY = ("AI.CLASSIFY(PROMPT('q5: {0}', r.body), "
            "['positive', 'negative']) AS sentiment")
EXTRACT = "AI.EXTRACT(c.text, {'end': 'When does it end?'}) AS lease"

REFUSED = {
    "a score beside a generative filter": (
        "SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) "
        "AND AI.SCORE(PROMPT('q2: {0}', r.body)) > 0.5",
        "AI.SCORE cannot be mixed with generative AI predicates"),
    "DISTINCT over an extraction": (
        f"SELECT DISTINCT {EXTRACT} FROM contracts c",
        "DISTINCT does not compare AI.EXTRACT struct columns"),
    "a label filter on an unknown label": (
        f"SELECT r.id, {CLASSIFY} FROM reviews r WHERE sentiment = 'mixed'",
        r"lists \['mixed'\], which are not among its labels"),
}


@pytest.mark.parametrize("name", REFUSED)
def test_a_query_the_builder_cannot_plan_is_refused_with_its_reason(name):
    sql, reason = REFUSED[name]
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
    for table_name, table in TABLES.items():
        session.register(table_name,
                         quail.DocumentProvider.from_table(table, id_col="id"))
    with pytest.raises(CompileError, match=reason):
        session.sql(sql)
    session.close()


def test_the_arrow_function_list_matches_pyarrow():
    from tools.write_arrow_functions import PATH, arrow_functions_text

    assert PATH.read_text() == arrow_functions_text()
