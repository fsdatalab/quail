"""The Rust logical plan builder refuses what the Python one refuses, word for word.

tests/conftest.py builds every SELECT the front end binds in Rust too.
These queries reach the builder and are refused there, so each test
checks that the Rust builder gave the same message.
"""

import pyarrow as pa
import pytest
from helpers import rust_builder
from helpers.sessions import CONFIG, fake_tok

import quail
from quail.common.expressions import CompileError

native = pytest.importorskip("_quail_native")

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
    "a score beside a generative filter":
        "SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) "
        "AND AI.SCORE(PROMPT('q2: {0}', r.body)) > 0.5",
    "an ON equality with no AI predicate":
        "SELECT r.id FROM reviews r JOIN aspects a ON r.id = a.id "
        "WHERE AI_FILTER(PROMPT('q1: {0}', r.body))",
    "a joined table no predicate reads":
        "SELECT r.id FROM reviews r CROSS JOIN aspects a "
        "WHERE AI_FILTER(PROMPT('q1: {0}', r.body))",
    "DISTINCT over an extraction":
        f"SELECT DISTINCT {EXTRACT} FROM contracts c",
    "a label filter on an unknown label":
        f"SELECT r.id, {CLASSIFY} FROM reviews r WHERE sentiment = 'mixed'",
}


@pytest.mark.parametrize("name", REFUSED)
def test_the_rust_builder_refuses_with_the_python_message(name):
    session = quail.Session(CONFIG, tokenizer=fake_tok)
    for table_name, table in TABLES.items():
        session.register(table_name,
                         quail.DocumentProvider.from_table(table, id_col="id"))
    compared = len(rust_builder.COMPARED_REFUSALS)
    with pytest.raises(CompileError) as refused:
        session.sql(REFUSED[name])
    session.close()
    assert rust_builder.COMPARED_REFUSALS[compared:] == [str(refused.value)]


def test_the_arrow_function_list_matches_pyarrow():
    from tools.write_arrow_functions import PATH, arrow_functions_text

    assert PATH.read_text() == arrow_functions_text()
