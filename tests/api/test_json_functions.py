"""JSON functions in SQL: fields read from a text column holding JSON."""

import pyarrow as pa
import pytest
from helpers.sessions import TOKENIZER

import quail

REPLIES = pa.table({
    "id": ["p1", "p2"],
    "reply": [
        '{"title": "Receipts, 1928", "years": [1926, 1927, 1928], "kind": "table"}',
        '{"title": "Outlays", "years": [], "kind": "blank"}',
    ],
})


@pytest.mark.parametrize("dialect", ["snowflake", "bq"])
def test_json_functions_read_fields_of_a_text_column(dialect):
    with quail.Session(quail.EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
                       tokenizer=TOKENIZER) as session:
        session.register("replies", REPLIES, id_col="id")
        rows = session.sql(
            "SELECT r.id, json_get_str(r.reply, 'title') AS title, "
            "json_get_int(r.reply, 'years', 0) AS first_year, "
            "json_length(r.reply, 'years') AS num_years, r.reply ->> 'kind' AS kind "
            "FROM replies r WHERE json_get_str(r.reply, 'kind') = 'table'",
            dialect=dialect).collect().to_pylist()

    assert rows == [{"r.id": "p1", "title": "Receipts, 1928", "first_year": 1926,
                     "num_years": 3, "kind": "table"}]
