"""A query's result columns take the names its SQL gives them.

A column the SQL selects without a name keeps its table, as `r.id`. A
column the SQL names with `AS` takes that name, even when it is the
column's own name, which DataFusion plans as the bare column. The
BigQuery dialect keeps the name's letter case, as BigQuery returns it.
"""

import pyarrow as pa
import pytest
from helpers.sessions import CONFIG, TOKENIZER

import quail

pytest.importorskip("_quail_native")

REVIEWS = pa.table({"id": ["a", "b"], "reviewId": ["1", "2"], "vin": ["x", "y"]})


@pytest.fixture
def session():
    with quail.Session(CONFIG, tokenizer=TOKENIZER) as session:
        session.register("reviews", REVIEWS, id_col="reviewId")
        yield session


@pytest.mark.parametrize(("dialect", "sql", "names"), [
    ("bq", "SELECT r.vin AS vin FROM reviews AS r", ["vin"]),
    ("bq", "SELECT r.id, r.`reviewId` AS reviewId1 FROM reviews r",
     ["r.id", "reviewId1"]),
    ("bq", "SELECT r1.vin FROM reviews r1 JOIN reviews r2 ON r1.id = r2.id",
     ["r1.vin"]),
    ("snowflake", "SELECT r.vin AS vin FROM reviews AS r", ["vin"]),
    ("snowflake", 'SELECT r.id, r."reviewId" AS reviewId1 FROM reviews r',
     ["r.id", "reviewid1"]),
])
def test_a_column_named_with_as_takes_that_name(session, dialect, sql, names):
    assert session.sql(sql, dialect=dialect).run().collect().schema.names == names
