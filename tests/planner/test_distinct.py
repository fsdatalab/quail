"""DataFusion's DISTINCT rules in a model region.

EliminateDistinct drops a DISTINCT that removes no row; PushDownDistinct
gives a one-table SemanticFilter the selected columns as its stop key.
"""


import pyarrow as pa
import pytest
from helpers.sessions import CONFIG, TOKENIZER

import quail
from quail.logical import SemanticFilter

REVIEWS = pa.table({"id": ["r0", "r1"], "stars": [5, 2], "body": ["good", "slow"]})


@pytest.fixture()
def session():
    value = quail.Session(CONFIG, tokenizer=TOKENIZER)
    value.register("reviews", quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    yield value
    value.close()


@pytest.mark.parametrize("text", [
    "SELECT DISTINCT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body))",
    "SELECT DISTINCT r.stars AS k, COUNT(*) AS n FROM reviews r "
    "WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) GROUP BY r.stars",
])
def test_a_distinct_that_removes_no_row_is_eliminated(session, text):
    assert not session.sql(text).logical.result.distinct


def test_a_distinct_on_one_tables_columns_stops_its_filter_at_the_first_match(session):
    plan = session.sql("SELECT DISTINCT r.stars FROM reviews r "
                       "WHERE AI_FILTER(PROMPT('q1: {0}', r.body))").logical
    assert plan.result.distinct
    (found,) = [node for node in plan.walk() if isinstance(node, SemanticFilter)]
    assert found.stop_key == ("stars",)
