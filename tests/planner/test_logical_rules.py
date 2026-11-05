"""The logical rules eliminate a DISTINCT.

The test suite's queries make should_push_down_projection,
should_push_down_distinct, and should_lift_column_operators rewrite;
these plans make should_eliminate_distinct rewrite as well.
"""


import pyarrow as pa
import pytest
from helpers.sessions import CONFIG, fake_tok

import quail
from quail.planner import native
from quail.planner.context import PlanningContext

REVIEWS = pa.table({"id": ["r0", "r1"], "stars": [5, 2], "body": ["good", "slow"]})
ASPECTS = pa.table({"id": ["a0", "a1"], "aspect": ["ending", "pacing"]})


@pytest.fixture()
def session():
    value = quail.Session(CONFIG, tokenizer=fake_tok)
    value.register("reviews", quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    value.register("aspects", quail.DocumentProvider.from_table(ASPECTS, id_col="id"))
    yield value
    value.close()


def _alternatives(session, plan):
    return native.logical_alternatives(
        plan, session.registry, PlanningContext(catalog=session.catalog))


@pytest.mark.parametrize("text", [
    "SELECT DISTINCT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body))",
    "SELECT DISTINCT r.stars AS k, COUNT(*) AS n FROM reviews r "
    "WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) GROUP BY r.stars",
])
def test_a_distinct_that_removes_no_row_is_eliminated(session, text):
    (alternative,) = _alternatives(session, session.sql(text).logical)
    assert "should_eliminate_distinct" in alternative.rules
    assert not alternative.plan.result.distinct
