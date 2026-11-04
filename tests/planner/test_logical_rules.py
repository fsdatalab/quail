"""The logical rules eliminate a DISTINCT and push a label filter below a join.

The test suite's queries make should_push_down_projection,
should_push_down_distinct, and should_lift_column_operators rewrite;
these plans make the other two rules rewrite as well.
"""

from dataclasses import replace

import pyarrow as pa
import pytest
from helpers.sessions import CONFIG, fake_tok

import quail
from quail.logical import Filter, LogicalPlan, Project, SemanticJoin
from quail.planner import native
from quail.planner.optimizer.context import PlanningContext

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


def test_a_label_filter_above_a_join_moves_below_it(session):
    text = ("SELECT r.id, a.id, AI.CLASSIFY(PROMPT('q5: {0}', r.body), "
            "['good', 'bad']) AS s FROM reviews r JOIN aspects a "
            "ON AI_FILTER(PROMPT('q4: {0} {1}', r.body, a.aspect)) WHERE s = 'good'")
    plan = session.sql(text).logical
    project = plan.root
    join = project.input
    while not isinstance(join, SemanticJoin):
        join = join.input
    # the SemanticJoin's left input is the table's label Filter: move it
    # above the join, as a plan built without the SQL front end might
    joined = join.input
    label = joined.left
    assert isinstance(label, Filter)
    lowered = replace(joined, left=label.input)
    moved = LogicalPlan(replace(project, input=replace(
        label, input=replace(join, input=lowered))))
    assert isinstance(moved.root, Project)
    alternatives = _alternatives(session, moved)
    assert any("should_push_down_filters" in alternative.rules
               for alternative in alternatives)
