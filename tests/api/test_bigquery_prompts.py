"""BigQuery's AI function forms plan the same as Quail's PROMPT forms.

BigQuery writes a prompt as a tuple, a STRUCT, a `||` chain or a
FORMAT call, names it with `prompt =>`, adds service arguments such as
`connection_id`, and casts with FLOAT64. Each must plan exactly as the
PROMPT call it means. Unquoted names keep their letter case, so a
camelCase column matches as written.
"""

import pyarrow as pa
import pytest
from helpers.sessions import CONFIG, TOKENIZER

import quail

pytest.importorskip("_quail_native")

REVIEWS = pa.table({
    "id": ["a", "b"],
    "reviewId": ["1", "2"],
    "reviewText": ["good", "bad"],
    "title": ["x", "y"],
})
POSITIVE = "PROMPT('Positive? {0}', r.reviewText)"
SERVICE = "connection_id => 'us.c', model_params => JSON '{\"labels\": {}}'"


@pytest.fixture
def session():
    with quail.Session(CONFIG, tokenizer=TOKENIZER) as session:
        session.register("reviews", REVIEWS, id_col="reviewId")
        yield session


def plan(session, sql):
    return session.sql(sql, dialect="bq").explain()


def filtered(alias, condition):
    return f"SELECT id FROM reviews {alias} WHERE AI.IF({condition})"


def ratio(condition, kind):
    return (f"SELECT CAST(SUM(CASE WHEN AI.IF({condition}) THEN 1 ELSE 0 END) "
            f"AS {kind}) / COUNT(*) AS ratio FROM reviews AS r")


@pytest.mark.parametrize(("bigquery", "native"), [
    (filtered("r", f"('Positive? ', r.reviewText), {SERVICE}"),
     filtered("r", POSITIVE)),
    (filtered("r", "prompt => ('Positive? ', r.reviewText), endpoint => 'g'"),
     filtered("r", POSITIVE)),
    (filtered("t", "t.title || ' is a comedy: ' || t.reviewText"),
     filtered("t", "PROMPT('{0} is a comedy: {1}', t.title, t.reviewText)")),
    (filtered("t", "STRUCT('Title: ', t.title, ', text: ', t.reviewText)"),
     filtered("t", "PROMPT('Title: {0}, text: {1}', t.title, t.reviewText)")),
    (filtered("t", "FORMAT('''A crash: %s.''', t.reviewText)"),
     filtered("t", "PROMPT('A crash: {0}.', t.reviewText)")),
    (ratio("('Positive? ', r.reviewText)", "FLOAT64"),
     ratio(POSITIVE, "DOUBLE")),
    ("SELECT id, AI.SCORE(('Score: ', reviewText), connection_id => 'c') AS s "
     "FROM reviews",
     "SELECT id, AI.SCORE(PROMPT('Score: {0}', reviewText)) AS s FROM reviews"),
    ("SELECT id, AI.CLASSIFY(('Review: ', reviewText), categories => ['P', 'N'], "
     "connection_id => 'c') AS label FROM reviews",
     "SELECT id, AI.CLASSIFY(PROMPT('Review: {0}', reviewText), ['P', 'N']) "
     "AS label FROM reviews"),
])
def test_a_bigquery_prompt_plans_as_its_prompt_call(session, bigquery, native):
    assert plan(session, bigquery) == plan(session, native)


def test_a_literal_naming_a_placeholder_is_left_unrewritten(session):
    with pytest.raises(Exception, match="needs a PROMPT"):
        plan(session, filtered("t", "('slot {0}: ', t.title)"))
