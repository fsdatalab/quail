"""Sparse feature results and relational queries over their values.

Each transcript test holds its expected transcript: the plan's AI
operators, the questions the model answered, and the rows
(helpers/transcript.py).
"""

import pyarrow as pa
import pytest
from helpers.sessions import TOKENIZER
from helpers.transcript import Transcripts, assert_transcript

import quail
from quail.api.config import EngineConfig
from quail.catalog import DocumentProvider
from quail.common.expressions import CompileError

CONFIG = EngineConfig(model="qwen3.5-2b-sae-bf16", device="h100-sxm")
REVIEWS = pa.table({"id": ["r0", "r1", "r2"],
                    "body": ["great sound", "meh", "broken zipper"]})
FEATURES = "AI.SAE_FEATURES(PROMPT('About? {0}', r.body), {'pooling': 'last'})"
TOPIC = "AI.SAE_FEATURES(PROMPT('Topic? {0}', r.body), {'pooling': 'last'})"
# one row per feature, as DataFusion's UNNEST in SELECT writes it
ROWS = f"(SELECT r.id, UNNEST({FEATURES}) AS f FROM reviews r) x"


@pytest.fixture()
def session():
    value = quail.Session(CONFIG, tokenizer=TOKENIZER)
    value.register("reviews", DocumentProvider.from_table(REVIEWS, id_col="id"))
    yield value
    value.close()


def _answer(kind, prompt, rows):
    """The model's features.

    r<d> has feature d at strength d + 1, and feature 100 at 0.5; asked
    the topic, every review has none.
    """
    if prompt == "Topic? {0}":
        return []
    index = int(rows[0][1:])
    return [{"feature": index, "strength": float(index + 1)},
            {"feature": 100, "strength": 0.5}]


def _transcript(*queries, columns=()) -> str:
    """The transcripts of `queries`; those in `columns` show column types."""
    with Transcripts({"reviews": REVIEWS}, _answer, config=CONFIG) as runs:
        return "\n".join(runs.query(sql, columns=sql in columns)
                         for sql in queries)


def test_features_return_a_list_per_document_that_datafusion_unnests_and_filters():
    listed = f"SELECT r.id, {FEATURES} AS features FROM reviews r"
    assert_transcript(_transcript(
        listed,
        f"SELECT r.id, UNNEST({FEATURES}) AS f FROM reviews r",
        "SELECT x.f['feature'] AS feature, COUNT(*) AS n, "
        f"MAX(x.f['strength']) AS top FROM {ROWS} GROUP BY x.f['feature']",
        f"SELECT x.id, x.f['feature'] AS feature FROM {ROWS} "
        "WHERE NOT (x.f['strength'] < 1 OR x.f['feature'] = 100)",
        f"SELECT x.f['feature'] AS feature FROM {ROWS} "
        "WHERE x.f['strength'] > 2 OR x.id = 'r0'",
        columns=(listed,)), """\
--- SELECT r.id, AI.SAE_FEATURES(PROMPT('About? {0}', r.body), {'pooling': 'last'}) AS
    features FROM reviews r ---
plan
  features = ai_sae_features('About? {0}', r.body, named_struct('pooling', 'last'))
asked 3 of 3
  'About? {0}': r0 [{'feature': 0, 'strength': 1.0}, {'feature': 100, 'strength': 0.5}],
      r1 [{'feature': 1, 'strength': 2.0}, {'feature': 100, 'strength': 0.5}], r2
      [{'feature': 2, 'strength': 3.0}, {'feature': 100, 'strength': 0.5}]
columns r.id string, features list<item: struct<feature: int32, strength: float>>
rows 3
  r0, [{'feature': 0, 'strength': 1.0}, {'feature': 100, 'strength': 0.5}]
  r1, [{'feature': 1, 'strength': 2.0}, {'feature': 100, 'strength': 0.5}]
  r2, [{'feature': 2, 'strength': 3.0}, {'feature': 100, 'strength': 0.5}]

--- SELECT r.id, UNNEST(AI.SAE_FEATURES(PROMPT('About? {0}', r.body), {'pooling':
    'last'})) AS f FROM reviews r ---
plan
  __unnest_placeholder(AI.SAE_FEATURES('About? {0}', r.body,named_struct('pooling',
      'last'))) = ai_sae_features('About? {0}', r.body, named_struct('pooling', 'last'))
asked 3 of 3
  'About? {0}': r0 [{'feature': 0, 'strength': 1.0}, {'feature': 100, 'strength': 0.5}],
      r1 [{'feature': 1, 'strength': 2.0}, {'feature': 100, 'strength': 0.5}], r2
      [{'feature': 2, 'strength': 3.0}, {'feature': 100, 'strength': 0.5}]
rows 6
  r0, {'feature': 0, 'strength': 1.0}
  r0, {'feature': 100, 'strength': 0.5}
  r1, {'feature': 1, 'strength': 2.0}
  r1, {'feature': 100, 'strength': 0.5}
  r2, {'feature': 100, 'strength': 0.5}
  r2, {'feature': 2, 'strength': 3.0}

--- SELECT x.f['feature'] AS feature, COUNT(*) AS n, MAX(x.f['strength']) AS top FROM
    (SELECT r.id, UNNEST(AI.SAE_FEATURES(PROMPT('About? {0}', r.body), {'pooling':
    'last'})) AS f FROM reviews r) x GROUP BY x.f['feature'] ---
plan
  __unnest_placeholder(AI.SAE_FEATURES('About? {0}', r.body,named_struct('pooling',
      'last'))) = ai_sae_features('About? {0}', r.body, named_struct('pooling', 'last'))
asked 3 of 3
  'About? {0}': r0 [{'feature': 0, 'strength': 1.0}, {'feature': 100, 'strength': 0.5}],
      r1 [{'feature': 1, 'strength': 2.0}, {'feature': 100, 'strength': 0.5}], r2
      [{'feature': 2, 'strength': 3.0}, {'feature': 100, 'strength': 0.5}]
rows 4
  0, 1, 1.0
  1, 1, 2.0
  100, 3, 0.5
  2, 1, 3.0

--- SELECT x.id, x.f['feature'] AS feature FROM (SELECT r.id,
    UNNEST(AI.SAE_FEATURES(PROMPT('About? {0}', r.body), {'pooling': 'last'})) AS f FROM
    reviews r) x WHERE NOT (x.f['strength'] < 1 OR x.f['feature'] = 100) ---
plan
  __unnest_placeholder(AI.SAE_FEATURES('About? {0}', r.body,named_struct('pooling',
      'last'))) = ai_sae_features('About? {0}', r.body, named_struct('pooling', 'last'))
asked 3 of 3
  'About? {0}': r0 [{'feature': 0, 'strength': 1.0}, {'feature': 100, 'strength': 0.5}],
      r1 [{'feature': 1, 'strength': 2.0}, {'feature': 100, 'strength': 0.5}], r2
      [{'feature': 2, 'strength': 3.0}, {'feature': 100, 'strength': 0.5}]
rows 3
  r0, 0
  r1, 1
  r2, 2

--- SELECT x.f['feature'] AS feature FROM (SELECT r.id,
    UNNEST(AI.SAE_FEATURES(PROMPT('About? {0}', r.body), {'pooling': 'last'})) AS f FROM
    reviews r) x WHERE x.f['strength'] > 2 OR x.id = 'r0' ---
plan
  __unnest_placeholder(AI.SAE_FEATURES('About? {0}', r.body,named_struct('pooling',
      'last'))) = ai_sae_features('About? {0}', r.body, named_struct('pooling', 'last'))
asked 3 of 3
  'About? {0}': r0 [{'feature': 0, 'strength': 1.0}, {'feature': 100, 'strength': 0.5}],
      r1 [{'feature': 1, 'strength': 2.0}, {'feature': 100, 'strength': 0.5}], r2
      [{'feature': 2, 'strength': 3.0}, {'feature': 100, 'strength': 0.5}]
rows 3
  0
  100
  2
""")


def test_an_empty_feature_list_is_not_null():
    head = f"SELECT r.id, {TOPIC} AS features FROM reviews r WHERE "
    assert_transcript(_transcript(
        f"{head}{TOPIC} IS NULL", f"{head}{TOPIC} IS NOT NULL"), """\
--- SELECT r.id, AI.SAE_FEATURES(PROMPT('Topic? {0}', r.body), {'pooling': 'last'}) AS
    features FROM reviews r WHERE AI.SAE_FEATURES(PROMPT('Topic? {0}', r.body),
    {'pooling': 'last'}) IS NULL ---
plan
  features = ai_sae_features('Topic? {0}', r.body, named_struct('pooling', 'last')),
      filter: features IS NULL
asked 3 of 3
  'Topic? {0}': r0 [], r1 [], r2 []
rows 0

--- SELECT r.id, AI.SAE_FEATURES(PROMPT('Topic? {0}', r.body), {'pooling': 'last'}) AS
    features FROM reviews r WHERE AI.SAE_FEATURES(PROMPT('Topic? {0}', r.body),
    {'pooling': 'last'}) IS NOT NULL ---
plan
  features = ai_sae_features('Topic? {0}', r.body, named_struct('pooling', 'last')),
      filter: features IS NOT NULL
asked 3 of 3
  'Topic? {0}': r0 [], r1 [], r2 []
rows 3
  r0, []
  r1, []
  r2, []
""")


def test_a_text_test_of_a_feature_list_is_refused(session):
    head = f"SELECT r.id, {TOPIC} AS features FROM reviews r WHERE "
    for test, reason in (
            ("{} = 'x'", r"AI\.SAE_FEATURES returns a list, so use UNNEST"),
            ("LENGTH({}) > 0", r"LENGTH needs text, so use array_length.*UNNEST")):
        with pytest.raises(CompileError, match=reason):
            session.sql(head + test.format(TOPIC))
