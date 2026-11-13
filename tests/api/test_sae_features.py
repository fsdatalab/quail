"""Sparse feature results and relational queries over their values.

The transcript tests' expected text is in transcripts/: the plan's AI
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
        columns=(listed,)))


def test_an_empty_feature_list_is_not_null():
    head = f"SELECT r.id, {TOPIC} AS features FROM reviews r WHERE "
    assert_transcript(_transcript(
        f"{head}{TOPIC} IS NULL", f"{head}{TOPIC} IS NOT NULL"))


def test_a_text_test_of_a_feature_list_is_refused(session):
    head = f"SELECT r.id, {TOPIC} AS features FROM reviews r WHERE "
    for test, reason in (
            ("{} = 'x'", r"AI\.SAE_FEATURES returns a list, so use UNNEST"),
            ("LENGTH({}) > 0", r"LENGTH needs text, so use array_length.*UNNEST")):
        with pytest.raises(CompileError, match=reason):
            session.sql(head + test.format(TOPIC))
