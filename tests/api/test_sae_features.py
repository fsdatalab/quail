"""Sparse feature results and relational queries over their values."""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish
from helpers.sessions import TOKENIZER

import quail
from quail.api.config import EngineConfig
from quail.backends.quail.executor.operators.sae_features import FEATURE_LIST
from quail.catalog import DocumentProvider
from quail.common.expressions import CompileError

REVIEWS = ["great sound", "meh", "broken zipper"]


def _tokenize(text):
    return [ord(character) for character in text]


@pytest.fixture()
def session():
    value = quail.Session(
        EngineConfig(model="qwen3.5-2b-sae-bf16", device="h100-sxm"),
        tokenizer=TOKENIZER)
    value.register("reviews", DocumentProvider.from_table(
        pa.table({"id": [7, 9, 11], "body": REVIEWS}), id_col="id"))
    yield value
    value.close()


FEATURES = ("AI.SAE_FEATURES(PROMPT('{0}\\n\\nWhat is this review about?', "
            "r.body), {'pooling': 'last'})")


def test_features_run_returns_the_lists_and_the_answer_table(session):
    query = session.sql(f"SELECT r.id, {FEATURES} AS features FROM reviews r")
    oracle = Oracle(features=lambda spec, document: [
        {"feature": document, "strength": float(document + 1) / 2}])
    result = finish(query, oracle)
    rows = result.collect()
    assert rows.column("features").type == FEATURE_LIST
    assert rows.column("features").to_pylist() == [
        [{"feature": index, "strength": (index + 1) / 2}]
        for index in range(3)]
    answers = result.answer_tables["features"]["features"]
    assert answers.num_rows == 3


# one row per feature, as DataFusion's UNNEST in SELECT writes it
ROWS = f"(SELECT r.id, UNNEST({FEATURES}) AS f FROM reviews r) x"


def _unnest_oracle():
    # document d has feature d at strength d + 1, and feature 100 at 0.5
    return Oracle(features=lambda spec, document: [
        {"feature": document, "strength": float(document + 1)},
        {"feature": 100, "strength": 0.5}])


def test_unnest_groups_rows_by_feature(session):
    query = session.sql(
        "SELECT x.f['feature'] AS feature, COUNT(*) AS n, "
        f"MAX(x.f['strength']) AS top FROM {ROWS} "
        "GROUP BY x.f['feature'] ORDER BY n DESC")
    rows = finish(query, _unnest_oracle()).collect()
    assert rows.to_pylist()[0] == {"feature": 100, "n": 3, "top": 0.5}
    assert sorted(row["feature"] for row in rows.to_pylist()) == [0, 1, 2, 100]


def test_not_and_or_of_unnested_fields_and_source_columns_filter_rows(session):
    # feature 100 has strength 0.5 in every document; document d has
    # feature d at strength d + 1
    fields = session.sql(
        f"SELECT x.id, x.f['feature'] AS feature FROM {ROWS} "
        "WHERE NOT (x.f['strength'] < 1 OR x.f['feature'] = 100) ORDER BY x.id")
    assert finish(fields, _unnest_oracle()).collect().to_pylist() == [
        {"x.id": 7, "feature": 0}, {"x.id": 9, "feature": 1},
        {"x.id": 11, "feature": 2}]
    mixed = session.sql(
        f"SELECT x.f['feature'] AS feature FROM {ROWS} "
        "WHERE x.f['strength'] > 2 OR x.id = 7 ORDER BY feature")
    assert finish(mixed, _unnest_oracle()).collect().to_pylist() == [
        {"feature": 0}, {"feature": 2}, {"feature": 100}]


def test_datafusion_unnests_the_feature_lists_the_region_returns(session):
    query = session.sql(f"SELECT r.id, UNNEST({FEATURES}) AS f FROM reviews r")
    assert "UnnestExec" in query.explain()
    rows = finish(query, _unnest_oracle()).collect()
    assert rows.column_names == ["r.id", "f"]
    assert sorted((row["r.id"], row["f"]["feature"]) for row in rows.to_pylist()) == [
        (7, 0), (7, 100), (9, 1), (9, 100), (11, 2), (11, 100)]


@pytest.mark.parametrize("comparison, num_rows", [("IS NULL", 0), ("IS NOT NULL", 3)])
def test_feature_list_null_tests_keep_empty_lists(session, comparison, num_rows):
    features = "AI.SAE_FEATURES(PROMPT('Topic? {0}', r.body), {'pooling': 'last'})"
    query = session.sql(
        f"SELECT r.id, {features} AS features FROM reviews r "
        f"WHERE {features} {comparison}")
    assert finish(query, Oracle(features=lambda spec, document: []))\
        .collect().num_rows == num_rows


@pytest.mark.parametrize("test, reason", [
    ("{} = 'x'", r"AI\.SAE_FEATURES returns a list, so use UNNEST"),
    ("LENGTH({}) > 0", r"LENGTH needs text, so use array_length.*UNNEST"),
])
def test_feature_lists_reject_string_comparisons_and_string_length(
        session, test, reason):
    features = "AI.SAE_FEATURES(PROMPT('Topic? {0}', r.body), {'pooling': 'last'})"
    with pytest.raises(CompileError, match=reason):
        session.sql(
            f"SELECT r.id, {features} AS features FROM reviews r "
            f"WHERE {test.format(features)}")
