"""Sparse feature results and relational queries over their values."""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish
from helpers.sessions import TOKENIZER

import quail
from quail.api.config import EngineConfig
from quail.backends.quail.executor.operators.sae_features import FEATURE_LIST
from quail.catalog import DocumentProvider
from quail.logical import CompileError
from quail.physical import decode_graph, encode_graph

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
            "r.body), pooling => 'last')")


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


UNNEST = f"UNNEST({FEATURES}) AS f"


def _unnest_oracle():
    # document d has feature d at strength d + 1, and feature 100 at 0.5
    return Oracle(features=lambda spec, document: [
        {"feature": document, "strength": float(document + 1)},
        {"feature": 100, "strength": 0.5}])


def test_unnest_groups_rows_by_feature(session):
    query = session.sql(
        f"SELECT f.feature, COUNT(*) AS n, MAX(f.strength) AS top "
        f"FROM reviews r, {UNNEST} GROUP BY f.feature ORDER BY n DESC")
    codecs = session.registry.codecs
    plan = query.plan()
    assert decode_graph(encode_graph(plan.graph, codecs), codecs) == plan.graph
    rows = finish(query, _unnest_oracle()).collect()
    assert rows.to_pylist()[0] == {"f.feature": 100, "n": 3, "top": 0.5}
    assert sorted(row["f.feature"] for row in rows.to_pylist()) == [0, 1, 2, 100]


def test_not_and_or_of_unnested_fields_and_source_columns_filter_rows(session):
    # feature 100 has strength 0.5 in every document; document d has
    # feature d at strength d + 1
    fields = session.sql(
        f"SELECT r.id, f.feature FROM reviews r, {UNNEST} "
        f"WHERE NOT (f.strength < 1 OR f.feature = 100) ORDER BY r.id")
    assert finish(fields, _unnest_oracle()).collect().to_pylist() == [
        {"r.id": 7, "f.feature": 0}, {"r.id": 9, "f.feature": 1},
        {"r.id": 11, "f.feature": 2}]
    mixed = session.sql(
        f"SELECT f.feature FROM reviews r, {UNNEST} "
        f"WHERE f.strength > 2 OR r.id = 7 ORDER BY f.feature")
    assert finish(mixed, _unnest_oracle()).collect().to_pylist() == [
        {"f.feature": 0}, {"f.feature": 2}, {"f.feature": 100}]


def test_explain_shows_the_columns_an_unnest_returns(session):
    query = session.sql(f"SELECT r.id, f.feature FROM reviews r, {UNNEST}")
    assert "Project: r.id, f.feature" in query.explain()
@pytest.mark.parametrize("comparison, num_rows", [("IS NULL", 0), ("IS NOT NULL", 3)])
def test_feature_list_null_tests_keep_empty_lists(session, comparison, num_rows):
    query = session.sql(
        "SELECT r.id, AI.SAE_FEATURES(PROMPT('Topic? {0}', r.body), "
        "pooling => 'last') AS features FROM reviews r "
        f"WHERE features {comparison}")
    assert finish(query, Oracle(features=lambda spec, document: []))\
        .collect().num_rows == num_rows


@pytest.mark.parametrize("test", ["features = 'x'", "LENGTH(features) > 0"])
def test_feature_lists_must_be_unnested_before_testing_their_fields(session, test):
    with pytest.raises(CompileError, match="UNNEST"):
        session.sql(
            "SELECT r.id, AI.SAE_FEATURES(PROMPT('Topic? {0}', r.body), "
            "pooling => 'last') AS features FROM reviews r "
            f"WHERE {test}")


