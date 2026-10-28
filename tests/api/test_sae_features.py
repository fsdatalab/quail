"""Sparse feature results and relational queries over their values."""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish

import quail
from quail.backends.quail.executor.operators.sae_features import FEATURE_LIST
from quail.catalog import DocumentProvider
from quail.logical import CompileError
from quail.physical import decode_graph, encode_graph
from quail.planner.plan import EngineConfig

QUESTION = "{0}\n\nWhat is this review about?"
REVIEWS = ["great sound", "meh", "broken zipper"]


def _tokenize(text):
    return [ord(character) for character in text]


@pytest.fixture()
def session():
    value = quail.Session(
        EngineConfig(model="qwen3.5-2b-sae-bf16", device="h100-sxm"),
        tokenizer=_tokenize)
    value.register("reviews", DocumentProvider.from_table(
        pa.table({"id": [7, 9, 11], "body": REVIEWS}), id_col="id"))
    yield value
    value.close()


def _features_query(session, pooling="last"):
    return (session.docs("reviews").alias("r")
            .ai_sae_features(quail.prompt(QUESTION, quail.col("r.body")),
                             pooling=pooling, name="features")
            .select("r.id", "features"))


def test_features_run_returns_the_lists_and_the_answer_table(session):
    query = _features_query(session)
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


UNNEST = ("UNNEST(AI.SAE_FEATURES(PROMPT('{0}\\n\\nWhat is this review about?', "
          "r.body), pooling => 'last')) AS f")


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


def _two_lists_oracle():
    # document d's list f has features d and 100; its list g has feature d
    return Oracle(features=lambda spec, document: [
        {"feature": document, "strength": float(document + 1)},
        {"feature": 100, "strength": 0.5}][:2 if spec.name == "f" else 1])


TWO_CALLS = ("AI.SAE_FEATURES(PROMPT('{0}\\n\\nWhat is this review about?', "
             "r.body), pooling => 'last'), AI.SAE_FEATURES(PROMPT('{0}\\n\\n"
             "Who wrote it?', r.body), pooling => 'max')")


def test_two_unnests_return_the_product_and_filter_above_their_own_list(session):
    sql = session.sql(
        f"SELECT r.id, f.feature, g.feature FROM reviews r, "
        f"{UNNEST}, {UNNEST.replace('this review about', 'its author')[:-4]} AS g "
        f"WHERE f.strength > 1 ORDER BY r.id, f.feature, g.feature")
    built = (session.docs("reviews").alias("r")
             .ai_sae_features(quail.prompt(QUESTION, quail.col("r.body")),
                              pooling="last", name="f")
             .ai_sae_features(quail.prompt("{0}\n\nWhat is its author about?",
                                           quail.col("r.body")),
                              pooling="last", name="g")
             .unnest("f").unnest("g").where(quail.col("f.strength") > 1)
             .order_by("r.id", "f.feature", "g.feature")
             .select("r.id", "f.feature", "g.feature"))
    for query in (sql, built):
        rows = finish(query, _unnest_oracle()).collect()
        assert rows.to_pylist() == [
            {"r.id": 9, "f.feature": 1, "g.feature": 1},
            {"r.id": 9, "f.feature": 1, "g.feature": 100},
            {"r.id": 11, "f.feature": 2, "g.feature": 2},
            {"r.id": 11, "f.feature": 2, "g.feature": 100}]


def test_a_cte_returns_two_zipped_lists_under_their_select_names(session):
    query = session.sql(
        f"WITH pairs AS (SELECT r.id, f.feature AS first_feature, "
        f"g.feature AS second_feature FROM reviews r, "
        f"UNNEST({TWO_CALLS}) AS u(f, g)) "
        f"SELECT id, first_feature, second_feature FROM pairs "
        f"WHERE id = 9 ORDER BY first_feature")
    rows = finish(query, _two_lists_oracle()).collect()
    assert [tuple(row.values()) for row in rows.to_pylist()] == [
        (9, 1, 1), (9, 100, None)]


def test_left_join_unnest_keeps_a_row_with_no_features_as_nulls(session):
    # document 0 has no features; document d > 0 has feature d
    oracle = Oracle(features=lambda spec, document: [
        {"feature": document, "strength": 1.0}] if document else [])
    sql = session.sql(
        f"SELECT r.id, f.feature FROM reviews r LEFT JOIN {UNNEST} ON TRUE "
        f"ORDER BY r.id")
    built = (session.docs("reviews").alias("r")
             .ai_sae_features(quail.prompt(QUESTION, quail.col("r.body")),
                              pooling="last", name="f")
             .unnest("f", outer=True).order_by("r.id")
             .select("r.id", "f.feature"))
    for query in (sql, built):
        assert finish(query, oracle).collect().to_pylist() == [
            {"r.id": 7, "f.feature": None}, {"r.id": 9, "f.feature": 1},
            {"r.id": 11, "f.feature": 2}]
    inner = session.sql(f"SELECT r.id, f.feature FROM reviews r, {UNNEST}")
    assert sorted(finish(inner, oracle).collect().column("r.id").to_pylist()) == [
        9, 11]


def test_unnested_fields_resolve_without_qualifiers_when_unambiguous(session):
    sql = session.sql(
        f"SELECT feature AS feature_id FROM reviews r, {UNNEST} "
        f"WHERE strength > 1 ORDER BY strength DESC")
    built = (session.docs("reviews").alias("r")
             .ai_sae_features(quail.prompt(QUESTION, quail.col("r.body")),
                              pooling="last", name="f")
             .unnest("f").where(quail.col("strength") > 1)
             .order_by(quail.col("strength").desc())
             .select(quail.col("feature").named("feature_id")))
    for query in (sql, built):
        result = finish(query, _unnest_oracle()).collect()
        assert result.to_pylist() == [{"feature_id": 2}, {"feature_id": 1}]
    with pytest.raises(CompileError, match="ambiguous"):
        session.sql(f"SELECT feature FROM reviews r, "
                    f"UNNEST({TWO_CALLS}) AS u(f, g)")


def test_count_star_returns_zero_when_no_features_pass_the_filter(session):
    query = session.sql(
        f"SELECT COUNT(*) AS n FROM reviews r, {UNNEST} "
        f"WHERE f.strength > 3")
    assert finish(query, _unnest_oracle()).collect().to_pylist() == [{"n": 0}]


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
