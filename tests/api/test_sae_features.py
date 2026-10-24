"""AI.SAE_FEATURES: prompt text, planning, refusals, execution, and results."""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish

import quail
from quail.backends.quail.executor.operators.sae_features import FEATURE_LIST
from quail.backends.quail.executor.sae import feature_lists_of_rows
from quail.catalog import DocumentProvider
from quail.cost.sae_features import SaeFeaturesCost
from quail.logical import CompileError
from quail.logical.prompts import bind_sae_features_prompt
from quail.physical import AiSaeFeatures, decode_graph, encode_graph
from quail.planner.plan import EngineConfig, Refusal
from quail.specs import MODELS, QWEN3_5_2B_SAE_BF16

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


def test_the_prompt_is_the_document_then_the_question_with_no_cue():
    reference = type("Ref", (), {"alias": "r", "column": "body"})()
    bound = bind_sae_features_prompt(QUESTION, (reference,))
    assert bound.preamble == "DOCUMENT:\n"
    assert bound.tail == "{0}\n\nWhat is this review about?"
    assert bound.kind == "features"
    with pytest.raises(CompileError, match="one document column"):
        bind_sae_features_prompt("{0} and {1}", (reference, reference))


def test_the_spec_loads_the_sae_and_stops_at_layer_11():
    spec = QWEN3_5_2B_SAE_BF16
    assert spec.layers == 12 and spec.checkpoint_layers == 24
    # layers 0 to 11 hold 9 Gated DeltaNet layers and 3 attention layers
    assert len(spec.gdn_layers) == 9 and len(spec.attention_layers) == 3
    assert spec.sae.num_features == 32_768 and spec.sae.keep == 100
    assert spec.sae.file == "layer11.sae.pt"
    # half the checkpoint's attention layers, so half the KV per token
    full = MODELS["qwen3.5-4b-bf16"]
    assert len(spec.attention_layers) == len(full.attention_layers) * 12 // 32


def test_features_plan_names_the_pooling_and_round_trips(session):
    query = _features_query(session, pooling="max")
    plan = query.plan()
    (node,) = [n for n in plan.nodes if isinstance(n, AiSaeFeatures)]
    assert node.spec.pooling == "max"
    assert node.spec.name == "features"
    assert node.rows_port == "rows:r"
    assert node.spec.estimated_seconds > 0
    codecs = session.registry.codecs
    assert decode_graph(encode_graph(plan.graph, codecs), codecs) == plan.graph


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


def test_three_feature_calls_share_the_documents(session):
    query = (session.docs("reviews").alias("r")
             .ai_sae_features(quail.prompt(QUESTION, quail.col("r.body")),
                              pooling="max", name="first_features")
             .ai_sae_features(quail.prompt("{0}\n\nWhat tone?",
                                           quail.col("r.body")),
                              pooling="max", name="second_features")
             .select("r.id", "first_features", "second_features"))
    plan = query.plan()
    nodes = [n for n in plan.nodes if isinstance(n, AiSaeFeatures)]
    assert [n.spec.name for n in nodes] == ["first_features",
                                            "second_features"]
    # the second call follows the first in one pipeline over resident KV
    from quail.physical.pipelines import build_pipelines

    pipelines = build_pipelines(plan.graph)
    assert pipelines[nodes[0].node_id] == pipelines[nodes[1].node_id]
    rows = finish(query, Oracle()).collect()
    assert rows.num_rows == 3
    assert rows.column("first_features").to_pylist() == rows.column(
        "second_features").to_pylist()


def test_sql_binds_the_pooling_and_rejects_a_bad_one(session):
    query = session.sql(
        "SELECT r.id, AI.SAE_FEATURES(PROMPT('{0}\\n\\nWhat is this review "
        "about?', r.body), pooling => 'max') AS features FROM reviews r")
    (node,) = [n for n in query.plan().nodes
               if isinstance(n, AiSaeFeatures)]
    assert node.spec.pooling == "max"
    with pytest.raises(CompileError, match="one of"):
        session.sql(
            "SELECT r.id, AI.SAE_FEATURES(PROMPT('{0} topic', r.body), "
            "pooling => 'mean') AS features FROM reviews r")
    with pytest.raises(CompileError, match="pooling"):
        session.sql(
            "SELECT r.id, AI.SAE_FEATURES(PROMPT('{0} topic', r.body)) "
            "AS features FROM reviews r")


def test_a_feature_list_cannot_be_sorted_grouped_or_deduplicated(session):
    call = ("AI.SAE_FEATURES(PROMPT('{0}\\n\\nWhat is this about?', r.body), "
            "pooling => 'last')")
    for sql in (f"SELECT r.id, {call} AS features FROM reviews r "
                f"ORDER BY features",
                f"SELECT {call} AS features, COUNT(*) AS n FROM reviews r "
                f"GROUP BY features",
                f"SELECT DISTINCT {call} AS features FROM reviews r"):
        with pytest.raises(CompileError, match="AI.SAE_FEATURES list"):
            session.sql(sql)


def test_features_and_a_score_in_one_query_are_refused(session):
    query = (session.docs("reviews").alias("r")
             .ai_sae_features(quail.prompt("{0}\n\nWhat is this about?",
                                           quail.col("r.body")),
                              pooling="last", name="features")
             .ai_score(quail.prompt("Is this positive?\n\n{0}",
                                    quail.col("r.body")), name="score")
             .select("r.id", "features", "score"))
    refusal = query.plan()
    assert isinstance(refusal, Refusal)
    assert "does not mix with other AI operators" in refusal.reasons[0]


def test_a_model_without_an_sae_refuses_features_and_the_reverse(session):
    with quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
                       tokenizer=_tokenize) as full:
        full.register("reviews", DocumentProvider.from_table(
            pa.table({"id": [1], "body": ["text"]}), id_col="id"))
        refusal = _features_query(full).plan()
        assert isinstance(refusal, Refusal)
        assert "needs a model loaded with an SAE" in refusal.reasons[0]
    # the SAE model stops at the SAE's layer, so it serves no AI.IF
    query = (session.docs("reviews").alias("r")
             .ai_filter(quail.prompt("{0}\nIs it positive?",
                                     quail.col("r.body")))
             .select("r.id"))
    refusal = query.plan()
    assert isinstance(refusal, Refusal)
    assert "AI.SAE_FEATURES only" in refusal.reasons[0]


def test_speed_of_light_prices_the_encode(session):
    estimate = quail.speed_of_light_estimate(
        _features_query(session, pooling="max"),
        answer=lambda prompt, assignment: True)
    encode = estimate.latency.component("sae_encode")
    assert encode.seconds > 0
    (stage,) = estimate.filter_stages
    assert stage["operator"] == "AI.SAE_FEATURES"
    # 'last' encodes one token per row, so its encode work is smaller
    last = quail.speed_of_light_estimate(
        _features_query(session, pooling="last"),
        answer=lambda prompt, assignment: True)
    assert (last.work.sae_encoded < estimate.work.sae_encoded)


def test_the_cost_model_charges_resident_calls_the_question_only():
    from types import SimpleNamespace

    from quail.planner.statistics.tables import summarize_alias
    from quail.specs import H100_SXM

    summary = summarize_alias(
        SimpleNamespace(arrow_array=pa.array([400, 600])), 0, memo={})
    cost = SaeFeaturesCost(
        token_lengths=(10, 20), summary=summary, pooling="max",
        model=QWEN3_5_2B_SAE_BF16, device=H100_SXM, chunk_tokens=8192,
        gpus=1)
    fresh_work, fresh_seconds = cost.estimate(2.0)
    resident_work, resident_seconds = cost.estimate(2.0, resident=True)
    # a resident call feeds and encodes only its question tokens; the
    # documents' maxima are kept from the call before it
    assert resident_work.tokens < fresh_work.tokens
    assert resident_work.sae_encoded == 2 * 30
    assert fresh_work.sae_encoded == 2 * 30 + 400 + 600
    assert resident_seconds < fresh_seconds


def test_packed_rows_unpack_without_zero_strengths():
    import numpy as np

    rows = np.array([[3, 5, 0, 1.5, 0.5, 0.0],
                     [7, 0, 0, 0.0, 0.0, 0.0],
                     [1, 2, 4, 0.25, 2.0, 1.0]], dtype=np.float32)
    lengths, indices, strengths = feature_lists_of_rows(rows, 3)
    assert lengths.tolist() == [2, 0, 3]
    assert indices.tolist() == [3, 5, 1, 2, 4]
    assert strengths.tolist() == [1.5, 0.5, 0.25, 2.0, 1.0]


UNNEST = ("UNNEST(AI.SAE_FEATURES(PROMPT('{0}\\n\\nWhat is this review about?', "
          "r.body), pooling => 'last')) AS f")


def _unnest_oracle():
    # document d has feature d at strength d + 1, and feature 100 at 0.5
    return Oracle(features=lambda spec, document: [
        {"feature": document, "strength": float(document + 1)},
        {"feature": 100, "strength": 0.5}])


def test_unnest_returns_a_row_per_feature_and_where_keeps_strong_ones(session):
    sql = session.sql(
        f"SELECT r.id, f.feature FROM reviews r, {UNNEST} "
        f"WHERE f.strength > 1 ORDER BY f.strength DESC")
    built = (session.docs("reviews").alias("r")
             .ai_sae_features(quail.prompt(QUESTION, quail.col("r.body")),
                              pooling="last", name="f")
             .unnest("f").where(quail.col("f.strength") > 1)
             .order_by(quail.col("f.strength").desc())
             .select("r.id", "f.feature"))
    for query in (sql, built):
        rows = finish(query, _unnest_oracle()).collect()
        assert rows.to_pylist() == [{"r.id": 11, "f.feature": 2},
                                    {"r.id": 9, "f.feature": 1}]


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


def _two_calls(session):
    return (session.docs("reviews").alias("r")
            .ai_sae_features(quail.prompt(QUESTION, quail.col("r.body")),
                             pooling="last", name="f")
            .ai_sae_features(quail.prompt("{0}\n\nWho wrote it?",
                                          quail.col("r.body")),
                             pooling="max", name="g"))


TWO_CALLS = ("AI.SAE_FEATURES(PROMPT('{0}\\n\\nWhat is this review about?', "
             "r.body), pooling => 'last'), AI.SAE_FEATURES(PROMPT('{0}\\n\\n"
             "Who wrote it?', r.body), pooling => 'max')")


def test_two_lists_in_one_unnest_zip_and_pad_the_shorter_with_nulls(session):
    sql = session.sql(
        f"SELECT r.id, f.feature, g.feature FROM reviews r, "
        f"UNNEST({TWO_CALLS}) AS u(f, g) WHERE r.id = 9")
    built = (_two_calls(session).where(quail.col("r.id") == 9).unnest("f", "g")
             .select("r.id", "f.feature", "g.feature"))
    for query in (sql, built):
        rows = finish(query, _two_lists_oracle()).collect()
        assert rows.to_pylist() == [
            {"r.id": 9, "f.feature": 1, "g.feature": 1},
            {"r.id": 9, "f.feature": 100, "g.feature": None}]


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
        nodes = [type(node).__name__ for node in query.plan().nodes]
        start = nodes.index("Project")
        assert nodes[start:start + 4] == ["Project", "Unnest", "Filter", "Unnest"]
        rows = finish(query, _unnest_oracle()).collect()
        assert rows.to_pylist() == [
            {"r.id": 9, "f.feature": 1, "g.feature": 1},
            {"r.id": 9, "f.feature": 1, "g.feature": 100},
            {"r.id": 11, "f.feature": 2, "g.feature": 2},
            {"r.id": 11, "f.feature": 2, "g.feature": 100}]


def test_unnest_needs_named_feature_lists_and_their_fields(session):
    bad = {
        "unnamed lists": f"SELECT r.id FROM reviews r, UNNEST({TWO_CALLS}) AS u",
        "a list named twice": f"SELECT r.id FROM reviews r, {UNNEST}, {UNNEST}",
        "no feature call": "SELECT f.feature FROM reviews r, UNNEST(r.body) AS f",
        "unknown field": f"SELECT f.weight FROM reviews r, {UNNEST}",
        "mixed test": f"SELECT f.feature FROM reviews r, {UNNEST} "
                      f"WHERE f.strength > r.id",
        "alias taken": f"SELECT f.feature FROM reviews f, "
                       f"{UNNEST.replace('r.body', 'f.body')}",
    }
    for name, sql in bad.items():
        with pytest.raises(CompileError):
            session.sql(sql)


def test_a_cte_can_unnest_features_for_the_query_that_reads_it(session):
    query = session.sql(
        f"WITH unnested AS (SELECT r.id, f.feature FROM reviews r, {UNNEST} "
        f"WHERE f.strength > 1) "
        f"SELECT feature, COUNT(*) AS n FROM unnested GROUP BY feature "
        f"ORDER BY feature")
    rows = finish(query, _unnest_oracle()).collect()
    assert [row["n"] for row in rows.to_pylist()] == [1, 1]
    assert rows.column(0).to_pylist() == [1, 2]
