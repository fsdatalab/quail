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


def test_features_plan_names_the_reading_and_round_trips(session):
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


def test_sql_binds_the_reading_and_rejects_a_bad_one(session):
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
