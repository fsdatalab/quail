"""Liquid's d1 decision models: prompt layout, answer ids, spec, and support."""

from types import SimpleNamespace

import pytest

from quail.baselines.request import RequestBackend
from quail.logical import ColumnRef
from quail.specs import D1_2B_BF16, H100_SXM

REVIEW = ColumnRef(alias="r", provider="reviews", column="body")
MOVIE = ColumnRef(alias="m", provider="movies", column="plot")
# d1-2B's prompt.render for a text state and a yes/no question, with the
# chat template's empty thinking block as the lead
D1_PROMPT = ("<|startoftext|><|im_start|>user\n{state}\n\n\nQUESTION:\n"
             "{question}\n\nReply with yes or no only.<|im_end|>\n"
             "<|im_start|>assistant\n<think></think>")
# text_config.layer_types of LiquidAI/d1-2B at the spec's revision
CHECKPOINT_LAYER_TYPES = (
    "conv", "conv", "full_attention", "conv", "sliding_attention", "conv",
    "full_attention", "conv", "sliding_attention", "conv", "full_attention",
    "conv", "sliding_attention", "conv", "full_attention", "conv",
    "sliding_attention", "conv", "full_attention", "conv",
    "sliding_attention", "conv", "full_attention", "conv")


def test_the_vllm_baseline_refuses_d1_2b():
    engine = SimpleNamespace(label="stock vLLM", kind="vllm")
    backend = RequestBackend(name="stock", engine=engine,
                             filter_submission="operator")

    support = backend.supports(D1_2B_BF16, H100_SXM, 1)

    assert not support.supported
    assert "lfm3" in support.reason


def test_d1_2b_extracts_but_refuses_map_with_a_reason(tmp_path):
    import pyarrow as pa

    import quail
    from quail.api.config import EngineConfig
    from quail.catalog import DocumentProvider
    from quail.physical.plan import Refusal

    session = quail.Session(EngineConfig(model="d1-2b-bf16", device="h100-sxm"),
                            tokenizer=str.split)
    session.register("reviews", DocumentProvider.from_table(
        pa.table({"id": [0, 1], "body": ["good food", "slow service"]}), id_col="id"))
    extracted = session.sql(
        "SELECT r.id, AI.EXTRACT(r.body, {'dish': 'Which dish?'}) AS x "
        "FROM reviews r").plan()
    mapped = session.sql(
        "SELECT r.id, AI.MAP(r.body, 'Name the topic.', 8) AS topic "
        "FROM reviews r").plan()

    assert not isinstance(extracted, Refusal)
    assert isinstance(mapped, Refusal)
    assert "writes text" in mapped.reasons[0]
    session.close()


def test_a_d1_join_makes_the_longer_table_the_state(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    import quail
    from quail.api.config import EngineConfig
    from quail.catalog import DocumentProvider
    from quail.physical import AiJoin

    body = " ".join(["word"] * 300)
    pq.write_table(pa.table({"id": list(range(40)), "body": [body] * 40}),
                   tmp_path / "reviews.parquet")
    pq.write_table(pa.table({"id": list(range(12)), "aspect": ["acting"] * 12}),
                   tmp_path / "aspects.parquet")
    session = quail.Session(EngineConfig(model="d1-2b-bf16", device="h100-sxm"),
                            tokenizer=str.split)
    for name in ("reviews", "aspects"):
        session.register(name, DocumentProvider.from_parquet(
            str(tmp_path / f"{name}.parquet"), id_col="id"))

    plan = session.sql(
        "SELECT r.id, a.id FROM aspects a JOIN reviews r ON AI_FILTER(PROMPT("
        "'Does {1} discuss {0}?', a.aspect, r.body))").plan()

    (join,) = [node for node in plan.nodes if isinstance(node, AiJoin)]
    assert join.anchor == "r"
    assert join.document_head[-2:] == ("reviews", "record:")
    assert join.label_token_ids == (("a", ("aspects", "record:")),)
    assert join.question_header_token_ids == ("QUESTION:",)
    assert not join.keep_kv
    session.close()


def test_an_option_whose_code_is_two_tokens_takes_d1s_first_free_fallback():
    from quail.common.prompts import option_codes
    from quail.physical.lowering import option_aliases

    def tokenizer(text):
        return list(text) if text.isupper() else [text]

    codes = option_codes([f"label {index}" for index in range(27)])

    assert codes[-1] == "AA"
    assert option_aliases(codes, tokenizer)[-1] == "00"


def test_an_lfm3_checkpoint_needs_its_text_tensors_and_nothing_unknown():
    from quail.backends.quail.executor.models.lfm3 import (
        check_tensors,
        text_tensors,
    )

    expected = text_tensors(["conv", "full_attention"], "model.language_model")
    vision = ["model.vision_tower.encoder.weight", "model.multi_modal_projector.w"]

    assert "model.language_model.layers.1.self_attn.gate_proj.weight" in expected
    check_tensors([*expected, *vision], expected)
    with pytest.raises(ValueError, match="missing"):
        check_tensors(expected[1:], expected)
    with pytest.raises(ValueError, match="does not use"):
        check_tensors([*expected, "model.audio_tower.weight"], expected)


def test_a_d1_classification_of_joined_rows_follows_its_join(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    import quail
    from quail.api.config import EngineConfig
    from quail.catalog import DocumentProvider
    from quail.physical import AiClassify, AiJoin
    from quail.physical.pipelines import build_pipelines

    body = " ".join(["word"] * 300)
    pq.write_table(pa.table({"id": list(range(40)), "body": [body] * 40}),
                   tmp_path / "reviews.parquet")
    pq.write_table(pa.table({"id": list(range(12)), "aspect": ["acting"] * 12}),
                   tmp_path / "aspects.parquet")
    session = quail.Session(EngineConfig(model="d1-2b-bf16", device="h100-sxm"),
                            tokenizer=str.split)
    for name in ("reviews", "aspects"):
        session.register(name, DocumentProvider.from_parquet(
            str(tmp_path / f"{name}.parquet"), id_col="id"))

    plan = session.sql(
        "SELECT r.id, a.id, AI.CLASSIFY(PROMPT('What does DOCUMENT {0} say "
        "about DOCUMENT {1}?', r.body, a.aspect), ARRAY['praise', 'complaint']) "
        "AS stance FROM reviews r JOIN aspects a ON AI_FILTER(PROMPT("
        "'Does {0} mention {1}?', r.body, a.aspect))").plan()

    (join,) = [node for node in plan.nodes if isinstance(node, AiJoin)]
    (label,) = [node for node in plan.nodes if isinstance(node, AiClassify)]
    spec = label.spec
    assert spec.scoring == "set_choice" and spec.aliases == ("r", "a")
    assert spec.join_layout == (("QUESTION:",), ("aspects", "record:"))
    assert spec.prompt_token_parts[0][-2:] == ("reviews", "record:")
    assert build_pipelines(plan.graph)[join.node_id] == (join, label)
    # priced by its own estimate over each anchor's kept pairs, not as a
    # pass group of one question for each pair
    assert label.pass_group == -1
    assert plan.estimates[label.node_id]["seconds"] == pytest.approx(
        spec.estimated_seconds)
    session.close()


def test_d1_2b_fp8_is_d1_2b_with_fp8_weights():
    from dataclasses import fields

    from quail.specs import D1_2B_FP8

    differ = {field.name for field in fields(D1_2B_BF16)
              if getattr(D1_2B_BF16, field.name) != getattr(D1_2B_FP8, field.name)}

    assert differ == {"name", "w_bytes", "weight_precision", "w_mem_bytes"}
    assert D1_2B_FP8.weight_precision == "fp8"
