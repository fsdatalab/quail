"""The per-architecture forward passes, their attention paths, and support checks."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from quail.backends.quail.backend import QuailBackend
from quail.backends.quail.executor.models import build_pipeline, supported_archs
from quail.backends.quail.executor.models.qwen3 import Qwen3Pipeline
from quail.backends.request import RequestBackend
from quail.cost.budgets import tree_attention_allowed
from quail.specs import (
    DECISION_2_KAI_0_6B_BF16,
    H100_SXM,
    MODELS,
    QWEN3_4B_FP8,
    QWEN3_RERANKER_0_6B_BF16,
    QWEN3_RERANKER_4B_BF16,
)


def test_backends_support_registered_generative_models_and_refuse_others():
    for model in MODELS.values():
        support = QuailBackend().supports(model, H100_SXM, 1)
        if model.arch in supported_archs():
            assert support.supported
        else:
            # registered for the request backends alone
            assert not support.supported
            assert model.arch in support.reason

    spec = replace(QWEN3_4B_FP8, name="other", arch="other")
    with pytest.raises(ValueError, match="other"):
        build_pipeline(spec, model=None, arena=None)
    support = QuailBackend().supports(spec, H100_SXM, 1)
    assert not support.supported
    assert "other" in support.reason

    engine = SimpleNamespace(label="stock vLLM", kind="vllm")
    backend = RequestBackend(name="stock", engine=engine, filter_submission="operator")
    assert backend.supports(QWEN3_4B_FP8, H100_SXM, 1).supported
    support = backend.supports(QWEN3_RERANKER_0_6B_BF16, H100_SXM, 1)
    assert not support.supported
    assert "generative" in support.reason


def _model(torch, dtype):
    weight = SimpleNamespace(dtype=dtype, shape=(4096, 1024))
    attn = SimpleNamespace(num_heads=16, num_kv_heads=8, head_dim=128,
                           rotary_emb=None, qkv_proj=SimpleNamespace(weight=weight))
    mlp = SimpleNamespace(gate_up_proj=SimpleNamespace(weight=weight))
    layer = SimpleNamespace(self_attn=attn, mlp=mlp)
    return SimpleNamespace(model=SimpleNamespace(
        layers=[layer], embed_tokens=None, norm=None))


def test_tree_attention_runs_at_either_precision_on_models_of_2b_and_up():
    torch = pytest.importorskip("torch")
    for dtype_name, spec, expected in [
            ("float8_e4m3fn", None, True), ("bfloat16", None, True),
            ("float8_e4m3fn", QWEN3_4B_FP8, True),
            ("bfloat16", QWEN3_RERANKER_4B_BF16, True),
            ("bfloat16", QWEN3_RERANKER_0_6B_BF16, False),
            ("bfloat16", DECISION_2_KAI_0_6B_BF16, False)]:
        engines = []

        def engine(arena, **kwargs):
            engines.append(kwargs)
            return SimpleNamespace(**kwargs)

        pipeline = Qwen3Pipeline(_model(torch, getattr(torch, dtype_name)), None,
                                 spec=spec, engine_class=engine)
        assert engines[0]["fp8"] is (dtype_name == "float8_e4m3fn")
        assert pipeline.tree_attention is expected, (dtype_name, spec)
        assert pipeline.max_chunk_tokens == (2**31 - 1) // 4096, dtype_name
    assert {name for name, model in MODELS.items()
            if tree_attention_allowed(model)} == {
        "qwen3-4b-fp8", "qwen3-32b-fp8", "qwen3-reranker-4b-bf16"}
