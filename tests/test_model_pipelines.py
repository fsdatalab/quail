"""The per-architecture forward passes, their attention paths, and support checks."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from quail.backends.quail.backend import QuailBackend
from quail.backends.quail.executor.models import build_pipeline, supported_archs
from quail.backends.quail.executor.models.qwen3 import Qwen3Pipeline
from quail.backends.request import RequestBackend
from quail.specs import H100_SXM, MODELS, QWEN3_4B_FP8, QWEN3_RERANKER_0_6B_BF16


def test_quail_backend_supports_every_registered_model():
    for model in MODELS.values():
        assert model.arch in supported_archs()
        assert QuailBackend().supports(model, H100_SXM, 1).supported


def test_unknown_architecture_is_refused_before_loading():
    spec = replace(QWEN3_4B_FP8, name="other", arch="other")
    with pytest.raises(ValueError, match="other"):
        build_pipeline(spec, model=None, arena=None)
    support = QuailBackend().supports(spec, H100_SXM, 1)
    assert not support.supported
    assert "other" in support.reason


def test_request_backends_serve_generative_models_only():
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


@pytest.mark.parametrize("dtype_name, expected", [
    ("float8_e4m3fn", "merge_quant"), ("bfloat16", "unified")])
def test_join_attention_follows_the_weight_precision(dtype_name, expected):
    torch = pytest.importorskip("torch")
    engines = []

    def engine(arena, **kwargs):
        engines.append(kwargs)
        return SimpleNamespace(**kwargs)

    pipeline = Qwen3Pipeline(_model(torch, getattr(torch, dtype_name)), None,
                             spec=None, engine_class=engine)
    assert engines[0]["fp8"] is (expected == "merge_quant")
    assert pipeline.join_attention == expected
    assert pipeline.max_chunk_tokens == (2**31 - 1) // 4096
