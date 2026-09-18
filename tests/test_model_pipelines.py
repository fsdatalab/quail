"""The per-architecture forward pass registry and the support checks."""

from dataclasses import replace

import pytest

from quail.backends.quail.backend import QuailBackend
from quail.backends.quail.executor.models import build_pipeline, supported_archs
from quail.backends.request import RequestBackend
from quail.specs import H100_SXM, MODELS, QWEN3_4B_FP8, QWEN3_RERANKER_0_6B_BF16


def test_every_registered_model_has_a_forward_pass():
    for model in MODELS.values():
        assert model.arch in supported_archs()


def test_unknown_architecture_is_refused_before_loading():
    spec = replace(QWEN3_4B_FP8, name="other", arch="other")
    with pytest.raises(ValueError, match="other"):
        build_pipeline(spec, model=None, arena=None, attention_mode="unified")
    support = QuailBackend().supports(spec, H100_SXM, 1)
    assert not support.supported
    assert "other" in support.reason


def test_quail_backend_accepts_generative_and_reranker_models():
    backend = QuailBackend()
    assert backend.supports(QWEN3_4B_FP8, H100_SXM, 1).supported
    assert backend.supports(QWEN3_RERANKER_0_6B_BF16, H100_SXM, 1).supported


def test_request_backends_serve_generative_models_only():
    class Engine:
        label = "stock vLLM"
        kind = "vllm"

    backend = RequestBackend(name="stock", engine=Engine(),
                             filter_submission="operator")
    assert backend.supports(QWEN3_4B_FP8, H100_SXM, 1).supported
    support = backend.supports(QWEN3_RERANKER_0_6B_BF16, H100_SXM, 1)
    assert not support.supported
    assert "generative" in support.reason
