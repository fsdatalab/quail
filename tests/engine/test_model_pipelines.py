"""The per-architecture forward passes, their attention paths, and support checks."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from quail.backends.quail.backend import QuailBackend
from quail.backends.quail.executor.models import build_pipeline, supported_archs
from quail.baselines.request import RequestBackend
from quail.planner.lowering import plan_region
from quail.specs import (
    H100_SXM,
    LFM2_2_6B_BF16,
    MODELS,
    QWEN3_4B_FP8,
    QWEN3_5_4B_BF16,
    QWEN3_RERANKER_0_6B_BF16,
)


def test_backends_support_registered_generative_models_and_refuse_others():
    for model in MODELS.values():
        assert model.arch in supported_archs()
        assert QuailBackend(plan_region).supports(model, H100_SXM, 1).supported

    assert QuailBackend(plan_region).supports(QWEN3_5_4B_BF16, H100_SXM, 1).supported
    assert QuailBackend(plan_region).supports(LFM2_2_6B_BF16, H100_SXM, 1).supported

    spec = replace(QWEN3_4B_FP8, name="other", arch="other")
    with pytest.raises(ValueError, match="other"):
        build_pipeline(spec, model=None, arena=None)
    support = QuailBackend(plan_region).supports(spec, H100_SXM, 1)
    assert not support.supported
    assert "other" in support.reason

    engine = SimpleNamespace(label="stock vLLM", kind="vllm")
    backend = RequestBackend(name="stock", engine=engine, filter_submission="operator")
    assert backend.supports(QWEN3_4B_FP8, H100_SXM, 1).supported
    support = backend.supports(QWEN3_RERANKER_0_6B_BF16, H100_SXM, 1)
    assert not support.supported
    assert "generative" in support.reason
