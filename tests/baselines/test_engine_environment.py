"""Model-specific vLLM process settings last only as long as their engine."""

import os
import sys
from contextlib import ExitStack, nullcontext
from functools import cache
from types import ModuleType, SimpleNamespace

import pytest

from quail.baselines import vllm
from quail.specs import DIFFUSION_GEMMA_26B_FP8, QWEN3_4B_FP8


@pytest.mark.parametrize("initial", [None, "0", "1"])
@pytest.mark.parametrize("fails", [False, True])
def test_diffusion_process_mode_is_restored_on_release(monkeypatch, initial, fails):
    name = "VLLM_ENABLE_V1_MULTIPROCESSING"
    if initial is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, initial)
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    envs = ModuleType("vllm.envs")

    @cache
    def read(key):
        if key != name:
            raise AttributeError(key)
        return bool(int(os.environ.get(name, "1")))

    envs.__getattr__ = read
    envs.disable_envs_cache = read.cache_clear
    expected = getattr(envs, name)
    modes = []

    def construct(**kwargs):
        modes.append(getattr(envs, name))
        if fails and len(modes) == 1:
            raise RuntimeError("model load failed")
        return SimpleNamespace(llm_engine=SimpleNamespace(
            engine_core=SimpleNamespace(shutdown=lambda: None),
            vllm_config=SimpleNamespace(
                cache_config=SimpleNamespace(enable_prefix_caching=True),
                model_config=SimpleNamespace(tokenizer_mode="gigatoken"))))

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        envs=envs, LLM=construct,
        SamplingParams=lambda **kw: SimpleNamespace(**({"logprobs": None} | kw))))
    monkeypatch.setattr(vllm, "diffusion_canvas", lambda spec: nullcontext(()))
    monkeypatch.setattr(vllm, "resolve_model_path", lambda *args: "/model")
    monkeypatch.setattr(vllm, "_capacity", lambda llm: {})
    monkeypatch.setattr(vllm.VLLMClient, "generate", lambda *args: None)
    failure = pytest.raises(RuntimeError, match="model load failed") if fails \
        else nullcontext()
    with failure, ExitStack() as resources:
        vllm.VLLMEngine().boot(DIFFUSION_GEMMA_26B_FP8, [1, 2], resources=resources)
        assert getattr(envs, name) is False
    assert modes == [False]
    assert os.environ.get(name) == initial
    assert getattr(envs, name) == expected
    with ExitStack() as resources:
        vllm.VLLMEngine().boot(QWEN3_4B_FP8, [1, 2], resources=resources)
    assert modes == [False, expected]
