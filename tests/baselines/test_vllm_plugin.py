"""Model lookup caches must not own loaded models."""

import gc
import sys
import weakref
from types import SimpleNamespace

import pytest

from quail.baselines import vllm_decision, vllm_plugin


@pytest.mark.parametrize("cached_before_registration", [False, True])
def test_language_model_lookup_releases_closed_models(
        monkeypatch, cached_before_registration):
    class Model:
        pass

    model = Model()
    model.child = Model()
    parent_ref, child_ref = weakref.ref(model), weakref.ref(model.child)
    interfaces = SimpleNamespace(_language_model_by_module={})
    if cached_before_registration:
        interfaces._language_model_by_module[model] = model.child
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models",
                        SimpleNamespace(interfaces=interfaces))
    monkeypatch.setitem(sys.modules, "vllm.renderers.registry", SimpleNamespace(
        RENDERER_REGISTRY=SimpleNamespace(renderers={
            "gigatoken": ("vllm.renderers.hf", "HfRenderer")})))
    monkeypatch.setitem(sys.modules, "vllm.tokenizers", SimpleNamespace(
        TokenizerRegistry=SimpleNamespace(tokenizers={
            "gigatoken": ("quail.baselines.vllm", "GigatokenVLLMTokenizer")})))
    monkeypatch.setattr(vllm_decision, "register", lambda: None)

    vllm_plugin.register()
    cache = interfaces._language_model_by_module
    if cached_before_registration:
        assert cache[model] is model.child
    else:
        cache[model] = model.child
    vllm_plugin.register()
    assert interfaces._language_model_by_module is cache
    assert cache[model] is model.child
    del model
    gc.collect()
    assert parent_ref() is None
    assert child_ref() is None
    assert not cache
