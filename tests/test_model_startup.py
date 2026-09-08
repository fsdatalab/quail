"""Model files are prepared once before GPU workers load them."""

import sys
from types import SimpleNamespace

import pytest

from quail.builtins import built_in_registry
from quail.executor import model


@pytest.fixture(autouse=True)
def clear_model_paths():
    model.resolve_model_path.cache_clear()
    yield
    model.resolve_model_path.cache_clear()


def test_model_download_is_cached_by_revision(monkeypatch, tmp_path):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / kwargs["revision"])

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    first = model.resolve_model_path("Qwen/model", "revision-a")
    assert model.resolve_model_path("Qwen/model", "revision-a") == first
    assert model.resolve_model_path("Qwen/model", "revision-b") != first
    assert [call["revision"] for call in calls] == ["revision-a", "revision-b"]
    assert all(call["repo_id"] == "Qwen/model" for call in calls)
    assert "*.safetensors" in calls[0]["allow_patterns"]


def test_model_loader_uses_local_directory_without_hub_calls(monkeypatch, tmp_path):
    arguments = []
    weights = object()

    def unexpected_download(**kwargs):
        raise AssertionError("local model load contacted the Hub")

    class EngineArgs:
        def __init__(self, **kwargs):
            arguments.append(kwargs)

        def create_engine_config(self):
            return object()

    from contextlib import nullcontext

    monkeypatch.setattr("huggingface_hub.snapshot_download", unexpected_download)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        empty_cache=lambda: None, synchronize=lambda: None)))
    monkeypatch.setitem(sys.modules, "vllm.config", SimpleNamespace(
        set_current_vllm_config=lambda config: nullcontext()))
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", SimpleNamespace(
        EngineArgs=EngineArgs))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.model_loader",
                        SimpleNamespace(get_model=lambda **kwargs: weights))
    monkeypatch.setattr(model, "_install_single_rank_groups", lambda torch: None)
    monkeypatch.setattr(model, "retain_answer_head", lambda *args: None)

    assert model.load_model(str(tmp_path), answer_token_ids=[1, 2]) is weights
    assert arguments == [{
        "model": str(tmp_path.resolve()), "dtype": "auto", "enforce_eager": True,
    }]


def test_parent_prepares_model_before_starting_children(monkeypatch, tmp_path):
    from quail.backends.quail import worker

    events = []
    registry = built_in_registry()

    def resolve(name, revision):
        events.append("resolve")
        assert revision == registry.model("qwen3-4b-fp8").revision
        return str(tmp_path)

    def ensure(count):
        events.append("children")
        assert count == 8

    def round_fn(kind, subs):
        events.append("registry")
        assert kind == "registry"
        assert len(subs) == 8
        assert all(sub == {"registry": registry, "model_path": str(tmp_path)}
                   for sub in subs)

    clock = iter([10.0, 12.0])
    monkeypatch.setattr(worker.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(worker, "resolve_model_path", resolve)
    monkeypatch.setattr(worker, "_ensure_children", ensure)
    monkeypatch.setattr(worker, "_round", round_fn)
    monkeypatch.setattr(worker, "execute_distributed_graph", lambda *args: {
        "_outputs": {}, "boot_s": 3.0, "boot": {"kind": "cold"}, "wall_s": 7.0,
    })
    monkeypatch.setattr("quail.runtime.volumes.commit_results", lambda: None)
    monkeypatch.setattr("quail.runtime.volumes.commit_kernel_cache", lambda: None)
    result = worker.execute_quail_multi({
        "docs": {}, "workers": 8, "model": "qwen3-4b-fp8",
        "physical_plan": {"device": "h100-sxm"},
    }, registry, object())
    assert events == ["resolve", "children", "registry"]
    assert result.metrics["boot_s"] == 5.0
    assert result.metrics["boot"]["model_files_s"] == 2.0
    assert result.metrics["wall_s"] == 7.0
