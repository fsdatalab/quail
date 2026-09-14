"""Model files are prepared once before GPU workers load them."""

import pytest

from quail.backends.quail.executor import model
from quail.builtins import built_in_registry


@pytest.fixture(autouse=True)
def clear_model_paths():
    model.resolve_model_path.cache_clear()
    yield
    model.resolve_model_path.cache_clear()


def test_model_files_cached_before_gpu_children_start(monkeypatch, tmp_path):
    with monkeypatch.context() as patch:
        calls = []

        def download(**kwargs):
            calls.append(kwargs)
            return str(tmp_path / kwargs["revision"])

        patch.setattr("huggingface_hub.snapshot_download", download)
        first = model.resolve_model_path("Qwen/model", "revision-a")
        assert model.resolve_model_path("Qwen/model", "revision-a") == first
        assert model.resolve_model_path("Qwen/model", "revision-b") != first
        assert [call["revision"] for call in calls] == ["revision-a", "revision-b"]
        assert all(call["repo_id"] == "Qwen/model" for call in calls)
        assert "*.safetensors" in calls[0]["allow_patterns"]

    with monkeypatch.context() as patch:
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
            events.append("boot")
            assert kind == "boot"
            assert len(subs) == 8
            assert all(sub["registry"] is registry
                       and sub["model_path"] == str(tmp_path) for sub in subs)
            return [{"boot_s": 3.0, "kind": "cold"}] * 8

        def run_graph(*args):
            assert events == ["resolve", "children", "boot"]
            events.append("query")
            return {"_outputs": {}, "boot_s": 0.0, "wall_s": 7.0}

        clock = iter([10.0, 12.0, 15.0])
        patch.setattr(worker.time, "perf_counter", lambda: next(clock))
        patch.setattr(worker, "resolve_model_path", resolve)
        patch.setattr(worker, "_ensure_children", ensure)
        patch.setattr(worker, "_round", round_fn)
        patch.setattr(worker, "execute_distributed_graph", run_graph)
        result = worker.execute_quail_multi({
            "docs": {}, "workers": 8, "model": "qwen3-4b-fp8",
            "physical_plan": {"device": "h100-sxm"},
            "chunk_tokens": 100, "true_ids": [1], "false_ids": [2],
        }, registry, object())
        assert events == ["resolve", "children", "boot", "query"]
        assert result.metrics["boot_s"] == 5.0
        assert result.metrics["boot"]["model_files_s"] == 2.0
        assert result.metrics["wall_s"] == 7.0
