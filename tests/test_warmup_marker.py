"""Tests for warmup marker path layout and identity stability."""

import json
import multiprocessing as mp
from types import SimpleNamespace

import pytest

from quail.executor import loop
from quail.executor.loop import _marker_identity, _marker_path


def _stub_torch():
    return SimpleNamespace(
        __version__="2.9.0",
        version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(get_device_name=lambda: "NVIDIA H100",
                             synchronize=lambda: None))


def test_marker_path_uses_kernel_cache_dir(monkeypatch):
    monkeypatch.setenv("DG_CACHE_DIR",
                       "/root/.cache/kernels/deep_gemm")
    path = _marker_path("Qwen/Qwen3-4B-FP8", 110376)
    # next to the caches, so one volume commit persists marker and
    # compiled kernels together; slash flattened for a filename
    assert path == ("/root/.cache/kernels/"
                    "quail-warm-Qwen--Qwen3-4B-FP8-110376.json")
    monkeypatch.delenv("DG_CACHE_DIR")
    assert _marker_path("m", 8).endswith(
        "quail-kernels/quail-warm-m-8.json")


def test_identity_changes_with_budget_and_model():
    t = _stub_torch()
    base = _marker_identity(t, "a", 100)
    assert _marker_identity(t, "a", 200) != base
    assert _marker_identity(t, "b", 100) != base
    assert _marker_identity(t, "a", 100) == base


def _concurrent_warmup(path, start, touch, compiles, results):
    loop._marker_path = lambda *args: path
    loop._marker_identity = lambda *args: {"version": 1}

    def compile_once(*args):
        with compiles.get_lock():
            compiles.value += 1

    def touch_together(*args):
        # Both cached workers must warm outside the compilation lock.
        touch.wait(timeout=10)

    loop.compile_kernels = compile_once
    loop.touch_kernels = touch_together
    start.wait(timeout=10)
    result = loop.warm_kernels(_stub_torch(), None, None, None, 100,
                               model_name="model")
    results.put(result["tier"])


def test_concurrent_processes_compile_once_and_touch_in_parallel(tmp_path):
    ctx = mp.get_context("spawn")
    path = str(tmp_path / "warm.json")
    start, touch = ctx.Barrier(3), ctx.Barrier(2)
    compiles = ctx.Value("i", 0)
    results = ctx.Queue()
    workers = [ctx.Process(target=_concurrent_warmup,
                           args=(path, start, touch, compiles, results))
               for _ in range(3)]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=20)
            assert worker.exitcode == 0
        assert compiles.value == 1
        assert sorted(results.get(timeout=2) for _ in workers) == [
            "compile", "touch", "touch"]
        assert json.loads((tmp_path / "warm.json").read_text()) == {"version": 1}
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        results.close()


@pytest.mark.parametrize("failure", ["compile", "synchronize"])
def test_failed_warmup_does_not_publish_completion(monkeypatch, tmp_path, failure):
    path = tmp_path / "warm.json"
    monkeypatch.setattr(loop, "_marker_path", lambda *args: str(path))
    monkeypatch.setattr(loop, "_marker_identity", lambda *args: {"version": 1})

    def fail(*args):
        raise RuntimeError("GPU failed")

    torch = _stub_torch()
    monkeypatch.setattr(loop, "compile_kernels", fail if failure == "compile"
                        else lambda *args: None)
    if failure == "synchronize":
        torch.cuda.synchronize = fail
    with pytest.raises(RuntimeError, match="GPU failed"):
        loop.warm_kernels(torch, None, None, None, 100, model_name="model")
    assert not path.exists()

    torch.cuda.synchronize = lambda: None
    monkeypatch.setattr(loop, "compile_kernels", lambda *args: None)
    assert loop.warm_kernels(torch, None, None, None, 100,
                             model_name="model")["tier"] == "compile"


def test_cached_warmup_exercises_join_path(monkeypatch):
    calls = []
    monkeypatch.setattr(loop, "_forward_warm",
                        lambda *args, **kwargs: calls.append(kwargs))
    loop.touch_kernels(None, None, None, None, 100)
    assert calls == [{"join_chunk": True}]
