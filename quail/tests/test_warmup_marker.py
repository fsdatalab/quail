"""The warmup marker: path layout and identity stability.

The marker decides whether a boot runs the compile pass (once ever
per stack+model+budget) or the touch pass (every container). A wrong
path or an unstable identity silently recompiles every boot or never
compiles at all, so both are pinned here. No GPU: the identity
builder takes the torch module as a parameter, so a stub stands in.
"""

import json
from types import SimpleNamespace

from quail.executor.loop import _marker_identity, _marker_path


def _stub_torch():
    return SimpleNamespace(
        __version__="2.9.0",
        version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(get_device_name=lambda: "NVIDIA H100"))


def test_marker_path_uses_kernel_cache_dir(monkeypatch):
    monkeypatch.setenv("DG_CACHE_DIR",
                       "/root/.cache/kernels/deep_gemm")
    path = _marker_path("Qwen/Qwen3-4B-FP8", 110376)
    # next to the caches, so one volume commit persists marker and
    # compiled kernels together; slash flattened for a filename
    assert path == ("/root/.cache/kernels/"
                    "quail-warm-Qwen--Qwen3-4B-FP8-110376.json")


def test_marker_path_without_env(monkeypatch):
    monkeypatch.delenv("DG_CACHE_DIR", raising=False)
    path = _marker_path("m", 8)
    assert path.endswith("quail-kernels/quail-warm-m-8.json")


def test_identity_json_roundtrip_stable():
    ident = _marker_identity(_stub_torch(), "Qwen/Qwen3-4B-FP8",
                             110376)
    assert json.loads(json.dumps(ident)) == ident


def test_identity_changes_with_budget_and_model():
    t = _stub_torch()
    base = _marker_identity(t, "a", 100)
    assert _marker_identity(t, "a", 200) != base
    assert _marker_identity(t, "b", 100) != base
    assert _marker_identity(t, "a", 100) == base
