"""Cache paths remain under the caller's control."""

import os

from quail.executor.loop import _marker_path


def test_warmup_marker_defaults_and_user_cache_paths(monkeypatch, tmp_path):
    monkeypatch.delenv("QUAIL_CACHE_DIR", raising=False)
    expected = os.path.expanduser("~/.cache/quail/kernels")
    assert _marker_path("Qwen/model", 100) == (
        f"{expected}/quail-warm-Qwen--model-100.json")

    for name in ("HF_HOME", "VLLM_CACHE_ROOT", "DG_CACHE_DIR",
                 "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        monkeypatch.setenv(name, str(tmp_path / name))
    monkeypatch.setenv("QUAIL_CACHE_DIR", str(tmp_path / "markers"))
    before = dict(os.environ)
    assert _marker_path("Qwen/model", 100) == str(
        tmp_path / "markers" / "quail-warm-Qwen--model-100.json")
    assert dict(os.environ) == before
