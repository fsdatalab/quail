"""Local kernel cache defaults and Modal volume helpers."""

import os

import modal

from quail.runtime import volumes
from quail.runtime.compute import default_local_caches


def test_default_local_caches_sets_unset_variables(monkeypatch, tmp_path):
    for name in ("VLLM_CACHE_ROOT", "DG_CACHE_DIR", "DG_JIT_CACHE_DIR",
                 "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
                 "PYTORCH_CUDA_ALLOC_CONF", "VLLM_USE_FLASHINFER_SAMPLER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TRITON_CACHE_DIR", "/elsewhere/triton")

    root = default_local_caches()

    assert root == str(tmp_path / ".cache" / "quail" / "kernels")
    assert os.environ["DG_CACHE_DIR"] == os.path.join(root, "deep_gemm")
    assert os.environ["VLLM_CACHE_ROOT"] == os.path.join(root, "vllm")
    assert os.environ["TRITON_CACHE_DIR"] == "/elsewhere/triton"
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_volume_commits_are_skipped_outside_modal(monkeypatch):
    class Explodes:
        def commit(self):
            raise AssertionError("must not commit outside Modal")

    monkeypatch.setattr(volumes, "results_vol", Explodes())
    monkeypatch.setattr(volumes, "kernel_cache", Explodes())
    assert modal.is_local()
    volumes.commit_results()
    volumes.commit_kernel_cache()


def test_volume_commits_run_inside_modal(monkeypatch):
    committed = []

    class Records:
        def __init__(self, name):
            self.name = name

        def commit(self):
            committed.append(self.name)

    monkeypatch.setattr(volumes.modal, "is_local", lambda: False)
    monkeypatch.setattr(volumes, "results_vol", Records("results"))
    monkeypatch.setattr(volumes, "kernel_cache", Records("kernels"))
    volumes.commit_results()
    volumes.commit_kernel_cache()
    assert committed == ["results", "kernels"]


def test_run_record_path_needs_writable_results_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(volumes, "RESULTS_ROOT", str(tmp_path / "missing"))
    assert volumes.run_record_path() is None
    monkeypatch.setattr(volumes, "RESULTS_ROOT", str(tmp_path))
    path = volumes.run_record_path()
    assert path.startswith(str(tmp_path / "runs" / "run_"))
    assert path.endswith(".json")
