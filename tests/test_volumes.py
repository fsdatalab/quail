"""CPU checks for the volume commit helpers."""

from quail.runtime import volumes


def test_commit_skips_volumes_that_are_not_attached(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(volumes.modal, "is_local", lambda: False)
    monkeypatch.setattr(volumes.results_vol, "commit",
                        lambda: calls.append("results"))
    monkeypatch.setattr(volumes.kernel_cache, "commit",
                        lambda: calls.append("kernels"))
    monkeypatch.setattr(volumes, "RESULTS_ROOT", str(tmp_path / "missing"))
    monkeypatch.setattr(volumes, "KERNEL_CACHE_ROOT", str(tmp_path))

    volumes.commit_results()
    volumes.commit_kernel_cache()

    assert calls == ["kernels"]


def test_commit_is_a_no_op_outside_modal(monkeypatch, tmp_path):
    monkeypatch.setattr(volumes.modal, "is_local", lambda: True)
    monkeypatch.setattr(volumes, "RESULTS_ROOT", str(tmp_path))
    monkeypatch.setattr(volumes.results_vol, "commit",
                        lambda: (_ for _ in ()).throw(AssertionError()))

    volumes.commit_results()
