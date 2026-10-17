"""Concurrent input preparation owns separate staging files."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pyarrow as pa
import pytest

from quail.catalog import ArrowDatasetProvider, DocumentProvider
from quail.remote import artifacts, inputs


@pytest.mark.parametrize("failure", ["workdir", "staging"])
def test_failed_snapshot_setup_does_not_leave_a_scan_open(
        tmp_path, monkeypatch, failure):
    provider = ArrowDatasetProvider(
        SimpleNamespace(schema=pa.schema([("id", pa.int64())])), "id")
    opened, closed = [], []

    def scan(self, request):
        source = object()
        opened.append(source)
        return SimpleNamespace(close=lambda: closed.append(source))

    monkeypatch.setattr(ArrowDatasetProvider, "scan", scan)
    workdir = tmp_path / "snapshots"
    if failure == "workdir":
        workdir.write_bytes(b"existing file")
    else:
        def fail(**_kwargs):
            raise OSError("no space for staging")

        monkeypatch.setattr(inputs.tempfile, "TemporaryDirectory", fail)
    with pytest.raises(OSError):
        inputs.describe(provider, workdir)
    assert closed == opened


def test_concurrent_snapshots_of_one_provider_do_not_share_staging(
        tmp_path, monkeypatch):
    provider = DocumentProvider.from_table(pa.table({"id": [1, 2]}), id_col="id")
    ready = threading.Barrier(2)
    digest = inputs.file_digest

    def hash_together(path):
        value = digest(path)
        ready.wait(timeout=5)
        return value

    monkeypatch.setattr(inputs, "file_digest", hash_together)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(inputs.describe, provider, tmp_path) for _ in range(2)]
        first, second = [future.result(timeout=10) for future in futures]
    assert first == second
    assert artifacts.read_table(first.upload_path).equals(provider.table)
    assert list(tmp_path.iterdir()) == [first.upload_path]


def test_failed_snapshot_hash_removes_staging(tmp_path, monkeypatch):
    provider = DocumentProvider.from_table(pa.table({"id": [1, 2]}), id_col="id")

    def fail(_path):
        raise OSError("cannot read snapshot")

    monkeypatch.setattr(inputs, "file_digest", fail)
    with pytest.raises(OSError, match="cannot read snapshot"):
        inputs.describe(provider, tmp_path)
    assert list(tmp_path.iterdir()) == []
