"""Input snapshots stage in private files and leave nothing behind on failure."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pyarrow as pa
import pytest

from quail.catalog import ArrowDatasetProvider, DocumentProvider
from quail.remote import artifacts, inputs


def test_a_failed_snapshot_closes_its_scan_and_leaves_no_staging(tmp_path, monkeypatch):
    provider = ArrowDatasetProvider(
        SimpleNamespace(schema=pa.schema([("id", pa.int64())])), "id")
    opened, closed = [], []

    def scan(self, request):
        source = object()
        opened.append(source)
        return SimpleNamespace(close=lambda: closed.append(source))

    monkeypatch.setattr(ArrowDatasetProvider, "scan", scan)
    workdir = tmp_path / "snapshots"
    workdir.write_bytes(b"a file where the directory should be")
    with pytest.raises(OSError):
        inputs.describe(provider, workdir)
    assert closed == opened

    table = DocumentProvider.from_table(pa.table({"id": [1, 2]}), id_col="id")
    monkeypatch.setattr(inputs, "file_digest", lambda _path: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        inputs.describe(table, tmp_path / "staging")
    assert list((tmp_path / "staging").iterdir()) == []


def test_concurrent_snapshots_of_one_provider_stage_apart(tmp_path, monkeypatch):
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
