"""Document provider tests."""

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from datasets import Dataset

from quail.catalog import (
    ArrowDatasetProvider,
    DocumentProvider,
    ScanRequest,
)
from quail.logical import CompileError


def test_from_dataset_reads_only_id_and_requested_column():
    dataset = ds.dataset(pa.table({
        "id": ["a", "b"],
        "body": ["first", "second"],
        "unused": [1, 2],
    }))

    provider = DocumentProvider.from_dataset(dataset, id_col="id")
    table = provider.read_column("body")

    assert isinstance(provider, ArrowDatasetProvider)
    assert provider.columns == ("id", "body", "unused")
    assert table.column_names == ["id", "body"]
    assert table.to_pydict() == {
        "id": ["a", "b"],
        "body": ["first", "second"],
    }


def test_from_dataset_rejects_non_dataset_and_missing_id():
    with pytest.raises(TypeError, match="pyarrow.dataset.Dataset"):
        DocumentProvider.from_dataset(
            pa.table({"id": ["a"], "body": ["first"]}), id_col="id")

    dataset = ds.dataset(pa.table({"body": ["first"]}))
    with pytest.raises(CompileError, match="id column 'id'"):
        DocumentProvider.from_dataset(dataset, id_col="id")


def test_from_parquet_builds_dataset(tmp_path):
    path = tmp_path / "documents.parquet"
    pq.write_table(pa.table({
        "id": ["a", "b"],
        "body": ["first", "second"],
    }), path)

    provider = DocumentProvider.from_parquet(str(path), id_col="id")

    assert isinstance(provider, ArrowDatasetProvider)
    assert provider.read_column("body").to_pydict() == {
        "id": ["a", "b"],
        "body": ["first", "second"],
    }


def test_scan_returns_bounded_batches_and_only_requested_columns():
    dataset = ds.dataset(pa.table({
        "id": [str(index) for index in range(10)],
        "body": [f"body {index}" for index in range(10)],
        "unused": list(range(10)),
    }))
    provider = DocumentProvider.from_dataset(dataset, id_col="id")

    reader = provider.scan(ScanRequest(
        columns=("body",), limit=5, batch_rows=2
    ))
    batches = list(reader)

    assert [batch.num_rows for batch in batches] == [2, 2, 1]
    assert all(batch.schema.names == ["body"] for batch in batches)


def test_hugging_face_provider_reads_through_arrow_dataset(monkeypatch):
    class Info:
        features = {"id": object(), "body": object(), "unused": object()}

    class Builder:
        info = Info()

    monkeypatch.setattr(
        "datasets.load_dataset_builder", lambda *args, **kwargs: Builder())
    monkeypatch.setattr(
        "datasets.load_dataset",
        lambda *args, **kwargs: Dataset.from_dict({
            "id": ["a", "b"],
            "body": ["first", "second"],
            "unused": [1, 2],
        }))

    provider = DocumentProvider.from_hf(
        "owner/documents", id_col="id", split="test", config="plain")
    table = provider.read_column("body")

    assert table.column_names == ["id", "body"]
    assert table.to_pydict() == {
        "id": ["a", "b"],
        "body": ["first", "second"],
    }
