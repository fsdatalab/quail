"""Document provider tests."""

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from datasets import Dataset

from quail.catalog import DocumentProvider
from quail.logical import CompileError


def test_from_dataset_reads_only_id_and_requested_column():
    dataset = ds.dataset(pa.table({
        "id": ["a", "b"],
        "body": ["first", "second"],
        "unused": [1, 2],
    }))

    provider = DocumentProvider.from_dataset(dataset, id_col="id")
    table = provider.read_column("body")

    assert provider.kind == "dataset"
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

    assert provider.kind == "dataset"
    assert provider.read_column("body").to_pydict() == {
        "id": ["a", "b"],
        "body": ["first", "second"],
    }


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
