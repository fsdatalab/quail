"""Document provider tests."""

import pyarrow as pa
import pyarrow.dataset as ds
import pytest
from datasets import Dataset

from quail.catalog import (
    DocumentProvider,
    ScanRequest,
)
from quail.logical import CompileError


def test_table_provider_scans_and_validation(monkeypatch):
    with pytest.raises(TypeError, match="pyarrow.dataset.Dataset"):
        DocumentProvider.from_dataset(
            pa.table({"id": ["a"], "body": ["first"]}), id_col="id")

    dataset = ds.dataset(pa.table({"body": ["first"]}))
    with pytest.raises(CompileError, match="id column 'id'"):
        DocumentProvider.from_dataset(dataset, id_col="id")

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

    with monkeypatch.context() as patch:
        class Info:
            features = {"id": object(), "body": object(), "unused": object()}

        class Builder:
            info = Info()

        patch.setattr(
            "datasets.load_dataset_builder", lambda *args, **kwargs: Builder())
        patch.setattr(
            "datasets.load_dataset",
            lambda *args, **kwargs: Dataset.from_dict({
                "id": ["a", "b"],
                "body": ["first", "second"],
                "unused": [1, 2],
            }))

        provider = DocumentProvider.from_hf(
            "owner/documents", id_col="id", split="test", config="plain")
        table = provider.scan(ScanRequest(("id", "body"))).read_all()

        assert table.column_names == ["id", "body"]
        assert table.to_pydict() == {
            "id": ["a", "b"],
            "body": ["first", "second"],
        }
