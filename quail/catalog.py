"""Table providers and the session catalog."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

import pyarrow as pa

from quail.logical import CompileError


@dataclass(frozen=True)
class ScanRequest:
    """Columns and ordinary restrictions requested from a table."""

    columns: tuple[str, ...]
    filter: Any = None
    limit: int | None = None
    batch_rows: int = 65_536


@dataclass(frozen=True)
class TableStatistics:
    """Statistics available without reading table values."""

    row_count: int | None = None
    byte_count: int | None = None


class TableProvider(Protocol):
    """Source of bounded Arrow record batches."""

    id_col: str

    @property
    def columns(self) -> tuple[str, ...]: ...

    def schema(self) -> pa.Schema: ...

    def content_identity(self) -> str: ...

    def statistics(self) -> TableStatistics: ...

    def scan(self, request: ScanRequest) -> pa.RecordBatchReader: ...

    def remote_source(self) -> Mapping[str, Any] | None: ...


def _check_request(schema: pa.Schema, request: ScanRequest) -> None:
    missing = [name for name in request.columns
               if name not in schema.names]
    if missing:
        raise CompileError(
            f"columns {missing} not in schema {tuple(schema.names)}")
    if request.limit is not None and request.limit < 0:
        raise ValueError("scan limit cannot be negative")
    if request.batch_rows <= 0:
        raise ValueError("scan batch_rows must be positive")


def _table_reader(table: pa.Table, batch_rows: int) -> pa.RecordBatchReader:
    return pa.RecordBatchReader.from_batches(
        table.schema, table.to_batches(max_chunksize=batch_rows)
    )


@dataclass(frozen=True)
class ArrowDatasetProvider:
    """Table provider backed by a PyArrow dataset."""

    dataset: Any
    id_col: str
    source: Mapping[str, Any] | None = field(
        default=None, compare=False, repr=False
    )

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.dataset.schema.names)

    def schema(self) -> pa.Schema:
        return self.dataset.schema

    def content_identity(self) -> str:
        files = tuple(sorted(getattr(self.dataset, "files", ()) or ()))
        value = repr((files, str(self.dataset.schema))).encode("utf-8")
        return "arrow:" + hashlib.sha256(value).hexdigest()

    def statistics(self) -> TableStatistics:
        return TableStatistics(row_count=int(self.dataset.count_rows()))

    def scan(self, request: ScanRequest) -> pa.RecordBatchReader:
        _check_request(self.dataset.schema, request)
        scanner = self.dataset.scanner(
            columns=list(request.columns),
            filter=request.filter,
            batch_size=request.batch_rows,
        )
        if request.limit is None:
            return scanner.to_reader()
        return _table_reader(scanner.head(request.limit), request.batch_rows)

    def remote_source(self) -> Mapping[str, Any] | None:
        """Return a source that another process can open."""
        return None if self.source is None else dict(self.source)

@dataclass(frozen=True)
class HuggingFaceProvider:
    """Table provider backed by a Hugging Face dataset."""

    dataset_name: str
    id_col: str
    split: str = "train"
    config: str = ""
    arrow_schema: pa.Schema = field(
        default_factory=lambda: pa.schema([])
    )

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.arrow_schema.names)

    def schema(self) -> pa.Schema:
        return self.arrow_schema

    def content_identity(self) -> str:
        value = repr((
            self.dataset_name,
            self.config,
            self.split,
            str(self.arrow_schema),
        )).encode("utf-8")
        return "hf:" + hashlib.sha256(value).hexdigest()

    def statistics(self) -> TableStatistics:
        from datasets import load_dataset_builder

        builder = load_dataset_builder(
            self.dataset_name, self.config or None
        )
        split = builder.info.splits.get(self.split)
        count = None if split is None else int(split.num_examples)
        byte_count = None if split is None else int(split.num_bytes)
        return TableStatistics(count, byte_count)

    def scan(self, request: ScanRequest) -> pa.RecordBatchReader:
        _check_request(self.arrow_schema, request)
        from datasets import load_dataset

        dataset = load_dataset(
            self.dataset_name,
            self.config or None,
            split=self.split,
        )
        table = dataset.data.table.select(request.columns)
        if request.filter is not None:
            raise ValueError(
                "Hugging Face provider does not support filter pushdown")
        if request.limit is not None:
            table = table.slice(0, request.limit)
        return _table_reader(table, request.batch_rows)

    def remote_source(self) -> Mapping[str, Any]:
        """Return a source that another process can open."""
        return {
            "type": "huggingface",
            "dataset": self.dataset_name,
            "id_col": self.id_col,
            "split": self.split,
            "config": self.config,
        }

@dataclass(frozen=True)
class MemoryTableProvider:
    """Table provider backed by one in memory Arrow table."""

    table: pa.Table
    id_col: str
    identity: str = "memory"

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.table.schema.names)

    def schema(self) -> pa.Schema:
        return self.table.schema

    def content_identity(self) -> str:
        return f"{self.identity}:{id(self.table)}"

    def statistics(self) -> TableStatistics:
        return TableStatistics(self.table.num_rows, self.table.nbytes)

    def scan(self, request: ScanRequest) -> pa.RecordBatchReader:
        _check_request(self.table.schema, request)
        if request.filter is not None:
            raise ValueError(
                "memory provider does not support filter pushdown")
        table = self.table.select(request.columns)
        if request.limit is not None:
            table = table.slice(0, request.limit)
        return _table_reader(table, request.batch_rows)

    def remote_source(self) -> None:
        """Return no remote source for process local memory."""
        return None

class DocumentProvider:
    """Factories for built in table providers."""

    @classmethod
    def from_dataset(cls, dataset, id_col: str) -> ArrowDatasetProvider:
        """Create a provider for an Arrow dataset."""
        import pyarrow.dataset as ds

        if not isinstance(dataset, ds.Dataset):
            raise TypeError(
                "dataset must be a pyarrow.dataset.Dataset, got "
                f"{type(dataset).__name__}")
        columns = tuple(dataset.schema.names)
        if id_col not in columns:
            raise CompileError(
                f"id column {id_col!r} not in dataset schema {columns}")
        return ArrowDatasetProvider(
            dataset,
            id_col,
            source=_remote_arrow_source(dataset, id_col),
        )

    @classmethod
    def from_parquet(
        cls, path: str | Sequence[str], id_col: str
    ) -> ArrowDatasetProvider:
        """Create a provider for a Parquet file or directory."""
        import pyarrow.dataset as ds

        dataset = ds.dataset(path, format="parquet")
        provider = cls.from_dataset(dataset, id_col=id_col)
        paths = (
            [os.fspath(path)]
            if isinstance(path, (str, os.PathLike))
            else [os.fspath(value) for value in path]
        )
        if paths and all(_is_remote_path(value) for value in paths):
            return ArrowDatasetProvider(
                dataset,
                id_col,
                source={
                    "type": "parquet",
                    "paths": paths,
                    "id_col": id_col,
                },
            )
        return provider

    @classmethod
    def from_ipc(cls, path: str, id_col: str) -> ArrowDatasetProvider:
        """Create a provider for an Arrow IPC file."""
        import pyarrow.dataset as ds

        return cls.from_dataset(ds.dataset(path, format="ipc"), id_col)

    @classmethod
    def from_hf(
        cls,
        dataset: str,
        id_col: str,
        split: str = "train",
        config: str = "",
    ) -> HuggingFaceProvider:
        """Create a provider for a Hugging Face dataset."""
        from datasets import load_dataset_builder

        builder = load_dataset_builder(dataset, config or None)
        features = builder.info.features
        schema = getattr(features, "arrow_schema", None)
        if schema is None:
            schema = pa.schema([
                pa.field(name, pa.string()) for name in features
            ])
        columns = tuple(schema.names)
        if id_col not in columns:
            raise CompileError(
                f"id column {id_col!r} not in dataset features {columns}")
        return HuggingFaceProvider(
            dataset_name=dataset,
            id_col=id_col,
            split=split,
            config=config,
            arrow_schema=schema,
        )

    @classmethod
    def from_table(
        cls, table: pa.Table, id_col: str, identity: str = "memory"
    ) -> MemoryTableProvider:
        """Create a provider for an in memory Arrow table."""
        if id_col not in table.schema.names:
            raise CompileError(
                f"id column {id_col!r} not in table schema "
                f"{tuple(table.schema.names)}")
        return MemoryTableProvider(table, id_col, identity)


@dataclass
class Catalog:
    providers: dict[str, TableProvider] = field(default_factory=dict)

    def register(self, name: str, provider: TableProvider) -> None:
        if name in self.providers:
            raise CompileError(f"provider {name!r} is already registered")
        self.providers[name] = provider

    def get(self, name: str) -> TableProvider:
        if name not in self.providers:
            raise CompileError(
                f"unknown provider {name!r}; registered: "
                f"{sorted(self.providers)}")
        return self.providers[name]

    def __contains__(self, name: str) -> bool:
        return name in self.providers


def _is_remote_path(path: str) -> bool:
    return urlparse(path).scheme.lower() in {
        "s3", "gs", "gcs", "abfs", "abfss", "hdfs"
    }


def _remote_arrow_source(dataset, id_col: str) -> Mapping[str, Any] | None:
    files = list(getattr(dataset, "files", ()) or ())
    if not files:
        return None
    file_format = type(getattr(dataset, "format", None)).__name__
    if file_format != "ParquetFileFormat":
        return None
    filesystem = type(getattr(dataset, "filesystem", None)).__name__
    prefixes = {
        "S3FileSystem": "s3://",
        "GcsFileSystem": "gs://",
        "AzureFileSystem": "abfs://",
        "HadoopFileSystem": "hdfs://",
    }
    prefix = prefixes.get(filesystem)
    if prefix is None:
        return None
    paths = [
        path if _is_remote_path(path) else prefix + path.lstrip("/")
        for path in files
    ]
    return {
        "type": "parquet",
        "paths": paths,
        "id_col": id_col,
    }


def _source_id_column(value: Mapping[str, Any]) -> str:
    id_col = str(value.get("id_col", ""))
    if not id_col:
        raise ValueError("a remote source needs an id_col")
    return id_col


def _read_remote_parquet(value: Mapping[str, Any]) -> TableProvider:
    paths = value.get("paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError("a Parquet source needs at least one path")
    if not all(isinstance(path, str) and path for path in paths):
        raise TypeError("Parquet source paths must be strings")
    source = paths[0] if len(paths) == 1 else paths
    return DocumentProvider.from_parquet(
        source, id_col=_source_id_column(value)
    )


def _read_remote_huggingface(value: Mapping[str, Any]) -> TableProvider:
    return DocumentProvider.from_hf(
        str(value["dataset"]),
        id_col=_source_id_column(value),
        split=str(value.get("split", "train")),
        config=str(value.get("config", "")),
    )


def built_in_source_readers() -> dict[str, Any]:
    """Return the built in remote table source readers."""
    return {
        "parquet": _read_remote_parquet,
        "huggingface": _read_remote_huggingface,
    }
