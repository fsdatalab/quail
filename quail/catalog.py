"""Table providers and the session catalog."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

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


# Arrow field metadata marking a column the model reads but a query
# cannot return or compare: a PDF provider's page references.
MODEL_ONLY_KEY = b"quail.model_only"


def model_only_columns(provider: TableProvider) -> frozenset[str]:
    """Names of the provider's columns that only a model call may read."""
    return frozenset(
        f.name for f in provider.schema()
        if f.metadata and MODEL_ONLY_KEY in f.metadata)


class PDFProvider:
    """Query rows formed from a table of local PDF paths.

    ``row_mode="page"`` makes one row per page and adds one based
    ``page_number`` and ``page_count`` columns beside the repeated
    source columns. ``row_mode="pdf"`` keeps one row per source and
    adds ``page_count``. Both expose a model only ``document`` column
    holding the row's page references; it can only be a document
    argument of an AI.FILTER or AI.JOIN prompt, and a join anchors on
    the rows whose pages it reads.

    The provider reads page counts and sizes once, on the first call
    that needs them, and never renders a page.
    """

    document_column = "document"
    PAGE_NUMBER = "page_number"
    PAGE_COUNT = "page_count"

    def __init__(self, sources: pa.Table, id_col: str, path_col: str,
                 row_mode: str):
        from quail.pdf import ROW_MODES

        if row_mode not in ROW_MODES:
            raise CompileError(
                f"row_mode must be one of {ROW_MODES}, got {row_mode!r}; it "
                f"decides whether a query row is one page or one PDF")
        names = tuple(sources.schema.names)
        for name, what in ((id_col, "id"), (path_col, "path")):
            if name not in names:
                raise CompileError(
                    f"{what} column {name!r} not in table schema {names}")
        taken = sorted(self._added_columns(row_mode) & set(names))
        if taken:
            raise CompileError(
                f"source columns {taken} clash with the columns a PDF "
                f"provider adds; rename them")
        if sources.num_rows == 0:
            raise CompileError("a PDF provider needs at least one source row")
        self.sources = sources
        self.id_col = id_col
        self.path_col = path_col
        self.row_mode = row_mode
        self._manifest = None

    @classmethod
    def _added_columns(cls, row_mode: str) -> set[str]:
        """The columns the provider adds to a row, which a source cannot hold."""
        added = {cls.document_column, cls.PAGE_COUNT}
        if row_mode == "page":
            added.add(cls.PAGE_NUMBER)
        return added

    # ---- TableProvider -----------------------------------------------

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.schema().names)

    def schema(self) -> pa.Schema:
        fields = list(self.sources.schema)
        if self.row_mode == "page":
            fields.append(pa.field(self.PAGE_NUMBER, pa.int32()))
        fields.append(pa.field(self.PAGE_COUNT, pa.int32()))
        fields.append(pa.field(
            self.document_column, pa.int64(),
            metadata={MODEL_ONLY_KEY: b"pdf"}))
        return pa.schema(fields)

    def content_identity(self) -> str:
        sources = self.manifest().sources
        value = repr((
            self.row_mode,
            str(self.sources.schema),
            tuple((s.path, s.size, s.mtime_ns) for s in sources),
        )).encode("utf-8")
        return "pdf:" + hashlib.sha256(value).hexdigest()

    def statistics(self) -> TableStatistics:
        return TableStatistics(row_count=len(self._rows()))

    def scan(self, request: ScanRequest) -> pa.RecordBatchReader:
        _check_request(self.schema(), request)
        if request.filter is not None:
            raise ValueError("PDF provider does not support filter pushdown")
        table = self._row_table().select(request.columns)
        if request.limit is not None:
            table = table.slice(0, request.limit)
        return _table_reader(table, request.batch_rows)

    # ---- PDF specifics ------------------------------------------------

    def paths(self) -> list[str]:
        return [str(p) for p in self.sources.column(self.path_col).to_pylist()]

    def manifest(self):
        """The cached page manifest; read from the files on first use."""
        if self._manifest is None:
            from quail.pdf import read_manifest

            self._manifest = read_manifest(self.paths())
        return self._manifest

    def pdf_input(self, visual_tokens: int):
        """The immutable input the executor renders this table from."""
        from quail.pdf import PDFInput

        manifest = self.manifest()
        return PDFInput(manifest.sources, manifest.pages, self._rows(),
                        self.row_mode, visual_tokens)

    def _rows(self):
        from quail.pdf import rows_for_mode

        manifest = self.manifest()
        return rows_for_mode(manifest.pages, len(manifest.sources),
                             self.row_mode)

    def _row_table(self) -> pa.Table:
        """Source columns per row, the page columns, and row positions."""
        manifest = self.manifest()
        counts = manifest.page_counts
        rows = self._rows()
        if self.row_mode == "page":
            source_rows = [manifest.pages[row.page_ids[0]].source_index
                           for row in rows]
            page_numbers = [manifest.pages[row.page_ids[0]].page_number
                            for row in rows]
        else:
            source_rows = list(range(len(counts)))
        table = self.sources.take(pa.array(source_rows, pa.int64()))
        if self.row_mode == "page":
            table = table.append_column(
                self.PAGE_NUMBER, pa.array(page_numbers, pa.int32()))
        table = table.append_column(
            self.PAGE_COUNT,
            pa.array([counts[i] for i in source_rows], pa.int32()))
        return table.append_column(
            self.schema().field(self.document_column),
            pa.array(range(len(rows)), pa.int64()))


class PDFPagesProvider(PDFProvider):
    """Query rows listed by the caller, one page each.

    The table names each row's PDF path and one based page number and
    carries its other columns along, the row id among them. Rows may
    list any pages of the files, in any order, and need not cover a
    file. The provider adds ``page_count`` and the model only
    ``document`` column; the caller's page column keeps its name.
    """

    def __init__(self, pages: pa.Table, id_col: str, path_col: str,
                 page_col: str):
        if page_col not in pages.schema.names:
            raise CompileError(
                f"page column {page_col!r} not in table schema "
                f"{tuple(pages.schema.names)}")
        if not pa.types.is_integer(pages.schema.field(page_col).type):
            raise CompileError(
                f"page column {page_col!r} must hold integer page numbers")
        super().__init__(pages, id_col, path_col, "page")
        self.page_col = page_col
        self.pages = pages
        # the sources are the distinct files, in first appearance order
        self.sources = pa.table({
            path_col: pa.array(list(dict.fromkeys(self.paths_of_rows())),
                               pa.string())})

    @classmethod
    def _added_columns(cls, row_mode: str) -> set[str]:
        return {cls.document_column, cls.PAGE_COUNT}

    def paths_of_rows(self) -> list[str]:
        """Each listed row's PDF path."""
        return [str(p) for p in self.pages.column(self.path_col).to_pylist()]

    def schema(self) -> pa.Schema:
        fields = list(self.pages.schema)
        fields.append(pa.field(self.PAGE_COUNT, pa.int32()))
        fields.append(pa.field(
            self.document_column, pa.int64(),
            metadata={MODEL_ONLY_KEY: b"pdf"}))
        return pa.schema(fields)

    def content_identity(self) -> str:
        listed = list(zip(self.paths_of_rows(),
                          self.pages.column(self.page_col).to_pylist()))
        value = repr((super().content_identity(), listed)).encode("utf-8")
        return "pdf:" + hashlib.sha256(value).hexdigest()

    def _rows(self):
        """One row per listed page, in table order.

        Raises:
            CompileError: A listed page number is null or outside its file.
        """
        from quail.pdf import PdfRowRef

        manifest = self.manifest()
        # the manifest lists sources in paths() order
        source_index = {path: i for i, path in enumerate(self.paths())}
        page_id = {(page.source_index, page.page_number): i
                   for i, page in enumerate(manifest.pages)}
        counts = manifest.page_counts
        rows = []
        for path, number in zip(self.paths_of_rows(),
                                self.pages.column(self.page_col).to_pylist()):
            source = source_index[path]
            if number is None or not 1 <= number <= counts[source]:
                raise CompileError(
                    f"page {number!r} of {path!r} does not exist; the file "
                    f"has {counts[source]} pages")
            rows.append(PdfRowRef((page_id[source, number],)))
        return tuple(rows)

    def _row_table(self) -> pa.Table:
        manifest = self.manifest()
        rows = self._rows()
        counts = [manifest.page_counts[manifest.pages[row.page_ids[0]].source_index]
                  for row in rows]
        table = self.pages.append_column(
            self.PAGE_COUNT, pa.array(counts, pa.int32()))
        return table.append_column(
            self.schema().field(self.document_column),
            pa.array(range(len(rows)), pa.int64()))


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
        return ArrowDatasetProvider(dataset, id_col)

    @classmethod
    def from_parquet(
        cls, path: str | Sequence[str], id_col: str
    ) -> ArrowDatasetProvider:
        """Create a provider for a Parquet file or directory."""
        import pyarrow.dataset as ds

        dataset = ds.dataset(path, format="parquet")
        return cls.from_dataset(dataset, id_col=id_col)

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

    @classmethod
    def from_pdfs(cls, sources: pa.Table, id_col: str, path_col: str,
                  row_mode: str) -> PDFProvider:
        """Create a provider over local PDF files, one source row per PDF.

        Args:
            sources: One row per PDF with its id, its local path, and
                any value columns to carry along.
            id_col: The source id column; it repeats in page mode.
            path_col: The column of local file paths.
            row_mode: "page" for one query row per page, "pdf" for one
                per PDF. Required: it changes what a row means.
        """
        return PDFProvider(sources, id_col, path_col, row_mode)

    @classmethod
    def from_pdf_pages(cls, pages: pa.Table, id_col: str, path_col: str,
                       page_col: str) -> PDFPagesProvider:
        """Create a provider over listed PDF pages, one query row per row.

        Use this when the rows already exist as a table with their own
        ids, such as a page table with per page labels. The rows may
        list any pages of any files, in any order.

        Args:
            pages: One row per page with its id, the PDF's local path,
                its one based page number, and any value columns.
            id_col: The row id column.
            path_col: The column of local file paths.
            page_col: The integer column of one based page numbers.
        """
        return PDFPagesProvider(pages, id_col, path_col, page_col)


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
