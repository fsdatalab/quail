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


def _table_digest(table: pa.Table) -> str:
    """A digest of a table's schema and values, for a content identity."""
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


class PDFProvider:
    """Query rows over local PDF files, with the caller's columns beside them.

    The rows come from a table in one of two layouts:

    * formed: one table row per file, named by ``path_col``.
      ``row_mode="page"`` repeats it once per page and adds a one based
      ``page_number``; ``row_mode="pdf"`` keeps one row per file.
    * listed: one table row per page, naming the file in ``path_col``
      and the one based page in the caller's own ``page_col``. Rows may
      list any pages of any files, in any order.

    Both add ``page_count`` and the model only ``document`` column
    holding the row's page references. A query cannot return or
    compare ``document``; a prompt reads it, and the model sees the
    row's pages rendered. ``ocr()`` gives the same rows read as text.

    The provider reads page counts and sizes once, on the first call
    that needs them, and never renders or parses a page.
    """

    document_column = "document"
    PAGE_NUMBER = "page_number"
    PAGE_COUNT = "page_count"

    def __init__(self, table: pa.Table, id_col: str, path_col: str, *,
                 row_mode: str | None = None, page_col: str | None = None):
        from quail.pdf import ROW_MODES

        if (row_mode is None) == (page_col is None):
            raise CompileError(
                "a PDF provider takes row_mode (rows formed from whole "
                "files) or page_col (rows listing pages), not both")
        if row_mode is not None and row_mode not in ROW_MODES:
            raise CompileError(
                f"row_mode must be one of {ROW_MODES}, got {row_mode!r}; it "
                f"decides whether a query row is one page or one PDF")
        names = tuple(table.schema.names)
        for name, what in ((id_col, "id"), (path_col, "path"),
                           (page_col, "page")):
            if name is not None and name not in names:
                raise CompileError(
                    f"{what} column {name!r} not in table schema {names}")
        if page_col is not None and not pa.types.is_integer(
                table.schema.field(page_col).type):
            raise CompileError(
                f"page column {page_col!r} must hold integer page numbers")
        self.table = table
        self.id_col = id_col
        self.path_col = path_col
        self.page_col = page_col
        self.row_mode = "page" if page_col is not None else row_mode
        taken = sorted(set(self.added_columns()) & set(names))
        if taken:
            raise CompileError(
                f"source columns {taken} clash with the columns a PDF "
                f"provider adds; rename them")
        if table.num_rows == 0:
            raise CompileError("a PDF provider needs at least one source row")
        self._pdf_input = None

    @property
    def layout(self) -> str:
        """"formed" from whole files, or "listed" pages."""
        return "listed" if self.page_col is not None else "formed"

    def added_columns(self) -> tuple[str, ...]:
        """The columns the provider adds after the table's, in order."""
        added = []
        if self.layout == "formed" and self.row_mode == "page":
            added.append(self.PAGE_NUMBER)
        return (*added, self.PAGE_COUNT, self.document_column)

    # ---- TableProvider -----------------------------------------------

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.schema().names)

    def schema(self) -> pa.Schema:
        fields = list(self.table.schema)
        for name in self.added_columns():
            if name == self.document_column:
                fields.append(pa.field(name, pa.int64(),
                                       metadata={MODEL_ONLY_KEY: b"pdf"}))
            else:
                fields.append(pa.field(name, pa.int32()))
        return pa.schema(fields)

    def content_identity(self) -> str:
        """Names the rows: the layout, the table's values, and the files."""
        value = repr((
            self.layout, self.row_mode, self.page_col,
            _table_digest(self.table),
            tuple((s.path, s.size, s.mtime_ns)
                  for s in self.pdf_input().sources),
        )).encode("utf-8")
        return "pdf:" + hashlib.sha256(value).hexdigest()

    def statistics(self) -> TableStatistics:
        return TableStatistics(row_count=len(self.pdf_input()))

    def scan(self, request: ScanRequest) -> pa.RecordBatchReader:
        _check_request(self.schema(), request)
        if request.filter is not None:
            raise ValueError("PDF provider does not support filter pushdown")
        table = self.row_table().select(request.columns)
        if request.limit is not None:
            table = table.slice(0, request.limit)
        return _table_reader(table, request.batch_rows)

    # ---- PDF specifics ------------------------------------------------

    def ocr(self, **options) -> OcrProvider:
        """The OCR operator over these rows: the same rows, read as text.

        Args:
            options: OcrOptions fields, such as ``language``.
        """
        from quail.pdf import OcrOptions

        return OcrProvider(self, OcrOptions(**options))

    def paths(self) -> list[str]:
        """The distinct files, in first appearance order."""
        return list(dict.fromkeys(
            str(p) for p in self.table.column(self.path_col).to_pylist()))

    def pdf_input(self):
        """The rows' page references; read from the files on first use.

        Raises:
            CompileError: A listed page number is null or outside its file.
        """
        if self._pdf_input is None:
            from quail.pdf import PDFInput, read_manifest

            manifest = read_manifest(self.paths())
            if self.layout == "formed":
                self._pdf_input = PDFInput.formed(
                    manifest.sources, manifest.pages, self.row_mode)
            else:
                source_index = {path: i for i, path in enumerate(self.paths())}
                listed = [
                    (source_index[str(path)], number)
                    for path, number in zip(
                        self.table.column(self.path_col).to_pylist(),
                        self.table.column(self.page_col).to_pylist())]
                try:
                    self._pdf_input = PDFInput.listed(
                        manifest.sources, manifest.pages, listed)
                except ValueError as error:
                    raise CompileError(str(error)) from error
        return self._pdf_input

    def table_rows(self) -> tuple[int, ...]:
        """The table row each query row carries its columns from."""
        pdf_input = self.pdf_input()
        if self.layout == "formed" and self.row_mode == "page":
            return pdf_input.source_rows
        return tuple(range(len(pdf_input)))

    def row_table(self) -> pa.Table:
        """The table's columns per query row, then the added columns."""
        pdf_input = self.pdf_input()
        counts = pdf_input.page_counts
        table = self.table.take(pa.array(self.table_rows(), pa.int64()))
        for name in self.added_columns():
            if name == self.PAGE_NUMBER:
                values = pa.array(
                    [pdf_input.row_pages(row)[0].page_number
                     for row in range(len(pdf_input))], pa.int32())
            elif name == self.PAGE_COUNT:
                values = pa.array(
                    [counts[source] for source in pdf_input.source_rows],
                    pa.int32())
            else:
                values = pa.array(range(len(pdf_input)), pa.int64())
            table = table.append_column(self.schema().field(name), values)
        return table


class OcrProvider:
    """The OCR operator: a PDF provider's rows with each row's page text.

    ``document`` is a string column holding the text of the row's
    pages, read with LiteParse; every other column is the PDF
    provider's. The first scan that asks for ``document`` parses the
    files in a worker pool and yields rows file by file, so the
    tokenizer works on the first file while the rest parse. The texts
    are kept for later scans. From here on the rows go through the
    ordinary tokenizer, token store, and text scan, on any backend.
    """

    def __init__(self, pdf: PDFProvider, options):
        self.pdf = pdf
        self.options = options
        self.id_col = pdf.id_col
        self._rows = None
        self._metrics = None

    @property
    def document_column(self) -> str:
        return self.pdf.document_column

    # ---- TableProvider -----------------------------------------------

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.schema().names)

    def schema(self) -> pa.Schema:
        fields = [
            pa.field(f.name, pa.string()) if f.name == self.document_column
            else f
            for f in self.pdf.schema()]
        return pa.schema(fields)

    def content_identity(self) -> str:
        return f"{self.pdf.content_identity()}|ocr:{self.options.identity()}"

    def statistics(self) -> TableStatistics:
        return self.pdf.statistics()

    def scan(self, request: ScanRequest) -> pa.RecordBatchReader:
        _check_request(self.schema(), request)
        if request.filter is not None:
            raise ValueError("PDF provider does not support filter pushdown")
        table = self.pdf.row_table()
        if self._rows is not None:
            table = self._with_texts(table, self._rows)
        if self._rows is not None or self.document_column not in request.columns:
            table = table.select(request.columns)
            if request.limit is not None:
                table = table.slice(0, request.limit)
            return _table_reader(table, request.batch_rows)
        return pa.RecordBatchReader.from_batches(
            pa.schema([self.schema().field(name) for name in request.columns]),
            self._streamed(table, request))

    def _with_texts(self, table: pa.Table, texts) -> pa.Table:
        return table.set_column(
            table.schema.get_field_index(self.document_column),
            self.schema().field(self.document_column),
            pa.array(texts, pa.string()))

    def _streamed(self, table: pa.Table, request: ScanRequest):
        """Yield the requested columns file by file as the pool parses them.

        Every row is read even past the limit, so the texts are whole
        when they are kept for later scans.
        """
        import time

        from quail.pdf import row_texts

        started = time.perf_counter()
        pdf_input = self.pdf.pdf_input()
        texts = []
        remaining = len(table) if request.limit is None else request.limit
        for start, chunk in row_texts(pdf_input, self.options):
            texts.extend(chunk)
            piece = self._with_texts(table.slice(start, len(chunk)), chunk)
            piece = piece.select(request.columns).slice(0, max(remaining, 0))
            remaining -= len(piece)
            yield from piece.to_batches(max_chunksize=request.batch_rows)
        self._rows = tuple(texts)
        self._metrics = {
            "rows": len(texts),
            "pages": pdf_input.page_count,
            "empty_rows": sum(not text.strip() for text in texts),
            "chars": sum(len(text) for text in texts),
            "extract_s": round(time.perf_counter() - started, 4),
            "processes": self.options.processes,
        }

    # ---- OCR specifics -----------------------------------------------

    def pdf_input(self):
        """The rows' page references, as the PDF provider reads them."""
        return self.pdf.pdf_input()

    def metrics(self) -> dict | None:
        """The extraction's counters, once a scan has read the text."""
        return None if self._metrics is None else dict(self._metrics)

    def estimate_token_lengths(self, count_tokens) -> list[int]:
        """Estimate each row's tokens from a sample of pages.

        Reads the first pages of the corpus, counts their tokens with
        ``count_tokens(texts)``, and scales every row by its page count.
        A whole-file row of many pages estimates as that many pages.
        """
        from quail.pdf import sample_page_texts

        pdf_input = self.pdf.pdf_input()
        sample = sample_page_texts(pdf_input, self.options)
        per_page = (sum(count_tokens(sample)) / len(sample)) if sample else 0.0
        return [max(1, round(len(row.page_ids) * per_page))
                for row in pdf_input.rows]


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
        return PDFProvider(sources, id_col, path_col, row_mode=row_mode)

    @classmethod
    def from_pdf_pages(cls, pages: pa.Table, id_col: str, path_col: str,
                       page_col: str) -> PDFProvider:
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
        return PDFProvider(pages, id_col, path_col, page_col=page_col)


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
