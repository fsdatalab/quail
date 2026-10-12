"""PDF tables: quail.read_pdfs and reading their pages.

read_pdfs registers PDF files as a table with one row per page, or one
row per file. Its ``document`` column refers to the row's pages; only
AI.PARSE reads it. Pages are opened, rendered, and their stored text
read with pypdfium2, in the process that runs the parser.
"""

from __future__ import annotations

import glob as globbing
import hashlib
import os
from collections.abc import Sequence
from dataclasses import dataclass

import pyarrow as pa

from quail.catalog import ScanRequest, TableStatistics, _check_request, _table_reader
from quail.logical import CompileError

# the column that refers to a row's pages
DOCUMENT_COLUMN = "document"
DOCUMENT_TYPE = pa.struct([pa.field("path", pa.string()),
                           pa.field("page", pa.int32())])
# rendering: 400 dpi, then shrunk to fit the longest edge, as
# LightOnOCR's card renders pages
RENDER_DPI = 400
# the text between two pages of one row
PAGE_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class PdfFile:
    """One registered file: its path, size, change time, and page count."""

    path: str
    size: int
    mtime_ns: int
    pages: int


def _stat(path: str) -> tuple[int, int]:
    stat = os.stat(path)
    return stat.st_size, stat.st_mtime_ns


def _open(path: str):
    import pypdfium2 as pdfium

    try:
        return pdfium.PdfDocument(path)
    except pdfium.PdfiumError as error:
        raise ValueError(f"{path}: PDFium cannot open it: {error}") from error


def read_pdfs(paths: str | Sequence[str], *,
              split_pages: bool = True) -> "PdfPagesProvider":
    """Register PDF files as a table, one row per page or per file.

    The table's columns are ``path``, ``page`` (from 1; only with
    split_pages), ``page_count``, and ``document``. ``document`` refers
    to the row's pages; read it with AI.PARSE. Rows are ordered by
    path, then page.

    Args:
        paths: A glob, or a list of paths and globs.
        split_pages: One row per page; False gives one row per file
            whose document covers every page in order.

    Returns:
        A table provider for Session.register.

    Raises:
        ValueError: No file matches, or PDFium cannot open one.
    """
    patterns = [paths] if isinstance(paths, str) else list(paths)
    found = set()
    for pattern in patterns:
        matched = globbing.glob(pattern) if globbing.has_magic(pattern) else [pattern]
        found.update(os.path.abspath(path) for path in matched)
    if not found:
        raise ValueError(f"no PDF file matches {paths!r}")
    files = []
    for path in sorted(found):
        size, mtime_ns = _stat(path)
        pdf = _open(path)
        try:
            files.append(PdfFile(path, size, mtime_ns, len(pdf)))
        finally:
            pdf.close()
    return PdfPagesProvider(tuple(files), split_pages)


@dataclass(frozen=True)
class PdfPagesProvider:
    """The table read_pdfs registers.

    Args:
        files: The files, sorted by path.
        split_pages: One row per page, or one per file.
    """

    files: tuple
    split_pages: bool = True
    id_col: str = DOCUMENT_COLUMN

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.schema().names)

    def schema(self) -> pa.Schema:
        fields = [pa.field("path", pa.string(), nullable=False)]
        if self.split_pages:
            fields.append(pa.field("page", pa.int32(), nullable=False))
        fields += [pa.field("page_count", pa.int32(), nullable=False),
                   pa.field(DOCUMENT_COLUMN, DOCUMENT_TYPE, nullable=False)]
        return pa.schema(fields)

    def content_identity(self) -> str:
        value = repr(([(f.path, f.size, f.mtime_ns) for f in self.files],
                      self.split_pages)).encode("utf-8")
        return "pdf:" + hashlib.sha256(value).hexdigest()

    def statistics(self) -> TableStatistics:
        rows = (sum(f.pages for f in self.files) if self.split_pages
                else len(self.files))
        return TableStatistics(row_count=rows,
                               byte_count=sum(f.size for f in self.files))

    def table(self) -> pa.Table:
        """Every row, in path then page order."""
        paths, pages, counts = [], [], []
        for f in self.files:
            for page in (range(1, f.pages + 1) if self.split_pages else (None,)):
                paths.append(f.path)
                pages.append(page)
                counts.append(f.pages)
        columns = {"path": pa.array(paths, pa.string())}
        if self.split_pages:
            columns["page"] = pa.array(pages, pa.int32())
        columns["page_count"] = pa.array(counts, pa.int32())
        columns[DOCUMENT_COLUMN] = pa.StructArray.from_arrays(
            [columns["path"], pa.array(pages, pa.int32())],
            fields=list(DOCUMENT_TYPE))
        return pa.table(columns, schema=self.schema())

    def scan(self, request: ScanRequest) -> pa.RecordBatchReader:
        schema = self.schema()
        _check_request(schema, request)
        if request.filter is not None:
            raise ValueError("the PDF table does not support filter pushdown")
        table = self.table().select(list(request.columns))
        if request.limit is not None:
            table = table.slice(0, request.limit)
        return _table_reader(table, request.batch_rows)

    def pages(self) -> "PageReferences":
        """Each row's pages, in row order."""
        return PageReferences(self.files, self.split_pages)


class PageReferences(Sequence):
    """Each row's file and pages; what the executor reads a row from.

    Args:
        files: The registered files.
        split_pages: One row per page, or one per file.
    """

    def __init__(self, files, split_pages: bool):
        self.files = tuple(files)
        self.split_pages = split_pages
        self._rows = [(index, page) for index, f in enumerate(self.files)
                      for page in (range(1, f.pages + 1) if split_pages else (None,))]

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, row):
        """The row's file and its pages (from 1), as (PdfFile, [pages])."""
        index, page = self._rows[row]
        f = self.files[index]
        return f, ([page] if page is not None else list(range(1, f.pages + 1)))

    def page_count(self, row) -> int:
        return len(self[row][1])


def check_unchanged(f: PdfFile) -> None:
    """Raise when a file changed after it was registered.

    Raises:
        CompileError: Its size or modification time differs.
    """
    if _stat(f.path) != (f.size, f.mtime_ns):
        raise CompileError(
            f"{f.path} changed after it was registered; register it again")


def stored_text(path: str, pages) -> list[str]:
    """Each page's stored text, as PDFium reads it; '' for a page with none.

    Args:
        path: The file.
        pages: Page numbers, from 1.
    """
    pdf = _open(path)
    try:
        texts = []
        for number in pages:
            page = pdf[number - 1]
            textpage = page.get_textpage()
            texts.append(textpage.get_text_bounded())
            textpage.close()
            page.close()
        return texts
    finally:
        pdf.close()


def render_png(path: str, page: int, longest_edge: int) -> bytes:
    """Render one page at RENDER_DPI, shrunk to fit longest_edge, as PNG bytes.

    Args:
        path: The file.
        page: The page number, from 1.
        longest_edge: The largest width or height, in pixels.
    """
    import io

    pdf = _open(path)
    try:
        handle = pdf[page - 1]
        picture = handle.render(scale=RENDER_DPI / 72).to_pil().convert("RGB")
        handle.close()
    finally:
        pdf.close()
    picture.thumbnail((longest_edge, longest_edge))
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return buffer.getvalue()
