"""Immutable descriptions of PDF sources, pages, and query rows.

Nothing here holds pixels, open files, process pools, or device
memory. A PDFInput is request data: it says which pages each query
row shows the model and where those pages come from. The executor
renders from it and owns every buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RowMode = Literal["page", "pdf"]
ROW_MODES: tuple[RowMode, ...] = ("page", "pdf")


def check_row_mode(row_mode: str) -> RowMode:
    """Return row_mode when it is a known mode; raise otherwise."""
    if row_mode not in ROW_MODES:
        raise ValueError(
            f"row_mode must be one of {ROW_MODES}, got {row_mode!r}; it "
            f"decides whether a query row is one page or one PDF")
    return row_mode


@dataclass(frozen=True)
class PdfSource:
    """One PDF file as it was when the manifest was read."""

    path: str
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class PdfPageRef:
    """One page of one source, with its displayed size in points."""

    source_index: int
    page_index: int      # zero based, as PDFium counts
    width_points: float
    height_points: float

    @property
    def page_number(self) -> int:
        """The one based page number a user sees."""
        return self.page_index + 1


@dataclass(frozen=True)
class PdfRowRef:
    """The pages one query row shows the model, in prompt order."""

    page_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.page_ids:
            raise ValueError("a PDF query row needs at least one page")


@dataclass(frozen=True)
class PDFInput:
    """The PDF pages one scan binds, row by row.

    Attributes:
        sources: Every source file, indexed by PdfPageRef.source_index.
        pages: The page manifest, indexed by PdfRowRef.page_ids.
        rows: One entry per query row, in document id order.
        row_mode: How rows were formed from sources.
        visual_tokens: The soft token budget every page is rendered for.
    """

    sources: tuple[PdfSource, ...]
    pages: tuple[PdfPageRef, ...]
    rows: tuple[PdfRowRef, ...]
    row_mode: RowMode
    visual_tokens: int

    def __post_init__(self) -> None:
        check_row_mode(self.row_mode)
        if not self.sources:
            raise ValueError("a PDF input needs at least one source")
        if self.visual_tokens <= 0:
            raise ValueError("a PDF input needs a positive visual token budget")
        for page in self.pages:
            if not 0 <= page.source_index < len(self.sources):
                raise ValueError(
                    f"page refers to source {page.source_index}, but the "
                    f"input has {len(self.sources)} sources")
        for row_index, row in enumerate(self.rows):
            owners = set()
            for page_id in row.page_ids:
                if not 0 <= page_id < len(self.pages):
                    raise ValueError(
                        f"row {row_index} refers to page {page_id}, but the "
                        f"input has {len(self.pages)} pages")
                owners.add(self.pages[page_id].source_index)
            if len(owners) != 1:
                raise ValueError(
                    f"row {row_index} mixes pages of sources {sorted(owners)}; "
                    f"a query row shows one PDF")

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def page_count(self) -> int:
        """Pages referenced by the rows, counting each reference."""
        return sum(len(row.page_ids) for row in self.rows)

    @property
    def pages_per_row_max(self) -> int:
        return max((len(row.page_ids) for row in self.rows), default=0)

    def row_pages(self, row_index: int) -> tuple[PdfPageRef, ...]:
        """The page manifest entries one row shows, in prompt order."""
        return tuple(self.pages[i] for i in self.rows[row_index].page_ids)


def rows_for_mode(pages: tuple[PdfPageRef, ...], n_sources: int,
                  row_mode: RowMode) -> tuple[PdfRowRef, ...]:
    """Form query rows from a page manifest.

    Page mode gives one row per manifest entry in manifest order. PDF
    mode gives one row per source, listing that source's pages in
    page order.

    Raises:
        ValueError: PDF mode and a source has no pages.
    """
    check_row_mode(row_mode)
    if row_mode == "page":
        return tuple(PdfRowRef((i,)) for i in range(len(pages)))
    by_source: dict[int, list[int]] = {i: [] for i in range(n_sources)}
    for page_id, page in enumerate(pages):
        by_source[page.source_index].append(page_id)
    for source_index, ids in by_source.items():
        if not ids:
            raise ValueError(f"source {source_index} has no pages")
    return tuple(PdfRowRef(tuple(sorted(
        ids, key=lambda page_id: pages[page_id].page_index)))
        for ids in by_source.values())
