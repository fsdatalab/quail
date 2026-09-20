"""The PDF rows of one scan as the session prepares them for a query."""

from __future__ import annotations

import pyarrow as pa

from quail.pdf import PDFInput
from quail.planner.plan import PdfDocuments


class PdfScanInput:
    """One scan's PDF rows, their planned lengths, and stored value columns.

    The text counterpart is ScanInput; a query treats both the same
    way: lengths for planning, column() for value tables, and
    physical_input() for the request binding. The lengths are
    PagePrompts.lengths for the session's image budget.
    """

    def __init__(self, pdf_input: PDFInput, lengths: tuple[int, ...],
                 columns: dict):
        if len(lengths) != len(pdf_input):
            raise ValueError("one planned length per PDF row")
        self.pdf_input = pdf_input
        self._lengths = lengths
        self._columns = dict(columns)

    @property
    def lengths(self) -> tuple[int, ...]:
        return self._lengths

    @property
    def projected_columns(self) -> tuple[str, ...]:
        return tuple(self._columns)

    def column(self, name: str) -> pa.ChunkedArray:
        return self._columns[name].values

    def physical_input(self) -> PDFInput:
        return self.pdf_input

    def documents(self) -> PdfDocuments:
        """What the planner needs beyond the per-row lengths."""
        return PdfDocuments(
            row_mode=self.pdf_input.row_mode,
            n_pages=self.pdf_input.page_count,
            visual_tokens=self.pdf_input.visual_tokens,
            pages_per_row_max=self.pdf_input.pages_per_row_max)
