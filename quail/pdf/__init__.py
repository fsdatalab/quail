"""PDF page inputs: immutable page references and the page manifest."""

from .manifest import (
    PdfManifest,
    PdfReadError,
    check_source_unchanged,
    read_manifest,
    stat_sources,
)
from .types import (
    ROW_MODES,
    PDFInput,
    PdfPageRef,
    PdfRowRef,
    PdfSource,
    RowMode,
    check_row_mode,
    rows_for_mode,
)

__all__ = [
    "PDFInput",
    "PdfManifest",
    "PdfPageRef",
    "PdfReadError",
    "PdfRowRef",
    "PdfSource",
    "ROW_MODES",
    "RowMode",
    "check_row_mode",
    "check_source_unchanged",
    "read_manifest",
    "rows_for_mode",
    "stat_sources",
]
