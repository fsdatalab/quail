"""PDF page inputs: page references, the page manifest, and the two readings.

A PDF table's rows are described once (PDFInput). A plan reads them
either as page images (prompt, prefetch: soft tokens and rendering)
or as extracted text (text: LiteParse, then the ordinary tokenizer).
"""

from .manifest import (
    PdfManifest,
    PdfReadError,
    check_source_unchanged,
    read_manifest,
    stat_sources,
)
from .prefetch import ImagePrefetcher, PdfiumPrefetcher, RenderedPage
from .text import PdfTextOptions, PdfTexts, row_texts, sample_page_texts
from .types import (
    ROW_MODES,
    PDFInput,
    PdfPageRef,
    PdfRowRef,
    PdfSource,
    RowMode,
    check_row_mode,
)

__all__ = [
    "ImagePrefetcher",
    "PDFInput",
    "PdfManifest",
    "PdfPageRef",
    "PdfReadError",
    "PdfRowRef",
    "PdfSource",
    "PdfTextOptions",
    "PdfTexts",
    "PdfiumPrefetcher",
    "ROW_MODES",
    "RenderedPage",
    "RowMode",
    "check_row_mode",
    "check_source_unchanged",
    "read_manifest",
    "row_texts",
    "sample_page_texts",
    "stat_sources",
]
