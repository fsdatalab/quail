"""PDF page inputs: page references, the page manifest, and the two readings.

A PDF table's rows are described once (PDFInput). A query reads them
as page images (prompt, prefetch: soft tokens and rendering) or, when
the table is registered through the OCR operator, as text (ocr:
LiteParse, then the ordinary tokenizer).
"""

from .manifest import (
    PdfManifest,
    PdfReadError,
    check_source_unchanged,
    read_manifest,
    stat_sources,
)
from .ocr import OcrOptions, row_texts, sample_page_texts
from .prefetch import ImagePrefetcher, PdfiumPrefetcher, RenderedPage
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
    "OcrOptions",
    "PDFInput",
    "PdfManifest",
    "PdfPageRef",
    "PdfReadError",
    "PdfRowRef",
    "PdfSource",
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
