"""Small PDFs for tests: blank pages of given sizes, or pages with text.

PDFium writes both. Text pages draw each line in Helvetica, one under
another from the top left, so PDFium's text layer and LiteParse read
the same words back.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence

LETTER = (612, 792)


def make_pdf(path, sizes: Sequence[tuple[float, float]]) -> str:
    """Write a PDF with one blank page per (width, height) in points."""
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument.new()
    for width, height in sizes:
        document.new_page(width, height)
    document.save(str(path))
    document.close()
    return str(path)


def make_text_pdf(path, pages: Sequence[str | Sequence[str]],
                  size: tuple[float, float] = LETTER, font_size: int = 12,
                  line_height: int = 16) -> str:
    """Write a PDF with one page per entry, each entry's lines drawn in order.

    An entry is a string (split on newlines) or a sequence of lines. An
    empty entry gives a page with no text at all.
    """
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c

    width, height = size
    document = pdfium.PdfDocument.new()
    font = pdfium_c.FPDFText_LoadStandardFont(document.raw, b"Helvetica")
    for entry in pages:
        lines = entry.split("\n") if isinstance(entry, str) else list(entry)
        page = document.new_page(width, height)
        for row, text in enumerate(line for line in lines if line):
            block = pdfium_c.FPDFPageObj_CreateTextObj(
                document.raw, font, font_size)
            encoded = text.encode("utf-16-le") + b"\x00\x00"
            buffer = ctypes.create_string_buffer(encoded, len(encoded))
            pdfium_c.FPDFText_SetText(
                block, ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ushort)))
            pdfium_c.FPDFPageObj_Transform(
                block, 1, 0, 0, 1, 72, height - 72 - line_height * row)
            pdfium_c.FPDFPage_InsertObject(page.raw, block)
        pdfium_c.FPDFPage_GenerateContent(page.raw)
    document.save(str(path))
    document.close()
    return str(path)
