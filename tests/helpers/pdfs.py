"""Small PDFs written in a test, one line of Helvetica text per page."""


def _escaped(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def pdf(*pages: str) -> bytes:
    """Return a PDF whose pages each show one line of text, in order.

    Args:
        *pages: Each page's text, in Latin-1.
    """
    font = 3 + 2 * len(pages)
    page_ids = [3 + 2 * index for index in range(len(pages))]
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{page} 0 R' for page in page_ids)}] "
        f"/Count {len(pages)} >>".encode(),
    ]
    for page, text in zip(page_ids, pages, strict=True):
        content = (f"BT /F1 12 Tf 72 720 Td ({_escaped(text)}) Tj ET"
                   .encode("latin-1"))
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {page + 1} 0 R "
            f"/Resources << /Font << /F1 {font} 0 R >> >> >>".encode())
        objects.append(f"<< /Length {len(content)} >>\nstream\n".encode() + content
                       + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
                   b"/Encoding /WinAnsiEncoding >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()
    return bytes(out)
