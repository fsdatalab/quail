"""Write small PDF files for tests."""


def write_pdf(path, pages) -> None:
    """Write a PDF with one Helvetica line per page; '' leaves a page blank.

    Args:
        path: The file to write.
        pages: Each page's text.
    """
    count = len(pages)
    font = 3 + 2 * count
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids ["
        + b" ".join(b"%d 0 R" % (3 + 2 * i) for i in range(count))
        + b"] /Count %d >>" % count,
    ]
    for i, text in enumerate(pages):
        escaped = (text.replace("\\", "\\\\").replace("(", "\\(")
                   .replace(")", "\\)").encode("latin-1"))
        stream = (b"BT /F1 12 Tf 72 720 Td (" + escaped + b") Tj ET"
                  if text else b"")
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
            % (font, 4 + 2 * i))
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream)
                       + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += (b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objects) + 1, xref))
    with open(path, "wb") as file:
        file.write(bytes(out))
