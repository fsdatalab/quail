"""Read PDF page counts and sizes without rendering anything.

PDFium opens each file once, reads every page's displayed size (its
/Rotate applied, as PDFium reports it), and closes it. The result is
a page manifest the provider caches and the planner prices.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

from quail.pdf.types import PdfPageRef, PdfSource, page_counts


class PdfReadError(RuntimeError):
    """A source PDF could not be opened or read."""


@dataclass(frozen=True)
class PdfManifest:
    """Every source and every page of a PDF provider, in source order."""

    sources: tuple[PdfSource, ...]
    pages: tuple[PdfPageRef, ...]

    @property
    def page_counts(self) -> tuple[int, ...]:
        return page_counts(len(self.sources), self.pages)


def stat_sources(paths: Sequence[str]) -> tuple[PdfSource, ...]:
    """Record each file's resolved path, size, and modification time.

    Raises:
        PdfReadError: A path is missing or not a regular file.
    """
    sources = []
    for path in paths:
        resolved = os.path.realpath(os.path.expanduser(str(path)))
        try:
            info = os.stat(resolved)
        except OSError as error:
            raise PdfReadError(f"cannot read PDF {path!r}: {error}") from error
        if not os.path.isfile(resolved):
            raise PdfReadError(f"PDF source {path!r} is not a regular file")
        sources.append(PdfSource(resolved, info.st_size, info.st_mtime_ns))
    return tuple(sources)


def check_source_unchanged(source: PdfSource) -> None:
    """Fail when a source differs from the manifest's record of it."""
    try:
        info = os.stat(source.path)
    except OSError as error:
        raise PdfReadError(
            f"PDF {source.path!r} disappeared after planning: {error}"
        ) from error
    if (info.st_size, info.st_mtime_ns) != (source.size, source.mtime_ns):
        raise PdfReadError(
            f"PDF {source.path!r} changed after planning; register the "
            f"table again")


def read_manifest(paths: Sequence[str]) -> PdfManifest:
    """Open each PDF, read its page sizes, and close it.

    Raises:
        PdfReadError: A file is missing, is not a PDF, or has no pages.
    """
    import pypdfium2 as pdfium

    sources = stat_sources(paths)
    pages = []
    for source_index, source in enumerate(sources):
        try:
            document = pdfium.PdfDocument(source.path)
        except Exception as error:
            raise PdfReadError(
                f"cannot open PDF {source.path!r}: {error}") from error
        try:
            count = len(document)
            if count == 0:
                raise PdfReadError(f"PDF {source.path!r} has no pages")
            for page_index in range(count):
                page = document[page_index]
                try:
                    width, height = page.get_size()
                finally:
                    page.close()
                pages.append(PdfPageRef(source_index, page_index,
                                        float(width), float(height)))
        finally:
            document.close()
    return PdfManifest(sources, tuple(pages))
