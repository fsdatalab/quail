"""Turn PDF pages into text with LiteParse, for the OCR operator.

LiteParse (Apache 2.0, runs locally) reads each page's text layer in
reading order and runs Tesseract over the pages it judges scanned,
sparse, or garbled, so a scanned page gets text too. It parses one
file per call, in a pool of persistent worker processes with a hard
per-file timeout; this module fans the files out over that pool and
hands their page texts back in file order, one file at a time, so the
tokenizer can start on the first file while the pool parses the rest.

Measured on FinanceBench 10-K filings (160 and 503 pages) on a CPU
box: 3 to 6 ms per page from the text layer, about 80 ms per page
when Tesseract runs, and 4 workers gave 900 text-layer pages per
second.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from quail.pdf.manifest import PdfReadError, check_source_unchanged
from quail.pdf.types import PDFInput

# Worker processes when the caller names none: text-layer parsing of
# a page is a few milliseconds, so a handful keep well ahead of the
# tokenizer.
DEFAULT_PROCESSES = 4
# Pages the length estimate reads before the full extraction runs.
SAMPLE_PAGES = 32
# Between the pages of a whole-file row.
PAGE_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class OcrOptions:
    """How the OCR operator runs LiteParse.

    Attributes:
        language: The Tesseract language code for scanned pages.
        processes: LiteParse worker processes; zero parses in the
            calling process, which CPU tests use.
        timeout_s: Hard limit per file in the worker pool.
    """

    language: str = "eng"
    processes: int = DEFAULT_PROCESSES
    timeout_s: float = 600.0

    def identity(self) -> str:
        """The part of the options that changes the extracted text."""
        return f"language={self.language}"


def _open_parser(options: OcrOptions, max_pages: int, **extra):
    from liteparse import LiteParse

    settings = dict(
        ocr_enabled=True, ocr_language=options.language,
        output_format="text", quiet=True, continue_on_page_error=True,
        # LiteParse stops at 1000 pages unless told the real bound
        max_pages=max(max_pages, 1), **extra)
    if options.processes > 0:
        settings.update(pool_size=options.processes,
                        parse_timeout=options.timeout_s)
    return LiteParse(**settings)


def page_texts(pdf_input: PDFInput, options: OcrOptions,
               sources: Sequence[int] | None = None, *,
               first_pages: int | None = None
               ) -> Iterator[tuple[int, list[str]]]:
    """Yield (source index, its page texts in page order), in the order asked.

    Every file is submitted to the pool at once; each is yielded as
    soon as it and every file before it are parsed.

    Args:
        pdf_input: The bound PDF rows; every source when `sources` is
            omitted.
        options: How LiteParse runs.
        sources: The source indices to read, in the order to yield.
        first_pages: Read only the first this many pages of each
            source, for a sample; every page when omitted.

    Raises:
        PdfReadError: A source changed since planning, cannot be
            parsed, or reports a page count other than the manifest's.
    """
    counts = pdf_input.page_counts
    wanted = (list(range(len(pdf_input.sources))) if sources is None
              else list(sources))
    for index in wanted:
        check_source_unchanged(pdf_input.sources[index])
    extra = {"target_pages": f"1-{first_pages}"} if first_pages else {}

    def parse(parser, index: int) -> tuple[int, list[str]]:
        source = pdf_input.sources[index]
        try:
            result = parser.parse(source.path)
        except Exception as error:
            raise PdfReadError(
                f"cannot parse PDF {source.path!r}: {error}") from error
        if result.total_pages != counts[index]:
            raise PdfReadError(
                f"PDF {source.path!r} has {result.total_pages} pages, but "
                f"the manifest lists {counts[index]}; register the table again")
        limit = counts[index] if first_pages is None else min(
            counts[index], first_pages)
        texts = [""] * limit
        for page in result.pages:
            if 1 <= page.page_num <= limit:
                texts[page.page_num - 1] = page.text or ""
        return index, texts

    with _open_parser(options, max(counts, default=0), **extra) as parser:
        with ThreadPoolExecutor(max(1, options.processes)) as threads:
            yield from threads.map(lambda index: parse(parser, index), wanted)


def row_texts(pdf_input: PDFInput, options: OcrOptions
              ) -> Iterator[tuple[int, list[str]]]:
    """Yield (first row index, the texts of consecutive rows), in row order.

    A row's text is its pages' text in prompt order, joined with
    PAGE_SEPARATOR. Rows are yielded as soon as every file their
    pages come from is parsed; files parse in source order, and a
    row's sources never follow its position, so nothing waits on a
    file it does not read.

    Raises:
        PdfReadError: See page_texts.
    """
    by_source: dict[int, list[str]] = {}
    last_source = [
        max(pdf_input.pages[page_id].source_index for page_id in row.page_ids)
        for row in pdf_input.rows]
    next_row = 0
    for index, texts in page_texts(pdf_input, options):
        by_source[index] = texts
        start = next_row
        while next_row < len(last_source) and last_source[next_row] <= index:
            next_row += 1
        if next_row == start:
            continue
        chunk = []
        for row in pdf_input.rows[start:next_row]:
            chunk.append(PAGE_SEPARATOR.join(
                by_source[pdf_input.pages[page_id].source_index]
                [pdf_input.pages[page_id].page_index]
                for page_id in row.page_ids))
        yield start, chunk
    if next_row != len(last_source):
        raise PdfReadError(
            f"rows from {next_row} on read pages of a source that was never "
            f"parsed; the rows do not follow source order")


def sample_page_texts(pdf_input: PDFInput, options: OcrOptions,
                      pages: int = SAMPLE_PAGES) -> list[str]:
    """The text of about `pages` pages from the first sources, for an estimate.

    Reads whole sources in order, each capped at `pages`, until that
    many pages are in hand, so a corpus of short files samples several
    of them and a corpus of long files samples the start of the first.
    """
    chosen = []
    in_hand = 0
    for index, count in enumerate(pdf_input.page_counts):
        if in_hand >= pages:
            break
        chosen.append(index)
        in_hand += min(count, pages)
    texts = [text for _index, page in page_texts(
        pdf_input, options, chosen, first_pages=pages) for text in page]
    return texts[:pages]
