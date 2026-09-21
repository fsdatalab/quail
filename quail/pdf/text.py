"""Extract the text of PDF pages with LiteParse, for the text reading.

LiteParse (Apache 2.0, runs locally) reads each page's text layer
laid out in reading order and, when asked, runs Tesseract OCR over
the page images so scanned pages get text too. It parses one file per
call, in a pool of persistent worker processes with a hard per-file
timeout; this module fans the sources out over that pool and puts the
page texts back into query row order.

Measured on FinanceBench 10-K filings (160 and 503 pages) on a CPU
box: 3 to 6 ms per page from the text layer alone, about 80 ms per
page with OCR on, and 4 workers gave 900 pages per second without
OCR. Nothing here touches the GPU or the token store; the session
tokenizes the text like any other document column.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
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
class PdfTextOptions:
    """How PDF pages become text.

    Attributes:
        ocr: Run Tesseract over the page images as well as reading the
            text layer. Off reads the text layer only, so a scanned
            page comes out empty; on costs about 80 ms per page.
        language: The Tesseract language code.
        processes: LiteParse worker processes; zero parses in the
            calling process, which CPU tests use.
        timeout_s: Hard limit per file in the worker pool.
    """

    ocr: bool = False
    language: str = "eng"
    processes: int = DEFAULT_PROCESSES
    timeout_s: float = 600.0

    def identity(self) -> str:
        """The part of the options that changes the extracted text."""
        return f"ocr={int(self.ocr)},language={self.language}"


@dataclass(frozen=True)
class PdfTexts:
    """Every row's text, with the counters of the extraction that made it.

    Attributes:
        rows: One string per query row, in row order; a whole-file
            row joins its pages with PAGE_SEPARATOR.
        metrics: pages read, pages that came out empty, characters,
            wall seconds, and the options that ran.
    """

    rows: tuple[str, ...]
    metrics: dict


def _open_parser(options: PdfTextOptions, max_pages: int, **extra):
    from liteparse import LiteParse

    settings = dict(
        ocr_enabled=options.ocr, ocr_language=options.language,
        output_format="text", quiet=True, continue_on_page_error=True,
        # LiteParse stops at 1000 pages unless told the real bound
        max_pages=max(max_pages, 1), **extra)
    if options.processes > 0:
        settings.update(pool_size=options.processes,
                        parse_timeout=options.timeout_s)
    return LiteParse(**settings)


def page_texts(pdf_input: PDFInput, options: PdfTextOptions,
               sources: Sequence[int] | None = None, *,
               first_pages: int | None = None) -> dict[int, list[str]]:
    """Each source's page texts in page order, keyed by source index.

    Args:
        pdf_input: The bound PDF rows; every source when `sources` is
            omitted.
        options: How the pages become text.
        sources: The source indices to read.
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
            return dict(threads.map(lambda index: parse(parser, index),
                                    wanted))


def row_texts(pdf_input: PDFInput, options: PdfTextOptions) -> PdfTexts:
    """Every row's text: its pages' text in prompt order.

    Raises:
        PdfReadError: See page_texts.
    """
    started = time.perf_counter()
    by_source = page_texts(pdf_input, options)
    rows = []
    empty = 0
    for row in pdf_input.rows:
        pages = []
        for page_id in row.page_ids:
            page = pdf_input.pages[page_id]
            text = by_source[page.source_index][page.page_index]
            empty += not text.strip()
            pages.append(text)
        rows.append(PAGE_SEPARATOR.join(pages))
    return PdfTexts(tuple(rows), {
        "pages": pdf_input.page_count,
        "empty_pages": empty,
        "chars": sum(len(text) for text in rows),
        "extract_s": round(time.perf_counter() - started, 4),
        "processes": options.processes,
        "ocr": options.ocr,
    })


def sample_page_texts(pdf_input: PDFInput, options: PdfTextOptions,
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
    by_source = page_texts(pdf_input, options, chosen, first_pages=pages)
    texts = [text for index in chosen for text in by_source[index]]
    return texts[:pages]
