"""The parsers AI.PARSE runs: PDF stored text, and OCR on page images.

A parser takes rows of a read_pdfs table, each one file and its pages,
and yields each row's text as the row finishes, in any order. A row
with several pages gets its pages' texts joined in page order.

- ``pdfium`` reads the text the PDF stores, on the CPU, in worker
  processes. A page with no stored text gives ''.
- ``lightonocr-3-0.8b`` renders each page and transcribes it with
  LightOnOCR-3-0.8B on the GPU
  (quail.backends.quail.executor.parse.RowParser).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from quail.pdf import PAGE_SEPARATOR, check_unchanged, stored_text

PDFIUM = "pdfium"
LIGHTONOCR = "lightonocr-3-0.8b"
PARSERS = (PDFIUM, LIGHTONOCR)
MODES = ("text", "layout")


@dataclass(frozen=True)
class ParserSpec:
    """What the planner knows about a parser before it runs.

    Attributes:
        name: The EngineConfig.parser value.
        gpu: Whether it runs a model on the GPU beside the reader.
        modes: The AI.PARSE modes it supports.
        tokens_per_page: Expected output tokens per page, for planning
            when no sample is parsed.
        version: What else decides its output; a saved parse with
            another version is not read.
    """

    name: str
    gpu: bool
    modes: tuple
    tokens_per_page: float
    version: str = ""


# LightOnOCR's mean output on ParseBench: 2,385,005 tokens over 2,080
# pages (run 20261012T022224Z-74d939); pdfium's is sampled per table
PARSER_SPECS = {
    PDFIUM: ParserSpec(PDFIUM, gpu=False, modes=("text",), tokens_per_page=0.0,
                       version="pypdfium2 5.13.0"),
    # the checkpoint revision, render size, output limit, and greedy decoding
    LIGHTONOCR: ParserSpec(
        LIGHTONOCR, gpu=True, modes=("text",), tokens_per_page=1147.0,
        version="36d8636925327c671bd8e479c517a004f7566bd9 2048px 4096 greedy"),
}


def parser_spec(name: str) -> ParserSpec:
    """The ParserSpec of a parser name.

    Raises:
        ValueError: The name is not a parser.
    """
    if name not in PARSER_SPECS:
        raise ValueError(f"unknown parser {name!r}; parsers: {list(PARSER_SPECS)}")
    return PARSER_SPECS[name]


def _stored_row(path_file, pages) -> tuple[str, int]:
    check_unchanged(path_file)
    texts = stored_text(path_file.path, pages)
    return PAGE_SEPARATOR.join(texts), sum(1 for text in texts if not text.strip())


class PdfiumParser:
    """Read rows' stored text in worker processes.

    Args:
        workers: Worker processes.
        window: Rows submitted ahead of the consumer at most.
    """

    name = PDFIUM

    def __init__(self, *, workers: int = 4, window: int = 64):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        # spawn: the process that owns the session may have CUDA initialized
        self.pool = ProcessPoolExecutor(
            workers, mp_context=multiprocessing.get_context("spawn"))
        self.window = window
        self.stats = {}

    def parse(self, rows):
        """Yield (key, text) per row, in submission order.

        Args:
            rows: Iterable of (key, PdfFile, page numbers from 1).
        """
        started = time.perf_counter()
        stats = self.stats = dict(pages_parsed=0, parse_empty_pages=0)
        ahead = deque()

        def finish():
            key, pages, future = ahead.popleft()
            text, empty = future.result()
            stats["pages_parsed"] += pages
            stats["parse_empty_pages"] += empty
            return key, text

        for key, path_file, pages in rows:
            ahead.append((key, len(pages),
                          self.pool.submit(_stored_row, path_file, list(pages))))
            if len(ahead) >= self.window:
                yield finish()
        while ahead:
            yield finish()
        stats["parse_s"] = time.perf_counter() - started

    def close(self) -> None:
        self.pool.shutdown(cancel_futures=True)


def stored_texts(rows) -> list[str]:
    """Each row's stored text in this process; for planning samples.

    Args:
        rows: Iterable of (PdfFile, page numbers from 1).
    """
    return [_stored_row(path_file, pages)[0] for path_file, pages in rows]


# the OCR parser's settings, as in the ParseBench runs: the card's
# output limit, a context that holds a square page's 4,096 image
# tokens, the chat turn, and a full answer, and rows per step
OCR_MAX_OUTPUT_TOKENS = 4096
OCR_CONTEXT_TOKENS = 8448
OCR_STEP_TOKENS = 16384
OCR_MAX_LANES = 256
OCR_LONGEST_EDGE = 2048
# memory for the vision tower's activations on one batch of pages,
# outside the parser's arena and step activations
OCR_VISION_BYTES = 2 * 2**30
# the share of memory left after both models' weights that the parser
# may take; the reader keeps the rest
OCR_MEMORY_SHARE = 0.5


def ocr_lane_bytes(model) -> float:
    """Bytes one OCR lane reserves: a full context of KV and one state slot."""
    return OCR_CONTEXT_TOKENS * model.kappa_full + model.state_bytes


def ocr_memory(model, free_bytes: float) -> tuple[int, float]:
    """The OCR parser's lanes and the bytes it takes beside the reader.

    Args:
        model: The parser's ModelSpec.
        free_bytes: Memory left after the reader's weights.

    Returns:
        Lanes, and the bytes of weights, arena, step activations, and
        vision activations.

    Raises:
        ValueError: Not one lane fits.
    """
    fixed = (model.W_mem + 2 * OCR_STEP_TOKENS * model.act_per_token
             + OCR_VISION_BYTES)
    lanes = min(OCR_MAX_LANES,
                int((free_bytes * OCR_MEMORY_SHARE - fixed) // ocr_lane_bytes(model)))
    if lanes < 1:
        raise ValueError("the OCR parser does not fit beside the reader")
    return lanes, fixed + (lanes + 1) * ocr_lane_bytes(model)
