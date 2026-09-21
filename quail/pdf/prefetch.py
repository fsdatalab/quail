"""Render PDF pages ahead of the filter chain, in admission order.

The chain admits rows in a known order, so the prefetcher submits
render jobs in that order to a pool of PDFium processes and keeps a
bounded number of pages outstanding. take(row) blocks until that row's
pages are back. Nothing is rendered twice, and rows the chain never
asks for are cancelled at close().

A worker keeps its PDF documents open between pages and checks each
file against the manifest's size and modification time when it opens
it, so a source edited after planning fails instead of rendering
different pixels.
"""

from __future__ import annotations

import atexit
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Protocol

import numpy as np

from quail.pdf.manifest import check_source_unchanged
from quail.pdf.render import patchify, render_page
from quail.pdf.types import PDFInput, PdfSource
from quail.specs.base import ModelSpec
from quail.specs.vision import render_size

# Render processes when the caller names none. The vision encoder takes
# about 30 ms per page on an H100 at budget 280 and one PDFium process
# renders a text page in about 20 ms, so two keep it fed with margin
# (/results/ablations/pdf_probe_render.json, pdf_probe_vision.json).
DEFAULT_PROCESSES = 2
# Pages queued or rendered but not yet taken, when the caller names
# none: a few chunks' worth of single pages at under 2 MB each.
DEFAULT_OUTSTANDING_PAGES = 64


@dataclass(frozen=True)
class RenderedPage:
    """One page as uint8 vision patches, before rescaling.

    Attributes:
        page_id: Index into PDFInput.pages.
        grid: The patch grid as (rows, cols).
        patches: (rows * cols, patch * patch * 3) uint8, patches in
            row-major order, each patch's pixels row-major, channel last.
        render_s: Seconds the worker spent drawing and cutting it.
    """

    page_id: int
    grid: tuple[int, int]
    patches: np.ndarray
    render_s: float


class ImagePrefetcher(Protocol):
    """What the filter chain needs from a page source."""

    def take(self, row: int) -> tuple[RenderedPage, ...]:
        """Block until the row's pages are rendered and return them in order."""
        ...

    def metrics(self) -> dict:
        """Counters for the query's timing record."""
        ...

    def close(self) -> None:
        """Cancel unrendered pages and release the workers."""
        ...


@dataclass(frozen=True)
class RenderJob:
    """One page to draw, with everything a worker needs to draw it."""

    page_id: int
    source: PdfSource
    page_index: int
    size_hw: tuple[int, int]
    patch: int


# One PDF stays open per process between its pages, keyed by the
# manifest record so a re-registered file is opened and checked again.
_OPEN_DOCUMENTS: dict[PdfSource, object] = {}
_CLOSE_REGISTERED = False


def _close_documents() -> None:
    for document in _OPEN_DOCUMENTS.values():
        document.close()
    _OPEN_DOCUMENTS.clear()


def _open_document(source: PdfSource):
    global _CLOSE_REGISTERED
    document = _OPEN_DOCUMENTS.get(source)
    if document is None:
        import pypdfium2 as pdfium

        if not _CLOSE_REGISTERED:
            # registered after pypdfium2's own exit handler, so the
            # documents close before it destroys the library
            atexit.register(_close_documents)
            _CLOSE_REGISTERED = True
        for stale in [s for s in _OPEN_DOCUMENTS if s.path == source.path]:
            _OPEN_DOCUMENTS.pop(stale).close()
        check_source_unchanged(source)
        document = pdfium.PdfDocument(source.path)
        _OPEN_DOCUMENTS[source] = document
    return document


def render_job(job: RenderJob) -> RenderedPage:
    """Render one page and cut it into patches; runs in a worker."""
    started = time.perf_counter()
    image = render_page(_open_document(job.source), job.page_index, job.size_hw)
    grid, patches = patchify(image, job.patch)
    return RenderedPage(job.page_id, grid, patches,
                        time.perf_counter() - started)


class _InlinePool:
    """A stand-in for the process pool that renders on submit()."""

    def submit(self, function: Callable, job: RenderJob) -> Future:
        future: Future = Future()
        try:
            future.set_result(function(job))
        except Exception as error:
            future.set_exception(error)
        return future

    def shutdown(self, **_: bool) -> None:
        pass


class PdfiumPrefetcher:
    """Bounded lookahead renderer over a spawn pool of PDFium processes.

    Args:
        pdf_input: The bound PDF rows.
        spec: The model, for the render geometry.
        order: Row indices in the order the chain will admit them,
            each at most once; rows left out are rendered only when
            taken. Row index order when omitted.
        processes: Render processes; DEFAULT_PROCESSES when None. Zero
            renders in the calling process when a row is submitted.
        max_outstanding_pages: Pages submitted but not yet taken before
            lookahead pauses. A row the chain asks for is always
            submitted, whatever the count, so the bound cannot deadlock.
    """

    def __init__(self, pdf_input: PDFInput, spec: ModelSpec,
                 order: Sequence[int] | None = None, *,
                 processes: int | None = None,
                 max_outstanding_pages: int | None = None):
        if not spec.image_patch_pixels:
            raise ValueError(f"model {spec.name!r} takes no images")
        self.pdf_input = pdf_input
        self.spec = spec
        self.patch = spec.image_patch_pixels
        self.budget = pdf_input.visual_tokens
        self.order = (list(range(len(pdf_input.rows))) if order is None
                      else [int(row) for row in order])
        if len(set(self.order)) != len(self.order) or any(
                not 0 <= row < len(pdf_input.rows) for row in self.order):
            raise ValueError("order lists row indices, each at most once")
        self.max_outstanding = (DEFAULT_OUTSTANDING_PAGES
                                if max_outstanding_pages is None
                                else max_outstanding_pages)
        self.processes = DEFAULT_PROCESSES if processes is None else processes
        self._pool = (_InlinePool() if self.processes == 0
                      else ProcessPoolExecutor(
                          self.processes, mp_context=get_context("spawn")))
        self._next = 0                   # position in order not yet submitted
        self._pending: dict[int, list[Future]] = {}
        self._outstanding_pages = 0
        self._taken: set[int] = set()
        self._pages_rendered = 0
        self._render_s = 0.0
        self._wait_s = 0.0
        self._fill()

    @property
    def outstanding_pages(self) -> int:
        """Pages submitted to the workers and not yet taken."""
        return self._outstanding_pages

    def _size(self, page_id: int) -> tuple[int, int]:
        page = self.pdf_input.pages[page_id]
        return render_size(self.spec, page.width_points, page.height_points,
                           self.budget)

    def _submit(self, row: int) -> None:
        futures = []
        for page_id in self.pdf_input.rows[row].page_ids:
            page = self.pdf_input.pages[page_id]
            job = RenderJob(
                page_id=page_id,
                source=self.pdf_input.sources[page.source_index],
                page_index=page.page_index,
                size_hw=self._size(page_id),
                patch=self.patch)
            futures.append(self._pool.submit(render_job, job))
        self._pending[row] = futures
        self._outstanding_pages += len(futures)

    def _fill(self) -> None:
        """Submit rows in admission order while under the page bound."""
        while (self._next < len(self.order)
               and self._outstanding_pages < self.max_outstanding):
            row = self.order[self._next]
            self._next += 1
            if row not in self._taken and row not in self._pending:
                self._submit(row)

    def take(self, row: int) -> tuple[RenderedPage, ...]:
        """Block until the row's pages are rendered and return them in order.

        Raises:
            ValueError: The row was taken before.
            PdfReadError: A source changed since planning.
        """
        if row in self._taken:
            raise ValueError(f"row {row} was already taken")
        if row not in self._pending:
            self._submit(row)
        futures = self._pending.pop(row)
        started = time.perf_counter()
        pages = tuple(self._checked(future.result()) for future in futures)
        self._wait_s += time.perf_counter() - started
        self._outstanding_pages -= len(futures)
        self._pages_rendered += len(futures)
        self._render_s += sum(page.render_s for page in pages)
        self._taken.add(row)
        self._fill()
        return pages

    def _checked(self, page: RenderedPage) -> RenderedPage:
        height, width = self._size(page.page_id)
        expected = (height // self.patch, width // self.patch)
        if page.grid != expected:
            raise RuntimeError(
                f"page {page.page_id} rendered as a {page.grid} patch grid; "
                f"the plan expected {expected}")
        return page

    def metrics(self) -> dict:
        """Counters for the query's timing record."""
        return {
            "render_processes": self.processes,
            "pages_rendered": self._pages_rendered,
            "render_worker_s": round(self._render_s, 4),
            "render_wait_s": round(self._wait_s, 4),
        }

    def close(self) -> None:
        """Cancel unrendered pages and release the workers."""
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._pending.clear()
        self._outstanding_pages = 0
