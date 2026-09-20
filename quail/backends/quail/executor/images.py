"""The image source of one filter chain over PDF rows.

The chain sees documents by local index in admission order. PageImages
maps that index to the alias's row, renders the row's pages ahead
through a PdfiumPrefetcher, and hands each page back with the prefix
span its soft tokens occupy, shifted past the shared preamble the
chain puts before every document.

Building a PageImages starts nothing: the chain calls open() when it
begins and close() when it ends, so a chain that is described but
never run (a filter whose survivors stream into a join builds its
inputs twice) leaves no render processes behind.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

from quail.pdf.prefetch import ImagePrefetcher, PdfiumPrefetcher, RenderedPage
from quail.pdf.prompt import ImageBlock, PagePrompts


class PageImages:
    """Rendered pages for the documents of one filter chain.

    Args:
        prompts: The alias's rows as PagePrompts.
        document_ids: The alias row of each local document, in the
            order the chain admits them.
        pre_tokens: Tokens the chain prepends to every document; block
            offsets shift by this many rows.
        open_prefetcher: Builds the page renderer from (pdf_input,
            spec, order) at open(); a PdfiumPrefetcher when omitted.
    """

    def __init__(self, prompts: PagePrompts, document_ids: Sequence[int],
                 pre_tokens: int,
                 open_prefetcher: Callable[..., ImagePrefetcher] | None = None):
        self.prompts = prompts
        self.document_ids = [int(row) for row in document_ids]
        self.pre_tokens = pre_tokens
        self._open_prefetcher = open_prefetcher or PdfiumPrefetcher
        self._prefetcher: ImagePrefetcher | None = None
        self._metrics: dict = {}

    def open(self) -> None:
        """Start rendering the documents' pages in admission order."""
        if self._prefetcher is None:
            self._prefetcher = self._open_prefetcher(
                self.prompts.pdf_input, self.prompts.spec,
                order=self.document_ids)

    def take(self, doc: int) -> tuple[tuple[ImageBlock, RenderedPage], ...]:
        """The document's pages with their soft token spans, in prompt order."""
        self.open()
        row = self.document_ids[doc]
        blocks = self.prompts.blocks(row)
        pages = self._prefetcher.take(row)
        if len(pages) != len(blocks) or any(
                page.page_id != block.page_id
                for page, block in zip(pages, blocks)):
            raise RuntimeError(
                f"row {row} rendered pages "
                f"{[page.page_id for page in pages]}, but its prompt lists "
                f"{[block.page_id for block in blocks]}")
        return tuple(
            (replace(block, offset=block.offset + self.pre_tokens), page)
            for block, page in zip(blocks, pages))

    def metrics(self) -> dict:
        """The renderer's counters; those at close() once closed."""
        if self._prefetcher is not None:
            return self._prefetcher.metrics()
        return self._metrics

    def close(self) -> None:
        """Stop rendering and release the render processes."""
        if self._prefetcher is not None:
            self._metrics = self._prefetcher.metrics()
            self._prefetcher.close()
            self._prefetcher = None
