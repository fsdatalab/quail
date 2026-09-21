"""The prompt tokens a PDF row's pages occupy, and where each image sits.

A row's document tokens are, per page in order, the model's image start
marker, one soft token id per pooled patch block, and the end marker.
The soft token ids are placeholders: the executor overwrites their
embeddings with the vision encoder's output for that page. Nothing here
touches pixels; the counts come from the page sizes in the manifest, so
the planner and the executor agree on every row's layout before a page
is rendered.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from quail.pdf.types import PDFInput
from quail.specs.base import ModelSpec
from quail.specs.vision import soft_tokens


@dataclass(frozen=True)
class ImageBlock:
    """One page's soft tokens inside a row's document tokens.

    Attributes:
        page_id: Index into PDFInput.pages.
        offset: Position of the first soft token within the row's
            document tokens (after the start marker).
        soft_tokens: How many soft tokens the page becomes.
    """

    page_id: int
    offset: int
    soft_tokens: int

    @property
    def end(self) -> int:
        """Position just past the last soft token."""
        return self.offset + self.soft_tokens


class PagePrompts(Sequence):
    """Per PDF row, its document token ids and its image blocks.

    Indexing gives a row's token list, so the filter chain can treat it
    like any other document sequence; blocks(row) says which of those
    positions are a page's soft tokens.

    Args:
        pdf_input: The bound PDF rows.
        spec: A model with image support; its marker and soft token
            ids and its image geometry.

    Raises:
        ValueError: The model has no soft token id.
    """

    def __init__(self, pdf_input: PDFInput, spec: ModelSpec):
        if spec.image_soft_id < 0:
            raise ValueError(f"model {spec.name!r} has no image soft token id")
        self.pdf_input = pdf_input
        self.spec = spec
        budget = pdf_input.visual_tokens
        self._soft = tuple(
            soft_tokens(spec, page.width_points, page.height_points, budget)
            for page in pdf_input.pages)
        self._blocks = tuple(self._row_blocks(row) for row in pdf_input.rows)

    def _row_blocks(self, row) -> tuple[ImageBlock, ...]:
        blocks = []
        offset = 0
        for page_id in row.page_ids:
            offset += self.spec.image_start_id >= 0
            blocks.append(ImageBlock(page_id, offset, self._soft[page_id]))
            offset += self._soft[page_id] + (self.spec.image_end_id >= 0)
        return tuple(blocks)

    def __len__(self) -> int:
        return len(self.pdf_input.rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        spec = self.spec
        tokens: list[int] = []
        for block in self._blocks[index]:
            if spec.image_start_id >= 0:
                tokens.append(spec.image_start_id)
            tokens.extend([spec.image_soft_id] * block.soft_tokens)
            if spec.image_end_id >= 0:
                tokens.append(spec.image_end_id)
        return tokens

    def blocks(self, row: int) -> tuple[ImageBlock, ...]:
        """The row's pages as soft token spans, in prompt order."""
        return self._blocks[row]

    @property
    def lengths(self) -> tuple[int, ...]:
        """Each row's document token count."""
        frame = self.spec.image_frame_tokens
        return tuple(sum(block.soft_tokens + frame for block in blocks)
                     for blocks in self._blocks)

    def pages_within(self, tokens: int) -> int:
        """The most pages whose prompt positions fit in `tokens`."""
        smallest = min(self._soft, default=0) + self.spec.image_frame_tokens
        return tokens // smallest if smallest else 0
