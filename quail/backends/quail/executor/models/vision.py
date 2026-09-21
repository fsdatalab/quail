"""Soft token embeddings for rendered pages, through the loaded vision tower.

A rendered page arrives as uint8 patches in the Gemma 4 processor's
layout (quail.pdf.render). The rescale to [0, 1] happens here on the
GPU in float32, as the processor does it, and the loaded model's
embed_multimodal runs the tower, the pooler, and the projection into
the language model's width. The result replaces the page's soft token
rows in the chunk's embeddings, unscaled: the model applies its
embedding normalizer to text rows only.

A chunk can hold a few hundred pages; the tower's activations for all
of them at once would take tens of GB. Pages therefore go through the
tower in runs whose patches fit the memory the model keeps free of KV
for images (ModelSpec.image_reserve_bytes).
"""

from __future__ import annotations

from collections.abc import Sequence

from quail.pdf.render import patch_positions

# vLLM's encoder path took 1.44 GB above the loaded weights for eight
# letter pages of 2,394 patches each at budget 280
# (/results/ablations/pdf_probe_vision.json, peak_bytes).
ACTIVATION_BYTES_PER_PATCH = 1.44 * 2**30 / (8 * 2394)


def embed_runs(images: Sequence, patch_budget: int) -> list[Sequence]:
    """Split images into runs of at most `patch_budget` patches each.

    A run always holds at least one image, so a page larger than the
    budget still embeds, alone.
    """
    runs: list[list] = []
    patches = 0
    for image in images:
        size = image.page.grid[0] * image.page.grid[1]
        if runs and patches + size > patch_budget:
            runs.append([])
            patches = 0
        if not runs:
            runs.append([])
        runs[-1].append(image)
        patches += size
    return runs


class VisionEmbedder:
    """The vision path of a vLLM Gemma 4 family model.

    Args:
        torch: The torch module.
        model: The loaded model; it must carry vision_tower and
            embed_multimodal(pixel_values=..., pixel_position_ids=...).
        device: Where the tower runs; host tensors go there through
            pinned memory when it is a CUDA device.
        reserve_bytes: GPU memory one tower call may take for its
            activations; 0 embeds a chunk's pages in one call.

    Raises:
        ValueError: The checkpoint has no vision tower.
    """

    def __init__(self, torch, model, device="cuda", reserve_bytes: float = 0.0):
        if getattr(model, "vision_tower", None) is None:
            raise ValueError("the loaded model has no vision tower")
        self.torch = torch
        self.model = model
        self.device = torch.device(device)
        self.patch_budget = (int(reserve_bytes / ACTIVATION_BYTES_PER_PATCH)
                             if reserve_bytes else None)
        self.images_embedded = 0
        self.embed_calls = 0

    def _to_device(self, array):
        tensor = self.torch.from_numpy(array)
        if self.device.type == "cuda":
            return tensor.pin_memory().to(self.device, non_blocking=True)
        return tensor.to(self.device)

    def _pixel_values(self, page):
        """The page's patches as float32 in [0, 1], on the device."""
        return self._to_device(page.patches).to(self.torch.float32).div_(255.0)

    def _position_ids(self, page):
        return self._to_device(patch_positions(page.grid))

    def embed(self, images: Sequence) -> object:
        """One (rows, hidden) tensor: every image's soft tokens, in image order.

        Args:
            images: ChunkImage values; each page's soft token count
                must equal its ChunkImage.rows.

        Raises:
            RuntimeError: The tower produced a different soft token
                count than the prompt reserved for a page.
        """
        runs = (embed_runs(images, self.patch_budget)
                if self.patch_budget else [images])
        embeddings = []
        for run in runs:
            pages = [image.page for image in run]
            embeddings.extend(self.model.embed_multimodal(
                pixel_values=[self._pixel_values(page) for page in pages],
                pixel_position_ids=[self._position_ids(page) for page in pages]))
            self.embed_calls += 1
        if len(embeddings) != len(images):
            raise RuntimeError(
                f"the vision tower returned {len(embeddings)} embeddings for "
                f"{len(images)} images")
        for image, embedding in zip(images, embeddings):
            if embedding.shape[0] != image.rows:
                raise RuntimeError(
                    f"page {image.page.page_id} embedded to "
                    f"{embedding.shape[0]} soft tokens; its prompt reserved "
                    f"{image.rows}")
        self.images_embedded += len(images)
        return self.torch.cat(embeddings, dim=0)
