"""Soft token embeddings for rendered pages, through the loaded vision tower.

A rendered page arrives as uint8 patches in the Gemma 4 processor's
layout (quail.pdf.render). The rescale to [0, 1] happens here on the
GPU in float32, as the processor does it, and the loaded model's
embed_multimodal runs the tower, the pooler, and the projection into
the language model's width. The result replaces the page's soft token
rows in the chunk's embeddings, unscaled: the model applies its
embedding normalizer to text rows only.
"""

from __future__ import annotations

from collections.abc import Sequence

from quail.pdf.render import patch_positions


class VisionEmbedder:
    """The vision path of a vLLM Gemma 4 family model.

    Args:
        torch: The torch module.
        model: The loaded model; it must carry vision_tower and
            embed_multimodal(pixel_values=..., pixel_position_ids=...).
        device: Where the tower runs; host tensors go there through
            pinned memory when it is a CUDA device.

    Raises:
        ValueError: The checkpoint has no vision tower.
    """

    def __init__(self, torch, model, device="cuda"):
        if getattr(model, "vision_tower", None) is None:
            raise ValueError("the loaded model has no vision tower")
        self.torch = torch
        self.model = model
        self.device = torch.device(device)
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
        pages = [image.page for image in images]
        embeddings = self.model.embed_multimodal(
            pixel_values=[self._pixel_values(page) for page in pages],
            pixel_position_ids=[self._position_ids(page) for page in pages])
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
        self.embed_calls += 1
        return self.torch.cat(list(embeddings), dim=0)
