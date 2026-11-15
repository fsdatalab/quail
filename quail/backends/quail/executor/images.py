"""Image documents on the GPU: pixels, the vision encoder, and positions.

A token store holds an image document as the vision start token, one
token per 32 by 32 pixels, and the vision end token. Each image token's
id is ``IMAGE_TOKEN_BASE`` or more: the image's first id plus the
token's index, row by row. ``RegionImages`` reads each image's PNG from
its scan's value columns, runs the model's vision encoder on the images
a chunk reads, and gives each image token its embedding.

Qwen's vision models give an image token at row ``i`` and column ``j``
of an image that starts at position ``s`` the rotary position
``(s, s + i, s + j)`` (time, height, width), and the token after the
image the position ``s + max(rows, columns)``: an image of ``N`` tokens
advances the positions by ``max(rows, columns)``, not ``N``. Quail's
positions also say where each token's KV goes, so they stay one per
token; ``RegionImages.rotary_positions`` gives every row after an image
its position less the image's ``N - max(rows, columns)``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from _quail_native import IMAGE_TOKEN_BASE, IMAGE_TOKEN_PIXELS

# Qwen's vision patch, the frames each patch covers, and the patches
# merged into one token along each side
PATCH_PIXELS = 16
TEMPORAL_PATCH = 2
MERGE = IMAGE_TOKEN_PIXELS // PATCH_PIXELS
# threads that decode a region's PNGs ahead of the vision encoder
DECODE_THREADS = 8
# the most patches one vision encoder call reads, which bounds its
# activations; a model spec's vision_mem_bytes reserves them
ENCODER_PATCHES = 32_768


@dataclass(frozen=True)
class Image:
    """One image of a region: its first token id, size, and PNG bytes."""

    first: int
    rows: int
    columns: int
    png: bytes

    @property
    def tokens(self) -> int:
        return self.rows * self.columns

    @property
    def shift(self) -> int:
        """How much less the rows after the image advance than its tokens."""
        return self.tokens - max(self.rows, self.columns)


def document_key(key):
    """The (alias, document) an arena key names, or None.

    A document's key is (alias, document); a map's branch and an
    extraction's own documents nest it in a longer tuple.
    """
    if not isinstance(key, tuple):
        return None
    if len(key) == 2 and isinstance(key[0], str) and isinstance(
            key[1], (int, np.integer)):
        return key[0], int(key[1])
    for part in key:
        found = document_key(part)
        if found is not None:
            return found
    return None


def png_size(png: bytes) -> tuple[int, int]:
    """The (width, height) of a PNG's header."""
    return (int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big"))


def pixel_patches(torch, rgb):
    """An image's pixels as the vision encoder's patches, in Qwen's order.

    Args:
        torch: The torch module.
        rgb: The image, (height, width, 3) uint8 on the GPU.

    Returns:
        (patches, grid): (rows * columns * 4, 1536) float32 patches, and
        the (1, height, width) patch grid, as Qwen2VLImageProcessor
        writes them.
    """
    height, width, _ = rgb.shape
    grid_h, grid_w = height // PATCH_PIXELS, width // PATCH_PIXELS
    x = rgb.permute(2, 0, 1).float().div(255.0).sub(0.5).div(0.5)
    x = x.unsqueeze(0).expand(TEMPORAL_PATCH, -1, -1, -1)
    x = x.reshape(1, TEMPORAL_PATCH, 3, grid_h // MERGE, MERGE, PATCH_PIXELS,
                  grid_w // MERGE, MERGE, PATCH_PIXELS)
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    patches = x.reshape(grid_h * grid_w, 3 * TEMPORAL_PATCH * PATCH_PIXELS ** 2)
    return patches, (1, grid_h, grid_w)


def decode_png(png: bytes) -> np.ndarray:
    """A PNG's pixels as (height, width, 3) uint8."""
    from PIL import Image as PILImage

    with PILImage.open(io.BytesIO(png)) as opened:
        return np.asarray(opened.convert("RGB"))


class RegionImages:
    """The images a region's documents hold, and their embeddings.

    Args:
        documents: The region's token documents by alias.
        columns: The region's value columns by alias; an alias whose
            documents are images has a binary column of their PNGs.
    """

    def __init__(self, documents: dict, columns: dict):
        self.images: dict[int, Image] = {}
        # by alias, each document's image's first token id, or -1
        self.firsts: dict[str, np.ndarray] = {}
        for alias, table in columns.items():
            pngs = _binary_column(table)
            if pngs is None or alias not in documents:
                continue
            store = documents[alias]
            firsts = np.full(len(store), -1, dtype=np.int64)
            for index, png in enumerate(pngs.to_pylist()):
                if png is None:
                    continue
                tokens = store[index]
                first = int(_token_at(tokens, 1))
                if first < IMAGE_TOKEN_BASE:
                    continue
                firsts[index] = first
                if first not in self.images:
                    width, height = png_size(png)
                    self.images[first] = Image(
                        first, height // IMAGE_TOKEN_PIXELS,
                        width // IMAGE_TOKEN_PIXELS, png)
            self.firsts[alias] = firsts
        order = sorted(self.images)
        self._sorted = np.asarray(order, dtype=np.int64)
        # each image's pixels, decoded off the GPU loop's thread
        self._pool = None
        self._pixels: dict = {}
        if self.images:
            from concurrent.futures import ThreadPoolExecutor

            self._pool = ThreadPoolExecutor(max_workers=DECODE_THREADS)
            self._pixels = {first: self._pool.submit(decode_png, image.png)
                            for first, image in self.images.items()}
        self._embeddings: dict[int, object] = {}
        # (start, end) CUDA events around each encoder call, the seconds
        # the GPU loop waited for decoded pixels, and the images encoded
        self._encoder_events: list = []
        self.decode_wait_s = 0.0
        self.encoded = 0
        # by arena key, and by image, the position of the image's first token
        self._starts: dict = {}
        self._image_starts: dict[int, int] = {}

    def __bool__(self) -> bool:
        return bool(self.images)

    def encoder_s(self) -> float:
        """The GPU seconds the vision encoder took so far."""
        return sum(start.elapsed_time(end) for start, end in
                   self._encoder_events) / 1000.0

    def close(self) -> None:
        """Stop decoding and drop every image's pixels and embeddings."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        self._pixels.clear()
        self._embeddings.clear()

    def image_of(self, token: int) -> Image:
        """The image a token id belongs to."""
        at = int(np.searchsorted(self._sorted, token, side="right")) - 1
        return self.images[int(self._sorted[at])]

    def key_image(self, key) -> Image | None:
        """The image an arena key's document holds, or None."""
        found = document_key(key)
        if found is None or found[0] not in self.firsts:
            return None
        first = int(self.firsts[found[0]][found[1]])
        return self.images.get(first) if first >= 0 else None

    # ---- embeddings ---------------------------------------------------

    def embed(self, torch, visual, chunk, text_embed):
        """The input embeddings of a chunk's token ids.

        The rows are found from the host's copy of the ids, so the GPU
        loop does not wait for the device.

        Args:
            torch: The torch module.
            visual: The model's vision encoder.
            chunk: The packed chunk.
            text_embed: The model's token embedding module.
        """
        ids = chunk.input_ids
        host = _host(chunk.host_ids, ids)
        image_rows = np.flatnonzero(host >= IMAGE_TOKEN_BASE)
        if not len(image_rows):
            return text_embed(ids)
        tokens = host[image_rows]
        hidden = text_embed(ids.masked_fill(ids >= IMAGE_TOKEN_BASE, 0))
        starts = self._sorted[np.searchsorted(self._sorted, tokens, side="right") - 1]
        firsts = np.unique(starts).tolist()
        self._encode(torch, visual, [first for first in firsts
                                     if first not in self._embeddings],
                     hidden.dtype)
        # each image's embeddings stacked, and each row's index into them
        stacked = torch.cat([self._embeddings[first] for first in firsts])
        offsets = dict(zip(firsts, np.cumsum([0] + [self.images[first].tokens
                                                     for first in firsts])))
        index = tokens - starts + np.array([offsets[first] for first in starts])
        device = hidden.device
        hidden[_upload(torch, image_rows, device)] = stacked[
            _upload(torch, index, device)]
        # an image's KV holds it once its last token is written; a
        # document recomputed after eviction encodes it again
        written = set(tokens.tolist())
        for first in firsts:
            if first + self.images[first].tokens - 1 in written:
                self._embeddings.pop(first, None)
        return hidden

    def _encode(self, torch, visual, firsts, dtype):
        """Run the vision encoder on images, keeping their embeddings.

        Images go in calls of at most ENCODER_PATCHES patches, and an
        image larger than that alone.
        """
        batch, patches = [], 0
        for first in firsts:
            size = MERGE * MERGE * self.images[first].tokens
            if batch and patches + size > ENCODER_PATCHES:
                self._encode_batch(torch, visual, batch, dtype)
                batch, patches = [], 0
            batch.append(first)
            patches += size
        if batch:
            self._encode_batch(torch, visual, batch, dtype)

    def _encode_batch(self, torch, visual, firsts, dtype):
        """Run the vision encoder on one call's images."""
        if not firsts:
            return
        import time

        device = visual_device(visual)
        patches, grids = [], []
        for first in firsts:
            waited = time.perf_counter()
            pixels = self._pixels.pop(first, None)
            pixels = (decode_png(self.images[first].png) if pixels is None
                      else pixels.result())
            self.decode_wait_s += time.perf_counter() - waited
            rgb = torch.from_numpy(pixels.copy()).to(device)
            found, grid = pixel_patches(torch, rgb)
            patches.append(found)
            grids.append(grid)
        pixel_values = torch.cat(patches).to(visual_dtype(visual))
        grid_thw = torch.tensor(grids, dtype=torch.int64, device=device)
        timed = pixel_values.is_cuda
        if timed:
            start, end = (torch.cuda.Event(enable_timing=True),
                          torch.cuda.Event(enable_timing=True))
            start.record()
        embeddings = visual(pixel_values, grid_thw=grid_thw)
        if timed:
            end.record()
            self._encoder_events.append((start, end))
        self.encoded += len(firsts)
        sizes = [self.images[first].tokens for first in firsts]
        for first, found in zip(firsts, embeddings.split(sizes), strict=True):
            self._embeddings[first] = found.to(dtype)

    # ---- positions ----------------------------------------------------

    def rotary_positions(self, chunk, answer_keys):
        """Each row's rotary position, or None for a chunk of no image.

        Returns (3, rows) positions for a chunk with image tokens, the
        rows' positions less their images' shifts for one without, and
        None when no row follows an image.

        Args:
            chunk: The packed chunk.
            answer_keys: The arena's answer slot keys, for decode rows.
        """
        ids = _host(chunk.host_ids, chunk.input_ids)
        positions = _host(chunk.host_positions, chunk.positions)
        rotary = np.broadcast_to(positions, (3, len(positions))).copy()
        changed = False
        for key, row0, _, end in chunk.row_spans:
            image = self.key_image(key)
            if image is None:
                continue
            rows = slice(row0, end)
            found = ids[rows]
            marked = found >= IMAGE_TOKEN_BASE
            if marked.any():
                index = np.flatnonzero(marked)
                within = found[index].astype(np.int64) - image.first
                start = int(positions[row0 + index[0]] - within[0])
                self._starts[key] = start
                self._image_starts[image.first] = start
                grid_row, grid_column = np.divmod(within, image.columns)
                rotary[0, row0 + index] = start
                rotary[1, row0 + index] = start + grid_row
                rotary[2, row0 + index] = start + grid_column
                changed = True
            # a document that borrowed its image's KV starts it where its
            # lender does
            start = self._starts.get(key, self._image_starts.get(image.first))
            if start is None:
                continue
            past = (positions[rows] >= start + image.tokens) & ~marked
            rotary[:, row0 + np.flatnonzero(past)] -= image.shift
            changed = True
        decode = chunk.decode_rows
        if decode is not None and len(decode):
            first = len(ids) - len(decode)
            for row, slot in enumerate(np.asarray(decode.slots).tolist()):
                key = answer_keys[slot]
                image = self.key_image(key)
                if image is None:
                    continue
                rotary[:, first + row] -= image.shift
                changed = True
        if not changed:
            return None
        import torch

        if (rotary[1:] == rotary[0]).all():
            rotary = rotary[0]
        return _upload(torch, rotary, chunk.positions.device)


def _upload(torch, values: np.ndarray, device):
    """A host array on the device, copied from pinned memory without waiting."""
    found = torch.from_numpy(np.ascontiguousarray(values, dtype=np.int64))
    if device.type != "cuda":
        return found
    return found.pin_memory().to(device, non_blocking=True)


def _host(values, device_values) -> np.ndarray:
    """A chunk's host copy of a tensor, or the tensor copied to the host."""
    if values is not None:
        return np.asarray(values, dtype=np.int64)
    return device_values.cpu().numpy().astype(np.int64)


def _token_at(tokens, index):
    parts = getattr(tokens, "token_parts", (tokens,))
    for part in parts:
        values = part.numpy() if hasattr(part, "numpy") else np.asarray(part)
        if index < len(values):
            return values[index]
        index -= len(values)
    return -1


def _binary_column(table: pa.Table):
    for index, field in enumerate(table.schema):
        if pa.types.is_binary(field.type) or pa.types.is_large_binary(field.type):
            return table.column(index)
    return None


def visual_device(visual):
    return next(visual.parameters()).device


def visual_dtype(visual):
    return next(visual.parameters()).dtype
