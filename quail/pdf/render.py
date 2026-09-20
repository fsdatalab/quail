"""Render one PDF page to pixels and cut it into vision patches.

Pure functions over PDFium and NumPy, so they run in a render worker
process and in CPU tests alike. The page is drawn straight at the
processor's target size, so the processor's resize step becomes the
identity and the patch layout below matches its output exactly:
patches in row-major order, each patch's pixels row-major with the
channel last, position ids as (column, row).
"""

from __future__ import annotations

import numpy as np


def render_page(document, page_index: int, size_hw: tuple[int, int]):
    """Draw one page as an RGB uint8 array of exactly size_hw.

    Args:
        document: An open pypdfium2.PdfDocument.
        page_index: Zero based page index.
        size_hw: Target (height, width) in pixels. PDFium stretches the
            page to it; the caller computes it from the page's aspect
            ratio so the stretch is under one pixel.
    """
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c

    height, width = size_hw
    page = document[page_index]
    try:
        bitmap = pdfium.PdfBitmap.new_native(
            width, height, format=pdfium_c.FPDFBitmap_BGR, rev_byteorder=True)
        bitmap.fill_rect(0, 0, width, height, (255, 255, 255, 255))
        # grayscale antialiasing: LCD subpixel text would add color
        # fringes the model never sees in its training images
        pdfium_c.FPDF_RenderPageBitmap(
            bitmap.raw, page.raw, 0, 0, width, height, 0, pdfium_c.FPDF_ANNOT)
        return np.array(bitmap.to_numpy(), copy=True)
    finally:
        page.close()


def patchify(image: np.ndarray, patch: int) -> tuple[tuple[int, int], np.ndarray]:
    """Cut an (H, W, 3) image into (rows * cols, patch * patch * 3) patches.

    Returns:
        The patch grid as (rows, cols) and the patches, dtype unchanged.

    Raises:
        ValueError: A side is not a whole number of patches.
    """
    height, width, channels = image.shape
    if height % patch or width % patch:
        raise ValueError(
            f"an image of {height}x{width} is not a whole number of "
            f"{patch} pixel patches")
    rows, cols = height // patch, width // patch
    patches = (image.reshape(rows, patch, cols, patch, channels)
               .transpose(0, 2, 1, 3, 4)
               .reshape(rows * cols, patch * patch * channels))
    return (rows, cols), np.ascontiguousarray(patches)


def patch_positions(grid: tuple[int, int]) -> np.ndarray:
    """The (column, row) position of each patch, in patch order."""
    rows, cols = grid
    ys, xs = np.divmod(np.arange(rows * cols), cols)
    return np.stack([xs, ys], axis=-1).astype(np.int64)
