"""PDF page rendering and the page prefetcher, on the CPU.

The pages are drawn with PDFium from small PDFs the tests write, so the
pixel layout, the patch grid, and the lookahead bound are all real.
"""

import os

import numpy as np
import pytest

from quail.pdf import (
    PDFInput,
    PdfiumPrefetcher,
    PdfReadError,
    read_manifest,
    rows_for_mode,
)
from quail.pdf.render import patch_positions, patchify, render_page
from quail.specs import DIFFUSION_GEMMA_26B_FP8
from quail.specs.vision import render_size, soft_tokens

pypdfium2 = pytest.importorskip("pypdfium2")
pdfium_c = pytest.importorskip("pypdfium2.raw")

GEMMA = DIFFUSION_GEMMA_26B_FP8
PATCH = GEMMA.image_patch_pixels
POOL = GEMMA.image_pool_kernel
LETTER = (612, 792)


def make_pdf(path, sizes, *, left_half_black=False):
    """Write a PDF with one page per (width, height) in points.

    With left_half_black, every page gets a black rectangle over its
    left half so a render can be told from a blank page.
    """
    document = pypdfium2.PdfDocument.new()
    for width, height in sizes:
        page = document.new_page(width, height)
        if left_half_black:
            rect = pdfium_c.FPDFPageObj_CreateNewRect(0, 0, width / 2, height)
            pdfium_c.FPDFPageObj_SetFillColor(rect, 0, 0, 0, 255)
            pdfium_c.FPDFPath_SetDrawMode(rect, pdfium_c.FPDF_FILLMODE_WINDING,
                                          False)
            pdfium_c.FPDFPage_InsertObject(page.raw, rect)
            pdfium_c.FPDFPage_GenerateContent(page.raw)
    document.save(str(path))
    document.close()
    return str(path)


def pdf_input(paths, row_mode="page", budget=280):
    manifest = read_manifest(paths)
    rows = rows_for_mode(manifest.pages, len(manifest.sources), row_mode)
    return PDFInput(sources=manifest.sources, pages=manifest.pages, rows=rows,
                    row_mode=row_mode, visual_tokens=budget)


def test_render_page_draws_at_the_target_size(tmp_path):
    path = make_pdf(tmp_path / "half.pdf", [LETTER], left_half_black=True)
    size_hw = render_size(GEMMA, *LETTER, 280)
    document = pypdfium2.PdfDocument(path)
    image = render_page(document, 0, size_hw)
    document.close()
    assert image.shape == (*size_hw, 3)
    assert image.dtype == np.uint8
    height, width = size_hw
    assert image[:, : width // 2 - 2].max() == 0
    assert image[:, width // 2 + 2:].min() == 255


def test_patchify_keeps_every_pixel_in_processor_order():
    rows, cols = 3, 5
    image = np.arange(rows * PATCH * cols * PATCH * 3, dtype=np.int32)
    image = image.reshape(rows * PATCH, cols * PATCH, 3)
    grid, patches = patchify(image, PATCH)
    assert grid == (rows, cols)
    assert patches.shape == (rows * cols, PATCH * PATCH * 3)
    # patch k covers block (k // cols, k % cols); its pixels are row-major
    k = 7
    block = image[(k // cols) * PATCH:(k // cols + 1) * PATCH,
                  (k % cols) * PATCH:(k % cols + 1) * PATCH]
    np.testing.assert_array_equal(patches[k], block.reshape(-1))
    positions = patch_positions(grid)
    assert positions.shape == (rows * cols, 2)
    assert tuple(positions[k]) == (k % cols, k // cols)
    with pytest.raises(ValueError, match="whole number"):
        patchify(image[1:], PATCH)


def test_prefetcher_returns_rows_in_order_with_the_planned_grid(tmp_path):
    paths = [make_pdf(tmp_path / "a.pdf", [LETTER] * 3, left_half_black=True),
             make_pdf(tmp_path / "b.pdf", [LETTER, (792, 612)])]
    inputs = pdf_input(paths)
    prefetcher = PdfiumPrefetcher(inputs, GEMMA, processes=0)
    try:
        pages = []
        for row in range(len(inputs.rows)):
            (page,) = prefetcher.take(row)
            ref = inputs.pages[page.page_id]
            assert page.page_id == inputs.rows[row].page_ids[0]
            rows, cols = page.grid
            assert rows * cols // (POOL * POOL) == soft_tokens(
                GEMMA, ref.width_points, ref.height_points, 280)
            assert page.patches.shape == (rows * cols, PATCH * PATCH * 3)
            assert page.patches.dtype == np.uint8
            pages.append(page)
        # a.pdf's pages are black on the left, white on the right;
        # b.pdf's are blank; the landscape page is wider than tall
        assert pages[0].patches[0].max() == 0
        assert pages[0].patches[pages[0].grid[1] - 1].min() == 255
        assert pages[3].patches.min() == 255
        assert pages[4].grid[0] < pages[4].grid[1]
        with pytest.raises(ValueError, match="already taken"):
            prefetcher.take(0)
        metrics = prefetcher.metrics()
        assert metrics["pages_rendered"] == 5
        assert metrics["render_processes"] == 0
        assert metrics["render_worker_s"] > 0
    finally:
        prefetcher.close()
    assert prefetcher.outstanding_pages == 0


def test_prefetcher_pdf_mode_hands_over_every_page_of_a_row(tmp_path):
    paths = [make_pdf(tmp_path / "a.pdf", [LETTER] * 3),
             make_pdf(tmp_path / "b.pdf", [LETTER, (792, 612)])]
    inputs = pdf_input(paths, row_mode="pdf")
    prefetcher = PdfiumPrefetcher(inputs, GEMMA, processes=0, order=[1, 0])
    try:
        second = prefetcher.take(1)
        assert [page.page_id for page in second] == [3, 4]
        first = prefetcher.take(0)
        assert [page.page_id for page in first] == [0, 1, 2]
    finally:
        prefetcher.close()
    with pytest.raises(ValueError, match="every row index once"):
        PdfiumPrefetcher(inputs, GEMMA, processes=0, order=[0, 0])


def test_prefetcher_lookahead_stays_under_the_page_bound(tmp_path):
    paths = [make_pdf(tmp_path / "a.pdf", [LETTER] * 6)]
    inputs = pdf_input(paths)
    prefetcher = PdfiumPrefetcher(inputs, GEMMA, processes=0,
                                  max_outstanding_pages=2)
    try:
        assert prefetcher.outstanding_pages == 2
        # a row past the lookahead is rendered on demand, and the
        # lookahead moves on from there
        prefetcher.take(5)
        assert prefetcher.outstanding_pages == 2
        prefetcher.take(0)
        assert prefetcher.outstanding_pages == 2
        for row in (1, 2, 3, 4):
            prefetcher.take(row)
        assert prefetcher.outstanding_pages == 0
        assert prefetcher.metrics()["pages_rendered"] == 6
    finally:
        prefetcher.close()


def test_prefetcher_refuses_a_source_edited_after_planning(tmp_path):
    path = make_pdf(tmp_path / "a.pdf", [LETTER])
    inputs = pdf_input([path])
    make_pdf(path, [LETTER, LETTER])
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, inputs.sources[0].mtime_ns + 10**9))
    prefetcher = PdfiumPrefetcher(inputs, GEMMA, processes=0)
    try:
        with pytest.raises(PdfReadError, match="changed after planning"):
            prefetcher.take(0)
    finally:
        prefetcher.close()


def test_prefetcher_reopens_a_file_registered_again(tmp_path):
    path = make_pdf(tmp_path / "a.pdf", [LETTER])
    before = pdf_input([path])
    prefetcher = PdfiumPrefetcher(before, GEMMA, processes=0)
    (page,) = prefetcher.take(0)
    prefetcher.close()
    assert page.patches.min() == 255
    # the same path now holds a different PDF with a new manifest record
    make_pdf(path, [LETTER], left_half_black=True)
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, before.sources[0].mtime_ns + 10**9))
    after = pdf_input([path])
    assert after.sources[0] != before.sources[0]
    prefetcher = PdfiumPrefetcher(after, GEMMA, processes=0)
    (page,) = prefetcher.take(0)
    prefetcher.close()
    assert page.patches[0].max() == 0


def test_prefetcher_renders_in_spawned_processes(tmp_path):
    paths = [make_pdf(tmp_path / "a.pdf", [LETTER, (792, 612)],
                      left_half_black=True)]
    inputs = pdf_input(paths)
    prefetcher = PdfiumPrefetcher(inputs, GEMMA, processes=1)
    try:
        (portrait,) = prefetcher.take(0)
        (landscape,) = prefetcher.take(1)
    finally:
        prefetcher.close()
    assert portrait.grid == (912 // PATCH, 672 // PATCH)
    assert landscape.grid == (672 // PATCH, 912 // PATCH)
    # left half black, right half white, in patch order: the first
    # patch of a row is black, the last is white
    cols = portrait.grid[1]
    assert portrait.patches[0].max() == 0
    assert portrait.patches[cols - 1].min() == 255
    assert prefetcher.metrics()["render_processes"] == 1
