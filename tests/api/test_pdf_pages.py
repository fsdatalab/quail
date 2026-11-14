"""AI.PDF_PAGES: each page of the PDFs in a binary column, as a PNG."""

import struct

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, oracle_executor
from helpers.pdfs import pdf
from helpers.sessions import CONFIG, TOKENIZER

import quail

FILES = pa.table({
    "id": ["f1", "f2", "f3"],
    "year": [2025, 2024, 2025],
    "pdf": pa.array([pdf("Lease from March 2025.", "Rent is $2,400 a month."),
                     pdf("Invoice for two chairs."),
                     None], pa.binary()),
})


@pytest.fixture()
def context():
    with quail.Session(CONFIG, tokenizer=TOKENIZER) as session:
        oracle = Oracle()
        oracle.catalog = session.catalog
        context = quail.SessionContext(
            session, physical_executor=oracle_executor(oracle))
        context.register(
            "files", quail.DocumentProvider.from_table(FILES, id_col="id"))
        yield context


def _png_size(image: bytes) -> tuple[int, int]:
    """Return the width and height of an 8-bit RGB PNG, from its header."""
    assert image[:8] == b"\x89PNG\r\n\x1a\n"
    assert image[12:16] == b"IHDR"
    width, height, depth, color = struct.unpack(">IIBB", image[16:26])
    assert (depth, color) == (8, 2)
    return width, height


def test_an_unnest_returns_one_row_for_each_page(context):
    table = context.sql(
        "SELECT p.id AS file, p.u['page'] AS page, p.u['image'] AS image "
        "FROM (SELECT f.id, UNNEST(AI.PDF_PAGES(f.pdf)) AS u FROM files f) p "
        "ORDER BY file, page")
    # f3, a NULL PDF, has no page
    assert table.column("file").to_pylist() == ["f1", "f1", "f2"]
    assert table.column("page").to_pylist() == [1, 2, 1]
    # a US Letter page, 612 by 792 points, at the default longest_edge
    assert [_png_size(image) for image in table.column("image").to_pylist()] == [
        (1184, 1536)] * 3


def test_a_field_of_an_unnest_needs_no_subquery(context):
    table = context.sql(
        "SELECT f.id AS file, UNNEST(AI.PDF_PAGES(f.pdf))['page'] AS page "
        "FROM files f WHERE f.year = 2025")
    assert table.to_pylist() == [{"file": "f1", "page": 1}, {"file": "f1", "page": 2}]


def test_a_longest_edge_sets_the_longer_side(context):
    # Snowflake's name for the function
    table = context.sql(
        "SELECT array_element(AI_PDF_PAGES(f.pdf, 512), 1)['image'] AS image "
        "FROM files f WHERE f.id = 'f2'")
    assert [_png_size(image) for image in table.column("image").to_pylist()] == [
        (384, 512)]


def test_an_unreadable_or_null_pdf_has_null_pages(context):
    context.register("broken", quail.DocumentProvider.from_table(
        pa.table({"id": ["f1", "f2"],
                  "pdf": pa.array([b"not a pdf", None], pa.binary())}), id_col="id"))
    table = context.sql("SELECT AI.PDF_PAGES(b.pdf) AS pages FROM broken b")
    assert table.column("pages").to_pylist() == [None, None]


def test_a_longest_edge_that_is_not_a_multiple_of_32_is_refused(context):
    with pytest.raises(RuntimeError, match="multiple of 32 from 32 to 8192 pixels, "
                                           "not 1000"):
        context.sql("SELECT AI.PDF_PAGES(f.pdf, 1000) FROM files f")
    with pytest.raises(RuntimeError, match="longest_edge as an integer literal"):
        context.sql("SELECT AI.PDF_PAGES(f.pdf, f.year) FROM files f")


def test_pages_of_text_are_refused(context):
    with pytest.raises(RuntimeError, match="reads a binary column of PDFs, not Utf8"):
        context.sql("SELECT AI.PDF_PAGES(f.id) FROM files f")
