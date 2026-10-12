import os

import pytest
from pdfs import write_pdf

from quail.catalog import ScanRequest
from quail.logical import CompileError
from quail.pdf import check_unchanged, read_pdfs, stored_text


@pytest.fixture
def folder(tmp_path):
    write_pdf(tmp_path / "b.pdf", ["second file"])
    write_pdf(tmp_path / "a.pdf", ["Hello world", "", "Third (page)"])
    return tmp_path


def test_one_row_per_page_in_path_then_page_order(folder):
    table = read_pdfs(str(folder / "*.pdf")).table()
    assert table.column_names == ["path", "page", "page_count", "document"]
    rows = [(os.path.basename(r["path"]), r["page"], r["page_count"])
            for r in table.to_pylist()]
    assert rows == [("a.pdf", 1, 3), ("a.pdf", 2, 3), ("a.pdf", 3, 3),
                    ("b.pdf", 1, 1)]
    assert table.column("document")[1].as_py()["page"] == 2


def test_one_row_per_file_covers_every_page(folder):
    provider = read_pdfs([str(folder / "a.pdf"), str(folder / "b.pdf")],
                         split_pages=False)
    assert provider.columns == ("path", "page_count", "document")
    assert provider.statistics().row_count == 2
    assert provider.pages()[0][1] == [1, 2, 3]
    assert provider.table().column("document")[0].as_py()["page"] is None


def test_scan_returns_requested_columns(folder):
    reader = read_pdfs(str(folder / "*.pdf")).scan(
        ScanRequest(columns=("page",), limit=2))
    assert reader.read_all().column("page").to_pylist() == [1, 2]


def test_stored_text_is_empty_for_a_page_without_text(folder):
    assert stored_text(str(folder / "a.pdf"), [1, 2, 3]) == [
        "Hello world", "", "Third (page)"]


def test_content_identity_changes_with_the_files(folder):
    first = read_pdfs(str(folder / "*.pdf"))
    assert first.content_identity() == read_pdfs(
        str(folder / "*.pdf")).content_identity()
    assert first.content_identity() != read_pdfs(
        str(folder / "*.pdf"), split_pages=False).content_identity()
    write_pdf(folder / "b.pdf", ["changed text"])
    with pytest.raises(CompileError, match="changed after"):
        check_unchanged(first.files[1])


def test_no_match_and_unreadable_files_are_refused(tmp_path):
    with pytest.raises(ValueError, match="no PDF file"):
        read_pdfs(str(tmp_path / "*.pdf"))
    (tmp_path / "bad.pdf").write_bytes(b"not a pdf")
    with pytest.raises(ValueError, match="cannot open"):
        read_pdfs(str(tmp_path / "bad.pdf"))


def test_pdfium_parser_joins_a_rows_pages_and_counts_empty_pages(folder):
    from quail.parsers import PdfiumParser

    provider = read_pdfs(str(folder / "*.pdf"), split_pages=False)
    rows = [(row, *provider.pages()[row]) for row in range(len(provider.pages()))]
    parser = PdfiumParser(workers=1)
    try:
        texts = dict(parser.parse(rows))
    finally:
        parser.close()
    assert texts == {0: "Hello world\n\n\n\nThird (page)", 1: "second file"}
    assert parser.stats["pages_parsed"] == 4
    assert parser.stats["parse_empty_pages"] == 1
