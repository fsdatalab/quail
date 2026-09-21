"""CPU checks for the CUAD contract tables, on a small synthetic archive."""

import ctypes
import hashlib
import io
import json
import zipfile

import pyarrow as pa
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
import pytest

from quail_b import cuad
from quail_b.cuad import (
    CuadArchive,
    build_contract_tables,
    parse_document_reference,
    resolve_documents,
    sample_titles,
    span_pages,
    verify_document_files,
)

CLAUSE = "The Distributor shall not compete with the Company in Europe."
CAP = "Liability is capped at the fees paid, except for breach of confidentiality."


def _pdf(pages: list[str]) -> bytes:
    """A PDF with one line of text per page, made with PDFium itself."""
    document = pdfium.PdfDocument.new()
    font = pdfium_c.FPDFText_LoadStandardFont(document.raw, b"Helvetica")
    for text in pages:
        page = document.new_page(612, 792)
        block = pdfium_c.FPDFPageObj_CreateTextObj(document.raw, font, 11)
        encoded = text.encode("utf-16-le") + b"\x00\x00"
        buffer = ctypes.create_string_buffer(encoded, len(encoded))
        pdfium_c.FPDFText_SetText(
            block, ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ushort)))
        pdfium_c.FPDFPageObj_Transform(block, 1, 0, 0, 1, 72, 720)
        pdfium_c.FPDFPage_InsertObject(page.raw, block)
        pdfium_c.FPDFPage_GenerateContent(page.raw)
    stream = io.BytesIO()
    document.save(stream)
    document.close()
    return stream.getvalue()


def _entry(title, spans):
    return {"title": title, "paragraphs": [{"context": "", "qas": [
        {"id": f"{title}__{category}", "is_impossible": not answers,
         "answers": [{"text": text, "answer_start": 0} for text in answers]}
        for category, answers in spans.items()]}]}


@pytest.fixture
def archive(tmp_path, monkeypatch):
    pdfs = {
        "Alpha - Distribution Agreement": _pdf([
            "ALPHA AGREEMENT page one.", CLAUSE, CAP]),
        "Beta - License Agreement_Option Exhibit": _pdf(
            ["BETA AGREEMENT single page."]),
    }
    entries = [
        _entry("Alpha - Distribution Agreement", {
            "Non-Compete": [CLAUSE], "Uncapped Liability": [CAP],
            "Audit Rights": []}),
        # the annotation title is a cut-off form of the PDF name
        _entry("Beta - License Agreement ", {
            "Non-Compete": [], "Uncapped Liability": [], "Audit Rights": []}),
    ]
    path = tmp_path / "CUAD_v1.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(cuad.CUAD_JSON_MEMBER, json.dumps({"data": entries}))
        for stem, pdf in pdfs.items():
            bundle.writestr(f"{cuad.CUAD_PDF_PREFIX}Part_I/{stem}.pdf", pdf)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(cuad, "CUAD_ARCHIVE_SHA256", digest)
    return CuadArchive(path)


def test_archive_rejects_other_content(tmp_path):
    path = tmp_path / "other.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("x", "y")
    with pytest.raises(ValueError, match="sha256"):
        CuadArchive(path)


def test_sample_is_nested_and_in_title_order():
    titles = [f"Contract {index:03d}" for index in range(40)]
    small = sample_titles(titles, 4, seed=1)
    large = sample_titles(titles, 10, seed=1)
    assert set(small) <= set(large)
    assert small == sorted(small) and large == sorted(large)
    assert sample_titles(titles, 4, seed=2) != small
    assert sample_titles(titles, 40, seed=1) == titles


def test_span_pages_marks_every_page_a_span_touches():
    pages = [cuad.normal_text(text) for text in
             ("first page text", "the clause starts here", "and ends here.")]
    touched = span_pages(pages, [
        ("Non-Compete", "clause starts here and ends"),
        ("Audit Rights", "first page"),
        ("Missing", "not in the contract"),
    ])
    assert touched == {"Non-Compete": {1, 2}, "Audit Rights": {0}}


def test_contract_tables_reference_files_and_list_clauses_per_page(
        archive, tmp_path):
    titles = sample_titles(
        [entry["title"] for entry in archive.entries()], 2, seed=1)
    contracts, pages = build_contract_tables(
        archive, titles, tmp_path / "files")
    assert contracts.schema == cuad.CONTRACT_SCHEMA
    assert pages.schema == cuad.PAGE_SCHEMA
    rows = {row["title"]: row for row in contracts.to_pylist()}
    alpha = rows["Alpha - Distribution Agreement"]
    assert alpha["page_count"] == 3
    assert alpha["clauses"] == ["Non-Compete", "Uncapped Liability"]
    assert alpha["document"] == f"files/{alpha['id']}.pdf"
    written = (tmp_path / "files" / f"{alpha['id']}.pdf").read_bytes()
    assert hashlib.sha256(written).hexdigest() == alpha["pdf_sha256"]
    beta = rows["Beta - License Agreement "]
    assert beta["page_count"] == 1 and beta["clauses"] == []
    by_page = {
        (row["contract_id"], row["page_number"]): row
        for row in pages.to_pylist()}
    assert len(by_page) == 4
    assert by_page[(alpha["id"], 1)]["clauses"] == []
    assert by_page[(alpha["id"], 2)]["clauses"] == ["Non-Compete"]
    assert by_page[(alpha["id"], 3)]["clauses"] == ["Uncapped Liability"]
    assert by_page[(alpha["id"], 2)]["id"] == f"{alpha['id']}p2"
    assert by_page[(alpha["id"], 2)]["document"] == (
        f"files/{alpha['id']}.pdf#page=2")
    assert by_page[(alpha["id"], 2)]["pdf_sha256"] == alpha["pdf_sha256"]

    verify_document_files(contracts, tmp_path)
    verify_document_files(pages, tmp_path)
    resolved = resolve_documents(pages, tmp_path)
    reference = resolved.column("document")[0].as_py()
    path, page = parse_document_reference(reference)
    assert path == str(tmp_path / "files" / f"{alpha['id']}.pdf") and page == 1
    (tmp_path / "files" / f"{alpha['id']}.pdf").write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        verify_document_files(contracts, tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_documents(contracts, tmp_path / "elsewhere")


def test_document_reference_parsing():
    assert parse_document_reference("files/ct0.pdf") == ("files/ct0.pdf", None)
    assert parse_document_reference("files/ct0.pdf#page=12") == (
        "files/ct0.pdf", 12)
    with pytest.raises(ValueError):
        parse_document_reference("files/ct0.pdf#page=0")
    with pytest.raises(ValueError):
        parse_document_reference("files/ct0.pdf#other")
    assert resolve_documents(pa.table({"id": ["a"]}), "/nowhere").num_rows == 1
