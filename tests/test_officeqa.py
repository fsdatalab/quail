"""CPU checks for the OfficeQA statement tables, on synthetic statements.

The dataset itself is gated, so these run on rows in its documented
shape and never reach Hugging Face.
"""

import csv
import hashlib
import io

import pytest
from test_cuad import _pdf

from quail_b import officeqa
from quail_b.officeqa import (
    build_statement_tables,
    evidence_by_document,
    page_answers_question,
    parse_source_docs,
    read_questions,
)

HISTORICAL = "combined_statement__historical__cs-1872"
MODERN = "combined_statement__modern__2003__outlays"


def _record(doc_name, page, description="N/A"):
    return (f"corpus_file={doc_name}.txt | pdf_page_number={page} | "
            f"year=1872 | month=N/A | description={description}")


def _csv(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(
        stream, ["uid", "question", "answer", "source_docs", "source_files"])
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _question(number, sources, **fields):
    docs = list(dict.fromkeys(doc for doc, _page in sources))
    return {
        "uid": f"qid_{number}",
        "question": f"What was figure {number}?",
        "answer": f"{number}.5",
        "source_docs": "; ".join(_record(doc, page) for doc, page in sources),
        "source_files": "; ".join(docs),
        **fields,
    }


def test_source_docs_parse_one_record_per_document_even_with_punctuation():
    text = "; ".join([
        _record(HISTORICAL, 12, "receipts; by source | 1871-72"),
        _record(MODERN, 3),
        _record(HISTORICAL, 14),
    ])
    assert parse_source_docs(text) == [(HISTORICAL, 12), (MODERN, 3),
                                       (HISTORICAL, 14)]
    row = {"sources": parse_source_docs(text)}
    assert evidence_by_document(row) == {HISTORICAL: [12, 14], MODERN: [3]}
    with pytest.raises(ValueError, match="names no corpus file and page"):
        parse_source_docs(f"corpus_file={MODERN}.txt | year=2003")


def test_questions_read_in_uid_order_and_check_their_source_files():
    text = _csv([_question(10, [(MODERN, 3)]),
                 _question(2, [(HISTORICAL, 12), (HISTORICAL, 14)])])
    rows = read_questions(text)
    assert [row["uid"] for row in rows] == ["qid_2", "qid_10"]
    assert rows[0]["sources"] == [(HISTORICAL, 12), (HISTORICAL, 14)]
    stray = _question(3, [(MODERN, 3)], source_files=HISTORICAL)
    with pytest.raises(ValueError, match="not in source_files"):
        read_questions(_csv([stray]))


def test_statement_tables_give_a_question_one_row_per_document(tmp_path):
    pdfs = {HISTORICAL: _pdf(["title", "receipts", "expenditures"]),
            MODERN: _pdf(["cover", "outlays"])}
    questions = read_questions(_csv([
        _question(1, [(MODERN, 2)]),
        _question(2, [(HISTORICAL, 2), (MODERN, 1), (HISTORICAL, 3)]),
        _question(3, [(MODERN, 1)]),
    ]))
    read = []

    def read_pdf(doc_name):
        read.append(doc_name)
        return pdfs[doc_name]

    table, pages = build_statement_tables(questions, read_pdf, tmp_path / "files")
    assert read == [MODERN, HISTORICAL]
    rows = table.to_pylist()
    assert [(r["id"], r["officeqa_uid"], r["statement"], r["source_count"],
             r["evidence_pages"]) for r in rows] == [
        ("tq0", "qid_1", "ts0", 1, [2]),
        ("tq1", "qid_2", "ts1", 2, [2, 3]),
        ("tq2", "qid_2", "ts0", 2, [1]),
        ("tq3", "qid_3", "ts0", 1, [1])]
    assert rows[1]["answer"] == "2.5" and rows[1]["doc_name"] == HISTORICAL
    page_rows = pages.to_pylist()
    assert [(p["id"], p["statement"], p["page_number"], p["page_count"])
            for p in page_rows] == [
        ("ts0p1", "ts0", 1, 2), ("ts0p2", "ts0", 2, 2),
        ("ts1p1", "ts1", 1, 3), ("ts1p2", "ts1", 2, 3), ("ts1p3", "ts1", 3, 3)]
    assert page_rows[2]["document"] == "files/ts1.pdf#page=1"
    written = (tmp_path / "files" / "ts1.pdf").read_bytes()
    assert written == pdfs[HISTORICAL]
    assert page_rows[2]["pdf_sha256"] == hashlib.sha256(written).hexdigest()
    # the named pages label the join, statement by statement
    truth = [[page_answers_question(page, question)
              for page in page_rows] for question in rows]
    assert truth == [[False, True, False, False, False],
                     [False, False, False, True, True],
                     [True, False, False, False, False],
                     [True, False, False, False, False]]
    with pytest.raises(ValueError, match="cites page 9"):
        build_statement_tables(
            read_questions(_csv([_question(4, [(MODERN, 9)])])), read_pdf,
            tmp_path / "more")


def test_pinned_source_urls_name_the_commit():
    assert officeqa.OFFICEQA_COMMIT in officeqa.QUESTIONS_URL
    assert officeqa.pdf_url(HISTORICAL).endswith(
        f"/{officeqa.OFFICEQA_COMMIT}/pdfs/{HISTORICAL}.pdf")


def test_building_the_statements_without_a_token_names_the_gate(
        tmp_path, monkeypatch):
    from quail_b._files import download_cache
    from quail_b.data import _officeqa_source

    monkeypatch.delenv("HF_TOKEN", raising=False)
    with download_cache(tmp_path), pytest.raises(
            PermissionError, match="accept its terms"):
        _officeqa_source(officeqa.QUESTIONS_FILE)
