"""CPU checks for the FinanceBench filing tables, on synthetic filings."""

import hashlib
import json

import pytest
from test_cuad import _pdf

from quail_b import financebench
from quail_b.financebench import (
    build_filing_tables,
    evidence_pages,
    page_answers_question,
    read_questions,
)


def _question(number, doc_name, pages, **fields):
    return {
        "financebench_id": f"financebench_id_{number:05d}",
        "company": doc_name.split("_")[0],
        "doc_name": doc_name,
        "question_type": "metrics-generated",
        "question": f"What is figure {number}?",
        "answer": number * 1.5,
        "evidence": [{"doc_name": doc_name, "evidence_page_num": page}
                     for page in pages],
        **fields,
    }


def test_questions_read_in_id_order_and_count_pages_from_one():
    rows = [_question(2, "ACME_2020_10K", [3, 3, 0]),
            _question(1, "ACME_2020_10K", [1])]
    text = "\n".join(json.dumps(row) for row in rows) + "\n"
    parsed = read_questions(text)
    assert [row["financebench_id"] for row in parsed] == [
        "financebench_id_00001", "financebench_id_00002"]
    assert evidence_pages(parsed[1]) == [1, 4]
    stray = _question(3, "ACME_2020_10K", [0])
    stray["evidence"][0]["doc_name"] = "OTHER_2020_10K"
    with pytest.raises(ValueError, match="outside its filing"):
        read_questions(json.dumps(stray))


def test_filing_tables_number_filings_by_first_use_and_list_every_page(
        tmp_path):
    pdfs = {"ACME_2020_10K": _pdf(["cover", "cash flow", "notes"]),
            "BOLT_2021_10Q": _pdf(["cover", "balance sheet"])}
    questions = [_question(1, "BOLT_2021_10Q", [1]),
                 _question(2, "ACME_2020_10K", [1, 2]),
                 _question(3, "BOLT_2021_10Q", [0])]
    read = []

    def read_pdf(doc_name):
        read.append(doc_name)
        return pdfs[doc_name]

    table, pages = build_filing_tables(questions, read_pdf, tmp_path / "files")
    assert read == ["BOLT_2021_10Q", "ACME_2020_10K"]
    rows = table.to_pylist()
    assert [(r["id"], r["filing"], r["evidence_pages"]) for r in rows] == [
        ("fq0", "fl0", [2]), ("fq1", "fl1", [2, 3]), ("fq2", "fl0", [1])]
    assert rows[1]["answer"] == "3.0" and rows[1]["company"] == "ACME"
    page_rows = pages.to_pylist()
    assert [(p["id"], p["filing"], p["page_number"], p["page_count"])
            for p in page_rows] == [
        ("fl0p1", "fl0", 1, 2), ("fl0p2", "fl0", 2, 2),
        ("fl1p1", "fl1", 1, 3), ("fl1p2", "fl1", 2, 3), ("fl1p3", "fl1", 3, 3)]
    assert page_rows[2]["document"] == "files/fl1.pdf#page=1"
    written = (tmp_path / "files" / "fl1.pdf").read_bytes()
    assert written == pdfs["ACME_2020_10K"]
    assert page_rows[2]["pdf_sha256"] == hashlib.sha256(written).hexdigest()
    assert page_rows[2]["doc_name"] == "ACME_2020_10K"
    # the evidence pages label the join
    truth = [[page_answers_question(page, question)
              for page in page_rows] for question in rows]
    assert truth == [[False, True, False, False, False],
                     [False, False, False, True, True],
                     [True, False, False, False, False]]
    with pytest.raises(ValueError, match="cites page 9"):
        build_filing_tables([_question(4, "BOLT_2021_10Q", [8])], read_pdf,
                            tmp_path / "more")


def test_pinned_source_urls_name_the_commit():
    assert financebench.FINANCEBENCH_COMMIT in financebench.QUESTIONS_URL
    assert financebench.pdf_url("3M_2018_10K").endswith(
        f"/{financebench.FINANCEBENCH_COMMIT}/pdfs/3M_2018_10K.pdf")
