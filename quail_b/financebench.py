"""The FinanceBench filing set: SEC filings as PDFs, questions with evidence.

FinanceBench (Islam et al., 2023) holds 150 open questions an analyst
would ask about a public company, each answered from one SEC filing
(10-K, 10-Q, 8-K, or earnings release) with the page or pages that
hold the evidence marked. QUAIL-B publishes two relations over the
sample. `filing_questions` has one row per question, with its filing
and its `evidence_pages`. `filing_pages` has one row per page of every
sampled question's filing, with a `document` file reference,
`files/<filing>.pdf#page=<n>` (RFC 8118), that an engine renders. The
reference labels of the page join come from `evidence_pages`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa

FINANCEBENCH_REPOSITORY = "patronus-ai/financebench"
FINANCEBENCH_COMMIT = "cc39aeb4afdf33909ee1412188bf89035950c2eb"
_RAW = (f"https://raw.githubusercontent.com/{FINANCEBENCH_REPOSITORY}/"
        f"{FINANCEBENCH_COMMIT}")
QUESTIONS_URL = f"{_RAW}/data/financebench_open_source.jsonl"
QUESTIONS = 150
FILES_DIR = "files"

QUESTION_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("financebench_id", pa.string()),
    ("filing", pa.string()),
    ("doc_name", pa.string()),
    ("company", pa.string()),
    ("question_type", pa.string()),
    ("question", pa.string()),
    ("answer", pa.string()),
    ("evidence_pages", pa.list_(pa.int32())),
])
PAGE_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("filing", pa.string()),
    ("doc_name", pa.string()),
    ("page_number", pa.int32()),
    ("page_count", pa.int32()),
    ("pdf_sha256", pa.string()),
    ("document", pa.string()),
])


def pdf_url(doc_name: str) -> str:
    """Where one filing's PDF is read at the pinned commit."""
    return f"{_RAW}/pdfs/{doc_name}.pdf"


def read_questions(text: str) -> list[dict]:
    """Parse the open-source question file, in FinanceBench id order."""
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    for row in rows:
        for evidence in row["evidence"]:
            if evidence["doc_name"] != row["doc_name"]:
                raise ValueError(
                    f"{row['financebench_id']} cites evidence outside its "
                    f"filing")
    return sorted(rows, key=lambda row: row["financebench_id"])


def evidence_pages(row: dict) -> list[int]:
    """The one-based pages holding a question's evidence, in page order.

    The source file counts pages from zero.
    """
    return sorted({int(evidence["evidence_page_num"]) + 1
                   for evidence in row["evidence"]})


def _page_count(pdf_bytes: bytes) -> int:
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(pdf_bytes)
    try:
        return len(document)
    finally:
        document.close()


def build_filing_tables(questions: list[dict], read_pdf, files_dir: Path
                        ) -> tuple[pa.Table, pa.Table]:
    """Write the sampled questions' filings and return both relations.

    Args:
        questions: The sampled question rows, in row order.
        read_pdf: doc_name -> the filing's PDF bytes.
        files_dir: Where `<filing>.pdf` files are written.

    Returns:
        The `filing_questions` and `filing_pages` tables. Filings are
        numbered in first-use order over the questions.

    Raises:
        ValueError: A question's evidence page is past its filing's
            last page.
    """
    files_dir = Path(files_dir)
    files_dir.mkdir(parents=True, exist_ok=True)
    filings: dict[str, dict] = {}
    pages = []
    question_rows = []
    for index, row in enumerate(questions):
        doc_name = row["doc_name"]
        filing = filings.get(doc_name)
        if filing is None:
            filing_id = f"fl{len(filings)}"
            pdf = read_pdf(doc_name)
            (files_dir / f"{filing_id}.pdf").write_bytes(pdf)
            count = _page_count(pdf)
            digest = hashlib.sha256(pdf).hexdigest()
            filing = filings[doc_name] = {
                "id": filing_id, "page_count": count, "sha256": digest}
            for page in range(1, count + 1):
                pages.append({
                    "id": f"{filing_id}p{page}",
                    "filing": filing_id,
                    "doc_name": doc_name,
                    "page_number": page,
                    "page_count": count,
                    "pdf_sha256": digest,
                    "document": f"{FILES_DIR}/{filing_id}.pdf#page={page}",
                })
        evidence = evidence_pages(row)
        if evidence and evidence[-1] > filing["page_count"]:
            raise ValueError(
                f"{row['financebench_id']} cites page {evidence[-1]} of "
                f"{doc_name}, which has {filing['page_count']} pages")
        question_rows.append({
            "id": f"fq{index}",
            "financebench_id": row["financebench_id"],
            "filing": filing["id"],
            "doc_name": doc_name,
            "company": row["company"],
            "question_type": row["question_type"],
            "question": row["question"],
            "answer": str(row["answer"]),
            "evidence_pages": evidence,
        })
    return (pa.Table.from_pylist(question_rows, schema=QUESTION_SCHEMA),
            pa.Table.from_pylist(pages, schema=PAGE_SCHEMA))


def page_answers_question(page: dict, question: dict) -> bool:
    """The annotation's answer: the page is an evidence page of the question."""
    return (page["filing"] == question["filing"]
            and page["page_number"] in question["evidence_pages"])
