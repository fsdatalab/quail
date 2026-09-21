"""The OfficeQA Pro v2 statement set: Treasury statements as PDFs, with pages.

OfficeQA Pro v2 (Databricks) holds 90 questions over the U.S.
Treasury's statements of receipts and expenditures, 1793 to 2024: the
Combined Statements of Receipts, Outlays, and Balances and the earlier
receipts documents on govinfo. Every question names the document or
documents it reads and the PDF page of each that holds the answer.
QUAIL-B publishes two relations over the sample. `treasury_questions`
has one row per question and source document, with the document's
`statement` id and its `evidence_pages`; a question over three
documents is three rows. `treasury_pages` has one row per page of
every sampled document, with a `document` file reference,
`files/<statement>.pdf#page=<n>` (RFC 8118), that an engine renders.
The reference labels of the page join come from `evidence_pages`.

The dataset is gated on Hugging Face: a reader accepts its terms at
https://huggingface.co/datasets/databricks/officeqa-pro-v2 and passes
the account's token as `HF_TOKEN`. Its `source_docs` column records
`pdf_page_number=<n>` per document; this module reads it as a one
based PDF page number, the number a viewer shows, which the build
checks against each document's page count.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from pathlib import Path

import pyarrow as pa

OFFICEQA_REPOSITORY = "databricks/officeqa-pro-v2"
OFFICEQA_COMMIT = "65a2b315780417bc50d7bfe6e5bdb904e63fda65"
OFFICEQA_TERMS_URL = f"https://huggingface.co/datasets/{OFFICEQA_REPOSITORY}"
_RESOLVE = f"{OFFICEQA_TERMS_URL}/resolve/{OFFICEQA_COMMIT}"
QUESTIONS_FILE = "officeqa_pro_v2.csv"
QUESTIONS_URL = f"{_RESOLVE}/{QUESTIONS_FILE}"
QUESTIONS = 90
FILES_DIR = "files"

QUESTION_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("officeqa_uid", pa.string()),
    ("statement", pa.string()),
    ("doc_name", pa.string()),
    ("source_count", pa.int32()),
    ("question", pa.string()),
    ("answer", pa.string()),
    ("evidence_pages", pa.list_(pa.int32())),
])
PAGE_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("statement", pa.string()),
    ("doc_name", pa.string()),
    ("page_number", pa.int32()),
    ("page_count", pa.int32()),
    ("pdf_sha256", pa.string()),
    ("document", pa.string()),
])

# One record per source document; a description may hold ";" or "|",
# so records split only where the next one's first field begins.
_RECORD_START = re.compile(r";\s*(?=corpus_file=)")
_CORPUS_FILE = re.compile(r"corpus_file=\s*([^|;]+?)\s*(?:\||$)")
_PAGE_NUMBER = re.compile(r"pdf_page_number=\s*(\d+)")


def pdf_url(doc_name: str) -> str:
    """Where one document's PDF is read at the pinned commit."""
    return f"{_RESOLVE}/pdfs/{doc_name}.pdf"


def parse_source_docs(text: str) -> list[tuple[str, int]]:
    """The (document basename, one based page) pairs of one `source_docs`.

    Raises:
        ValueError: A record names no corpus file or no page.
    """
    sources = []
    for record in _RECORD_START.split(text.strip()):
        if not record.strip():
            continue
        corpus_file = _CORPUS_FILE.search(record)
        page = _PAGE_NUMBER.search(record)
        if corpus_file is None or page is None:
            raise ValueError(
                f"source record names no corpus file and page: {record!r}")
        doc_name = corpus_file.group(1).strip()
        if doc_name.endswith(".txt"):
            doc_name = doc_name[:-len(".txt")]
        sources.append((doc_name, int(page.group(1))))
    return sources


def read_questions(text: str) -> list[dict]:
    """Parse the question file, in uid order, with `sources` per row.

    Raises:
        ValueError: A question's `source_docs` names a document its
            `source_files` does not.
    """
    rows = list(csv.DictReader(io.StringIO(text)))
    for row in rows:
        row["sources"] = parse_source_docs(row["source_docs"])
        for doc_name, _page in row["sources"]:
            if doc_name not in row["source_files"]:
                raise ValueError(
                    f"{row['uid']} reads {doc_name} in source_docs but not "
                    f"in source_files")
    return sorted(rows, key=lambda row: _uid_order(row["uid"]))


def _uid_order(uid: str) -> tuple[int, str]:
    number = re.search(r"(\d+)$", uid)
    return (int(number.group(1)) if number else -1, uid)


def evidence_by_document(row: dict) -> dict[str, list[int]]:
    """Document basename -> the one based pages of it the question reads.

    Documents keep the order they first appear in `source_docs`.
    """
    pages: dict[str, set[int]] = {}
    for doc_name, page in row["sources"]:
        pages.setdefault(doc_name, set()).add(page)
    return {doc_name: sorted(found) for doc_name, found in pages.items()}


def _page_count(pdf_bytes: bytes) -> int:
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(pdf_bytes)
    try:
        return len(document)
    finally:
        document.close()


def build_statement_tables(questions: list[dict], read_pdf, files_dir: Path
                           ) -> tuple[pa.Table, pa.Table]:
    """Write the sampled questions' documents and return both relations.

    Args:
        questions: The sampled question rows, in row order.
        read_pdf: doc_name -> the document's PDF bytes.
        files_dir: Where `<statement>.pdf` files are written.

    Returns:
        The `treasury_questions` and `treasury_pages` tables. Documents
        are numbered in first-use order over the questions, and a
        question has one row per document it reads.

    Raises:
        ValueError: A question's evidence page is past its document's
            last page.
    """
    files_dir = Path(files_dir)
    files_dir.mkdir(parents=True, exist_ok=True)
    statements: dict[str, dict] = {}
    pages = []
    question_rows = []
    for row in questions:
        evidence = evidence_by_document(row)
        for doc_name, evidence_pages in evidence.items():
            statement = statements.get(doc_name)
            if statement is None:
                statement_id = f"ts{len(statements)}"
                pdf = read_pdf(doc_name)
                (files_dir / f"{statement_id}.pdf").write_bytes(pdf)
                count = _page_count(pdf)
                digest = hashlib.sha256(pdf).hexdigest()
                statement = statements[doc_name] = {
                    "id": statement_id, "page_count": count, "sha256": digest}
                for page in range(1, count + 1):
                    pages.append({
                        "id": f"{statement_id}p{page}",
                        "statement": statement_id,
                        "doc_name": doc_name,
                        "page_number": page,
                        "page_count": count,
                        "pdf_sha256": digest,
                        "document": f"{FILES_DIR}/{statement_id}.pdf#page={page}",
                    })
            if evidence_pages[-1] > statement["page_count"]:
                raise ValueError(
                    f"{row['uid']} cites page {evidence_pages[-1]} of "
                    f"{doc_name}, which has {statement['page_count']} pages")
            question_rows.append({
                "id": f"tq{len(question_rows)}",
                "officeqa_uid": row["uid"],
                "statement": statement["id"],
                "doc_name": doc_name,
                "source_count": len(evidence),
                "question": row["question"],
                "answer": str(row["answer"]),
                "evidence_pages": evidence_pages,
            })
    return (pa.Table.from_pylist(question_rows, schema=QUESTION_SCHEMA),
            pa.Table.from_pylist(pages, schema=PAGE_SCHEMA))


def page_answers_question(page: dict, question: dict) -> bool:
    """The annotation's answer: the page is an evidence page of the question."""
    return (page["statement"] == question["statement"]
            and page["page_number"] in question["evidence_pages"])
