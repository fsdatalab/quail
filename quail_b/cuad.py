"""The CUAD contract set: PDF files, one row per contract and per page.

CUAD v1 (Hendrycks et al., 2021) holds 510 commercial contracts as
PDFs with lawyer-annotated clause spans in 41 categories. QUAIL-B
publishes two relations over them. `contracts` has one row per PDF
and `contract_pages` one row per page. Both carry a `document`
column that is a file reference, `files/<id>.pdf` for a contract and
`files/<id>.pdf#page=<n>` for one page (RFC 8118); an engine renders
the referenced pages. The `clauses` column lists the categories with
an annotated span in the contract, or with a span that touches the
page. The reference labels are derived from it.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import zipfile
from bisect import bisect_left, bisect_right
from pathlib import Path

import pyarrow as pa

from quail_b._sampling import stable_sample

CUAD_ARCHIVE_URL = (
    "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1"
)
CUAD_ARCHIVE_SHA256 = (
    "88b694d99007d39777fa44cd72daf8297773d285dc3eab0091ba32078888d18e"
)
CUAD_JSON_MEMBER = "CUAD_v1/CUAD_v1.json"
CUAD_PDF_PREFIX = "CUAD_v1/full_contract_pdf/"
CONTRACTS = 510
FILES_DIR = "files"

CONTRACT_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("title", pa.string()),
    ("page_count", pa.int32()),
    ("pdf_sha256", pa.string()),
    ("document", pa.string()),
    ("clauses", pa.list_(pa.string())),
])
PAGE_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("contract_id", pa.string()),
    ("page_number", pa.int32()),
    ("pdf_sha256", pa.string()),
    ("document", pa.string()),
    ("clauses", pa.list_(pa.string())),
])


def normal_text(text: str) -> str:
    """Lowercase alphanumerics only, so spans match across PDF text extraction."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _title_key(name: str) -> str:
    return unicodedata.normalize("NFC", name).strip().casefold()


def sample_titles(titles, n: int, seed: int) -> list[str]:
    """The n contracts of lowest stable rank, in title order."""
    return stable_sample(titles, n, seed)


def span_pages(page_texts: list[str], spans) -> dict[str, set[int]]:
    """Map each clause category to the pages its annotated spans touch.

    Args:
        page_texts: The normalized text of each page, in order.
        spans: (category, span text) pairs from the annotation.

    Returns:
        Category to zero-based page indices. A span that is not found
        in the contract's text maps to no page.
    """
    offsets = [0]
    for text in page_texts:
        offsets.append(offsets[-1] + len(text))
    full = "".join(page_texts)
    pages: dict[str, set[int]] = {}
    for category, span in spans:
        key = normal_text(span)
        if not key:
            continue
        start = full.find(key)
        if start < 0:
            continue
        end = start + len(key)
        first = bisect_right(offsets, start) - 1
        last = bisect_left(offsets, end) - 1
        pages.setdefault(category, set()).update(
            range(first, min(last, len(page_texts) - 1) + 1))
    return pages


def _category(question_id: str) -> str:
    return question_id.rsplit("__", 1)[-1]


def annotated_spans(entry: dict) -> tuple[list[tuple[str, str]], set[str]]:
    """One contract's (category, span) pairs and its annotated categories."""
    spans = []
    categories = set()
    for question in entry["paragraphs"][0]["qas"]:
        category = _category(question["id"])
        if question.get("is_impossible") or not question["answers"]:
            continue
        categories.add(category)
        for answer in question["answers"]:
            spans.append((category, answer["text"]))
    return spans, categories


def _page_texts(pdf_bytes: bytes) -> list[str]:
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(pdf_bytes)
    try:
        return [
            normal_text(document[index].get_textpage().get_text_bounded())
            for index in range(len(document))
        ]
    finally:
        document.close()


class CuadArchive:
    """The pinned CUAD_v1.zip: its annotation and its contract PDFs."""

    def __init__(self, path: Path):
        self.path = Path(path)
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if digest != CUAD_ARCHIVE_SHA256:
            raise ValueError(
                f"{self.path} has sha256 {digest}, expected {CUAD_ARCHIVE_SHA256}")
        self._zip = zipfile.ZipFile(self.path)
        self._pdf_members = {}
        for name in self._zip.namelist():
            if name.startswith(CUAD_PDF_PREFIX) and name.lower().endswith(".pdf"):
                stem = Path(name).stem
                self._pdf_members[_title_key(stem)] = name

    def entries(self) -> list[dict]:
        """The annotation entries, one per contract, in title order."""
        data = json.loads(self._zip.read(CUAD_JSON_MEMBER))["data"]
        return sorted(data, key=lambda entry: entry["title"])

    def pdf_member(self, title: str) -> str:
        """The archive member holding one annotated contract's PDF.

        A few annotation titles differ from their file name by trailing
        spaces or by a cut-off ending, so a title also matches the one
        file name it is a prefix of.

        Raises:
            KeyError: No PDF, or more than one, matches the title.
        """
        key = _title_key(title)
        if key in self._pdf_members:
            return self._pdf_members[key]
        matches = [name for stem, name in self._pdf_members.items()
                   if stem.startswith(key)]
        if len(matches) != 1:
            raise KeyError(f"CUAD has {len(matches)} PDFs for {title!r}")
        return matches[0]

    def pdf_bytes(self, title: str) -> bytes:
        """The PDF of one annotated contract."""
        return self._zip.read(self.pdf_member(title))


def build_contract_tables(archive: CuadArchive, titles, files_dir: Path
                          ) -> tuple[pa.Table, pa.Table]:
    """Write the sampled contracts' PDFs and return both relations.

    Args:
        archive: The CUAD archive.
        titles: The sampled contract titles, in row order.
        files_dir: Where `<id>.pdf` files are written.

    Returns:
        The `contracts` and `contract_pages` tables.
    """
    files_dir = Path(files_dir)
    files_dir.mkdir(parents=True, exist_ok=True)
    entries = {entry["title"]: entry for entry in archive.entries()}
    contracts = []
    pages = []
    for index, title in enumerate(titles):
        contract_id = f"ct{index}"
        pdf = archive.pdf_bytes(title)
        (files_dir / f"{contract_id}.pdf").write_bytes(pdf)
        spans, categories = annotated_spans(entries[title])
        page_texts = _page_texts(pdf)
        touched = span_pages(page_texts, spans)
        digest = hashlib.sha256(pdf).hexdigest()
        contracts.append({
            "id": contract_id,
            "title": title,
            "page_count": len(page_texts),
            "pdf_sha256": digest,
            "document": f"{FILES_DIR}/{contract_id}.pdf",
            "clauses": sorted(categories),
        })
        for page_index in range(len(page_texts)):
            pages.append({
                "id": f"{contract_id}p{page_index + 1}",
                "contract_id": contract_id,
                "page_number": page_index + 1,
                "pdf_sha256": digest,
                "document": f"{FILES_DIR}/{contract_id}.pdf#page={page_index + 1}",
                "clauses": sorted(
                    category for category, indices in touched.items()
                    if page_index in indices),
            })
    return (pa.Table.from_pylist(contracts, schema=CONTRACT_SCHEMA),
            pa.Table.from_pylist(pages, schema=PAGE_SCHEMA))


def parse_document_reference(reference: str) -> tuple[str, int | None]:
    """Split a document reference into its file and optional page number."""
    path, separator, fragment = reference.partition("#")
    if not separator:
        return path, None
    match = re.fullmatch(r"page=(\d+)", fragment)
    if match is None or int(match.group(1)) < 1:
        raise ValueError(f"unsupported document reference {reference!r}")
    return path, int(match.group(1))


def resolve_documents(table: pa.Table, directory: Path) -> pa.Table:
    """Point a table's document references at files under a directory.

    A published reference is relative, `files/<id>.pdf` with an
    optional `#page=<n>`. The returned table names the absolute file
    so an engine can open it; the page fragment is kept.

    Raises:
        FileNotFoundError: A referenced file is missing.
    """
    if "document" not in table.column_names:
        return table
    directory = Path(directory)
    resolved = []
    for reference in table.column("document").to_pylist():
        path, page = parse_document_reference(reference)
        local = directory / path
        if not local.is_file():
            raise FileNotFoundError(f"document file {local} is missing")
        resolved.append(
            str(local) if page is None else f"{local}#page={page}")
    index = table.schema.get_field_index("document")
    return table.set_column(index, "document", pa.array(resolved, pa.string()))


def verify_document_files(table: pa.Table, directory: Path) -> None:
    """Check every referenced file against the table's recorded hash.

    Raises:
        ValueError: A file's content differs from the published corpus.
    """
    expected = {}
    for reference, digest in zip(table.column("document").to_pylist(),
                                 table.column("pdf_sha256").to_pylist()):
        path, _page = parse_document_reference(reference)
        expected.setdefault(path, digest)
    for path, digest in expected.items():
        actual = hashlib.sha256((Path(directory) / path).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"{path} does not match the published corpus")

