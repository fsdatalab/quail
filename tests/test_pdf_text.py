"""The text reading of PDF rows: LiteParse extraction through the text path.

The PDFs are written by hand with real text, so LiteParse reads real
pages; no model runs.
"""

from dataclasses import replace

import pyarrow as pa
import pytest
from pdfs import make_pdf, make_text_pdf

import quail
from quail.catalog import PdfTextProvider, model_only_columns
from quail.execution.types import TokenizedInput
from quail.pdf import (
    PdfReadError,
    PdfTextOptions,
    read_manifest,
    row_texts,
    sample_page_texts,
)
from quail.pdf.text import PAGE_SEPARATOR
from quail.physical import AiJoin, PDFScan, PdfTextScan
from quail.planner.plan import EngineConfig, Refusal
from quail.specs import DIFFUSION_GEMMA_26B_FP8

pytest.importorskip("liteparse")

INLINE = PdfTextOptions(processes=0)


def fake_tok(text):
    return text.split()


@pytest.fixture()
def sources(tmp_path):
    """Two PDFs with text: three pages (the middle one blank), then two."""
    return pa.table({
        "doc_id": ["a", "b"],
        "path": [make_text_pdf(tmp_path / "a.pdf",
                               ["alpha one", "", "gamma three words here"]),
                 make_text_pdf(tmp_path / "b.pdf", ["beta one", "beta two"])],
        "title": ["first", "second"],
    })


def _pdf_input(sources, row_mode):
    from quail.pdf import PDFInput

    manifest = read_manifest(sources.column("path").to_pylist())
    return PDFInput.formed(manifest.sources, manifest.pages, row_mode)


def test_row_texts_follow_the_rows(sources):
    by_page = row_texts(_pdf_input(sources, "page"), INLINE)
    assert by_page.rows == ("alpha one", "", "gamma three words here",
                            "beta one", "beta two")
    assert by_page.metrics["pages"] == 5 and by_page.metrics["empty_pages"] == 1
    assert by_page.metrics["ocr"] is False
    by_file = row_texts(_pdf_input(sources, "pdf"), INLINE)
    assert by_file.rows == (
        PAGE_SEPARATOR.join(["alpha one", "", "gamma three words here"]),
        PAGE_SEPARATOR.join(["beta one", "beta two"]))
    assert by_file.metrics["chars"] == sum(len(row) for row in by_file.rows)


def test_sample_reads_the_first_pages_across_sources(sources):
    pdf_input = _pdf_input(sources, "page")
    assert sample_page_texts(pdf_input, INLINE, pages=2) == ["alpha one", ""]
    assert sample_page_texts(pdf_input, INLINE, pages=4) == [
        "alpha one", "", "gamma three words here", "beta one"]
    assert len(sample_page_texts(pdf_input, INLINE, pages=99)) == 5


def test_a_source_changed_after_planning_is_refused(tmp_path):
    path = make_text_pdf(tmp_path / "c.pdf", ["one"])
    pdf_input = _pdf_input(pa.table({"path": [path]}), "page")
    make_text_pdf(tmp_path / "c.pdf", ["one", "two"])
    with pytest.raises(PdfReadError, match="changed after planning"):
        row_texts(pdf_input, INLINE)


def test_blank_pages_give_empty_text(tmp_path):
    path = make_pdf(tmp_path / "blank.pdf", [(612, 792)] * 2)
    texts = row_texts(_pdf_input(pa.table({"path": [path]}), "pdf"), INLINE)
    assert texts.rows == (PAGE_SEPARATOR,)
    assert texts.metrics["empty_pages"] == 2


def test_text_provider_is_a_text_table_over_the_pdf_rows(sources):
    pdf = quail.DocumentProvider.from_pdfs(
        sources, id_col="doc_id", path_col="path", row_mode="page")
    view = PdfTextProvider(pdf, INLINE)
    assert view.columns == pdf.columns
    assert view.schema().field("document").type == pa.string()
    assert model_only_columns(view) == frozenset()
    assert view.statistics().row_count == 5
    assert view.metrics() is None
    rows = view.scan(quail.ScanRequest(
        columns=("doc_id", "page_number", "document"))).read_all().to_pydict()
    assert rows == {
        "doc_id": ["a", "a", "a", "b", "b"],
        "page_number": [1, 2, 3, 1, 2],
        "document": ["alpha one", "", "gamma three words here",
                     "beta one", "beta two"],
    }
    assert view.metrics()["pages"] == 5
    # the identity names the reading as well as the rows
    assert view.content_identity().startswith(pdf.content_identity())
    other = PdfTextProvider(pdf, PdfTextOptions(ocr=True, processes=0))
    assert other.content_identity() != view.content_identity()
    # the estimate scales a page sample by each row's page count
    whole = PdfTextProvider(quail.DocumentProvider.from_pdfs(
        sources, id_col="doc_id", path_col="path", row_mode="pdf"), INLINE)
    counted = []

    def count(texts):
        counted.append(len(texts))
        return [len(fake_tok(text)) for text in texts]

    # 10 words over 5 pages, so 2 tokens a page: 3 pages and 2 pages
    assert whole.estimate_token_lengths(count) == [6, 4]
    assert counted == [5]


def text_session(model="qwen3-4b-fp8", tokenizer=fake_tok, **config):
    session = quail.Session(EngineConfig(model=model, device="h100-sxm",
                                         **config), tokenizer=tokenizer)
    # parse in this process rather than a LiteParse worker pool
    session.pdf_text_options = replace(session.pdf_text_options, processes=0)
    return session


FILTER_SQL = """
    SELECT p.doc_id, p.page_number
    FROM pages p
    WHERE AI_FILTER(PROMPT('{0} Does this page mention beta?', p.document))
"""


def test_text_model_plans_pdf_rows_as_extracted_text(sources):
    with text_session() as session:
        session.register("pages", quail.DocumentProvider.from_pdfs(
            sources, id_col="doc_id", path_col="path", row_mode="page"))
        query = session.sql(FILTER_SQL)
        plan = query.plan()
        assert not isinstance(plan, Refusal), plan
        (scan,) = [n for n in plan.nodes if isinstance(n, PdfTextScan)]
        assert not any(isinstance(n, PDFScan) for n in plan.nodes)
        assert (scan.n_docs, scan.n_pages, scan.row_mode, scan.ocr) == (
            5, 5, "page", False)
        assert "read as text, ocr=off" in query.explain()
        request = query._prepare_physical()
        bound = request.inputs[scan.input_id]
        assert isinstance(bound, TokenizedInput) and len(bound) == 5
        # the tokens are the pages' words; the blank page has none
        store = query.token_inputs()["p"]
        assert list(store.lengths) == [2, 0, 4, 2, 2]
        assert store.column("page_number").to_pylist() == [1, 2, 3, 1, 2]
        assert request.plan["graph"]["nodes"][0]["type"] == "quail.pdf_text_scan"
        # the extraction's counters are reported per alias
        metrics = query.pdf_text_metrics()
        assert set(metrics) == {"p"}
        assert (metrics["p"]["pages"], metrics["p"]["empty_pages"],
                metrics["p"]["ocr"]) == (5, 1, False)
        assert metrics["p"]["extract_s"] >= 0


def test_pdf_read_text_on_an_image_model_and_the_round_trip(sources):
    from quail.builtins import built_in_registry
    from quail.physical import decode_graph

    with text_session(DIFFUSION_GEMMA_26B_FP8.name, pdf_read="text",
                      pdf_ocr=True) as session:
        assert session.pdf_reading == "text"
        session.register("pages", quail.DocumentProvider.from_pdfs(
            sources, id_col="doc_id", path_col="path", row_mode="pdf"))
        plan = session.sql("""
            SELECT p.doc_id FROM pages p
            WHERE AI_FILTER(PROMPT('{0} Is this an invoice?', p.document))
        """).plan()
        (scan,) = [n for n in plan.nodes if isinstance(n, PdfTextScan)]
        assert (scan.n_docs, scan.n_pages, scan.pages_per_row_max) == (2, 5, 3)
        envelope = plan.to_envelope(session.registry.codecs)
        graph = decode_graph(envelope["graph"], built_in_registry().codecs)
        (decoded,) = [n for n in graph.nodes if isinstance(n, PdfTextScan)]
        assert decoded == scan and decoded.ocr is True


JOIN_SQL = """
    SELECT q.id, p.doc_id FROM questions q JOIN pages p
    ON q.doc_id = p.doc_id
    AND AI.IF(PROMPT('Question: {0}\\nDoes this page answer it?\\n{1}',
                     q.question, p.document))
"""


def test_a_join_over_pages_read_as_text_may_anchor_on_either_side(sources):
    with text_session() as session:
        session.register("pages", quail.DocumentProvider.from_pdfs(
            sources, id_col="doc_id", path_col="path", row_mode="page"))
        session.register("questions", quail.DocumentProvider.from_table(
            pa.table({"id": [1, 2, 3],
                      "doc_id": ["a", "a", "b"],
                      "question": ["alpha?", "gamma?", "beta?"]}),
            id_col="id"))
        plan = session.sql(JOIN_SQL, dialect="bq").plan()
        assert not isinstance(plan, Refusal), plan
        assert any(isinstance(n, PdfTextScan) for n in plan.nodes)
        # the pages are text now, so an explicit question anchor holds
        forced = session.sql(JOIN_SQL.replace(
            "p.document))", "p.document), {'anchor': 'q'})"),
            dialect="bq").plan()
        assert not isinstance(forced, Refusal), forced
        (join,) = [n for n in forced.nodes if isinstance(n, AiJoin)]
        assert join.anchor == "q"


def test_request_backends_read_pdf_rows_as_text_only(sources):
    def byte_tok(text):
        return list(text.encode())

    with text_session(backend="stock_vllm", tokenizer=byte_tok) as session:
        session.register("pages", quail.DocumentProvider.from_pdfs(
            sources, id_col="doc_id", path_col="path", row_mode="page"))
        plan = session.sql(FILTER_SQL).plan()
        assert not isinstance(plan, Refusal), plan
        assert any(isinstance(n, PdfTextScan) for n in plan.nodes)
        assert plan.backend == "stock_vllm"
    with text_session(DIFFUSION_GEMMA_26B_FP8.name, backend="stock_vllm",
                      pdf_read="image", tokenizer=byte_tok) as session:
        session.register("pages", quail.DocumentProvider.from_pdfs(
            sources, id_col="doc_id", path_col="path", row_mode="page"))
        plan = session.sql(FILTER_SQL).plan()
        assert isinstance(plan, Refusal)
        assert plan.constraint == "pdf_input_unsupported"
