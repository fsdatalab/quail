"""PDF page inputs: provider, planner, compiler rules, and request binding.

No page is rendered here. The tests build small blank PDFs with
PDFium so the page manifest and the token arithmetic are real.
"""

import pyarrow as pa
import pytest

import quail
from quail.catalog import model_only_columns
from quail.execution.execute import check_scan_input
from quail.execution.types import PhysicalRequest, document_input
from quail.logical import CompileError
from quail.pdf import PDFInput, PdfPageRef, PdfRowRef, PdfSource, rows_for_mode
from quail.physical import PDFScan, TextScan
from quail.planner.plan import EngineConfig, Refusal
from quail.specs import DIFFUSION_GEMMA_26B_FP8, QWEN3_4B_FP8
from quail.specs.vision import render_size, resolve_image_tokens, soft_tokens

pypdfium2 = pytest.importorskip("pypdfium2")

LETTER = (612, 792)
GEMMA = DIFFUSION_GEMMA_26B_FP8


def fake_tok(text):
    return text.split()


def make_pdf(path, sizes):
    """Write a PDF with one blank page per (width, height) in points."""
    document = pypdfium2.PdfDocument.new()
    for width, height in sizes:
        document.new_page(width, height)
    document.save(str(path))
    document.close()
    return str(path)


@pytest.fixture()
def sources(tmp_path):
    """Two PDFs: three letter pages, then one letter and one landscape page."""
    return pa.table({
        "doc_id": ["a", "b"],
        "path": [make_pdf(tmp_path / "a.pdf", [LETTER] * 3),
                 make_pdf(tmp_path / "b.pdf", [LETTER, (792, 612)])],
        "title": ["first", "second"],
    })


def letter_soft(budget):
    return soft_tokens(GEMMA, *LETTER, budget)


def test_soft_tokens_match_the_processor_at_every_budget():
    # the Phase 0 probe ran the checkpoint's own processor on a letter
    # page at each budget (/results/ablations/pdf_probe_vision.json)
    assert [letter_soft(b) for b in GEMMA.image_token_budgets] == [
        63, 130, 266, 520, 1102]
    assert render_size(GEMMA, *LETTER, 280) == (912, 672)
    # a landscape page has the same token count as its portrait twin
    assert soft_tokens(GEMMA, 792, 612, 280) == 266
    with pytest.raises(ValueError, match="no image geometry"):
        soft_tokens(QWEN3_4B_FP8, *LETTER, 280)


def test_resolve_image_tokens():
    assert resolve_image_tokens(GEMMA, None) == 280
    assert resolve_image_tokens(GEMMA, 560) == 560
    assert resolve_image_tokens(QWEN3_4B_FP8, None) == 0
    with pytest.raises(ValueError, match="accepts image_tokens in"):
        resolve_image_tokens(GEMMA, 300)
    with pytest.raises(ValueError, match="takes text only"):
        resolve_image_tokens(QWEN3_4B_FP8, 280)


def test_pdf_input_validation():
    src = (PdfSource("/a.pdf", 1, 1), PdfSource("/b.pdf", 1, 1))
    pages = (PdfPageRef(0, 0, 612, 792), PdfPageRef(0, 1, 612, 792),
             PdfPageRef(1, 0, 612, 792))
    assert rows_for_mode(pages, 2, "page") == (
        PdfRowRef((0,)), PdfRowRef((1,)), PdfRowRef((2,)))
    assert rows_for_mode(pages, 2, "pdf") == (PdfRowRef((0, 1)), PdfRowRef((2,)))
    with pytest.raises(ValueError, match="row_mode must be one of"):
        rows_for_mode(pages, 2, "chapter")
    with pytest.raises(ValueError, match="source 2 has no pages"):
        rows_for_mode(pages, 3, "pdf")
    good = PDFInput(src, pages, rows_for_mode(pages, 2, "pdf"), "pdf", 280)
    assert (len(good), good.page_count, good.pages_per_row_max) == (2, 3, 2)
    assert good.row_pages(0) == pages[:2]
    with pytest.raises(ValueError, match="mixes pages of sources"):
        PDFInput(src, pages, (PdfRowRef((1, 2)),), "pdf", 280)
    with pytest.raises(ValueError, match="refers to page 7"):
        PDFInput(src, pages, (PdfRowRef((7,)),), "page", 280)
    with pytest.raises(ValueError, match="at least one page"):
        PdfRowRef(())


def test_page_mode_provider_rows_and_schema(sources):
    provider = quail.DocumentProvider.from_pdfs(
        sources, id_col="doc_id", path_col="path", row_mode="page")
    assert provider.columns == (
        "doc_id", "path", "title", "page_number", "page_count", "document")
    assert model_only_columns(provider) == {"document"}
    assert provider.statistics().row_count == 5
    assert provider.content_identity().startswith("pdf:")
    assert provider.content_identity() == provider.content_identity()
    reader = provider.scan(quail.ScanRequest(
        columns=("doc_id", "page_number", "page_count", "title")))
    rows = reader.read_all().to_pydict()
    assert rows == {
        "doc_id": ["a", "a", "a", "b", "b"],
        "page_number": [1, 2, 3, 1, 2],
        "page_count": [3, 3, 3, 2, 2],
        "title": ["first", "first", "first", "second", "second"],
    }
    pdf_input = provider.pdf_input(280)
    assert pdf_input.row_mode == "page"
    assert [p.page_index for p in pdf_input.pages] == [0, 1, 2, 0, 1]
    assert pdf_input.pages[4].width_points == 792.0


def test_pdf_mode_provider_rows(sources):
    provider = quail.DocumentProvider.from_pdfs(
        sources, id_col="doc_id", path_col="path", row_mode="pdf")
    assert provider.columns == ("doc_id", "path", "title", "page_count", "document")
    rows = provider.scan(quail.ScanRequest(
        columns=("doc_id", "page_count"))).read_all().to_pydict()
    assert rows == {"doc_id": ["a", "b"], "page_count": [3, 2]}
    pdf_input = provider.pdf_input(140)
    assert [r.page_ids for r in pdf_input.rows] == [(0, 1, 2), (3, 4)]
    with pytest.raises(CompileError, match="row_mode"):
        quail.DocumentProvider.from_pdfs(
            sources, id_col="doc_id", path_col="path", row_mode="chapter")
    with pytest.raises(CompileError, match="clash with the columns"):
        quail.DocumentProvider.from_pdfs(
            sources.append_column("page_count", pa.array([1, 2])),
            id_col="doc_id", path_col="path", row_mode="pdf")


def test_missing_pdf_is_reported(tmp_path):
    provider = quail.DocumentProvider.from_pdfs(
        pa.table({"id": ["x"], "path": [str(tmp_path / "missing.pdf")]}),
        id_col="id", path_col="path", row_mode="page")
    with pytest.raises(quail.PdfReadError, match="cannot read PDF"):
        provider.statistics()


def gemma_session(image_tokens=None):
    return quail.Session(EngineConfig(
        model=GEMMA.name, device="h100-sxm", image_tokens=image_tokens),
        tokenizer=fake_tok)


FILTER_SQL = """
    SELECT p.doc_id, p.page_number
    FROM pages p
    WHERE AI_FILTER(PROMPT('{0} Does this page show a table?', p.document))
"""


def test_plan_binds_pdf_scan_with_planned_lengths(sources):
    with gemma_session() as session:
        session.register("pages", quail.DocumentProvider.from_pdfs(
            sources, id_col="doc_id", path_col="path", row_mode="page"))
        query = session.sql(FILTER_SQL)
        plan = query.plan()
        (scan,) = [n for n in plan.nodes if isinstance(n, PDFScan)]
        per_page = letter_soft(280) + GEMMA.image_frame_tokens
        assert (scan.n_docs, scan.n_pages, scan.row_mode) == (5, 5, "page")
        assert (scan.visual_tokens, scan.pages_per_row_max) == (280, 1)
        assert scan.total_tokens == 5 * per_page
        assert scan.shards == (range(0, 5),)
        assert "row_mode=page" in query.explain()
        request = query._prepare_physical()
        assert isinstance(request, PhysicalRequest)
        bound = request.inputs[scan.input_id]
        assert isinstance(bound, PDFInput) and len(bound) == 5
        check_scan_input(scan, bound)
        # the value table for page_number came from the provider scan
        assert query.token_inputs()["p"].column("page_number").to_pylist() == [
            1, 2, 3, 1, 2]


def test_pdf_mode_plans_one_row_per_pdf(sources):
    with gemma_session(image_tokens=140) as session:
        session.register("pages", quail.DocumentProvider.from_pdfs(
            sources, id_col="doc_id", path_col="path", row_mode="pdf"))
        plan = session.sql("""
            SELECT p.doc_id FROM pages p
            WHERE AI_FILTER(PROMPT('{0} Is this an invoice?', p.document))
        """).plan()
        (scan,) = [n for n in plan.nodes if isinstance(n, PDFScan)]
        per_page = letter_soft(140) + GEMMA.image_frame_tokens
        assert (scan.n_docs, scan.n_pages, scan.pages_per_row_max) == (2, 5, 3)
        assert scan.total_tokens == 5 * per_page
        assert scan.visual_tokens == 140


def test_pdf_scan_round_trips_through_the_envelope(sources):
    from quail.builtins import built_in_registry
    from quail.physical import decode_graph

    with gemma_session() as session:
        session.register("pages", quail.DocumentProvider.from_pdfs(
            sources, id_col="doc_id", path_col="path", row_mode="page"))
        plan = session.sql(FILTER_SQL).plan()
        envelope = plan.to_envelope(session.registry.codecs)
        graph = decode_graph(envelope["graph"], built_in_registry().codecs)
        scans = [n for n in graph.nodes if isinstance(n, PDFScan)]
        assert scans and scans[0].attributes()["row_mode"] == "page"


def test_text_model_refuses_pdf_rows(sources):
    with quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
                       tokenizer=fake_tok) as session:
        session.register("pages", quail.DocumentProvider.from_pdfs(
            sources, id_col="doc_id", path_col="path", row_mode="page"))
        plan = session.sql(FILTER_SQL).plan()
        assert isinstance(plan, Refusal)
        assert plan.constraint == "model_takes_text_only"


def test_image_tokens_are_checked_at_session_start():
    with pytest.raises(quail.RefusalError, match="accepts image_tokens"):
        gemma_session(image_tokens=300)
    with pytest.raises(quail.RefusalError, match="takes text only"):
        quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm",
                                   image_tokens=280), tokenizer=fake_tok)


def test_compiler_keeps_the_document_column_out_of_values(sources, tmp_path):
    import pyarrow.parquet as pq

    with gemma_session() as session:
        session.register("pages", quail.DocumentProvider.from_pdfs(
            sources, id_col="doc_id", path_col="path", row_mode="page"))
        pq.write_table(pa.table({"doc_id": ["a"], "body": ["text"]}),
                       str(tmp_path / "t.parquet"))
        session.register("texts", quail.DocumentProvider.from_parquet(
            str(tmp_path / "t.parquet"), id_col="doc_id"))
        with pytest.raises(CompileError, match="cannot appear in SELECT"):
            session.sql("""
                SELECT p.document FROM pages p
                WHERE AI_FILTER(PROMPT('{0} table?', p.document))
            """)
        star = session.sql("""
            SELECT * FROM pages p
            WHERE AI_FILTER(PROMPT('{0} table?', p.document))
        """)
        assert "document" not in {
            ref.column for ref in star.logical.output_schema()}
        with pytest.raises(CompileError, match="AI.JOIN over PDF rows"):
            session.sql("""
                SELECT p.doc_id FROM pages p JOIN texts t
                ON AI_FILTER(PROMPT('{0} matches {1}?', p.document, t.body))
            """)
        with pytest.raises(CompileError, match="a join condition"):
            session.sql("""
                SELECT p.doc_id FROM pages p JOIN texts t
                ON p.document = t.doc_id
                WHERE AI_FILTER(PROMPT('{0} table?', p.document))
            """)


def test_builder_applies_the_same_column_rules(sources):
    with gemma_session() as session:
        session.register("pages", quail.DocumentProvider.from_pdfs(
            sources, id_col="doc_id", path_col="path", row_mode="page"))
        plan = (session.docs("pages").alias("p")
                .ai_filter(quail.prompt("{0} table?", quail.col("document")))
                .select("*").plan())
        assert any(isinstance(n, PDFScan) for n in plan.nodes)
        with pytest.raises(CompileError, match="cannot appear in select"):
            (session.docs("pages").alias("p")
             .ai_filter(quail.prompt("{0} table?", quail.col("document")))
             .select("document"))


def test_check_scan_input_rejects_mismatched_bindings():
    text = TextScan(node_id="scan:t", alias="t", input_id="t", n_docs=1)
    pdf = PDFScan(node_id="scan:p", alias="p", input_id="p", n_docs=1,
                  row_mode="page", n_pages=1, visual_tokens=280,
                  pages_per_row_max=1)
    src = (PdfSource("/a.pdf", 1, 1),)
    pages = (PdfPageRef(0, 0, 612, 792),)
    pdf_input = PDFInput(src, pages, (PdfRowRef((0,)),), "page", 280)
    check_scan_input(pdf, pdf_input)
    check_scan_input(text, document_input([[1, 2]]))
    with pytest.raises(TypeError, match="needs PDFInput"):
        check_scan_input(pdf, document_input([[1, 2]]))
    with pytest.raises(TypeError, match="needs TokenizedInput"):
        check_scan_input(text, pdf_input)
    with pytest.raises(ValueError, match="plans 1 documents but its input holds 2"):
        check_scan_input(text, document_input([[1], [2]]))
    with pytest.raises(ValueError, match="visual tokens"):
        check_scan_input(pdf, PDFInput(src, pages, (PdfRowRef((0,)),),
                                       "page", 560))
    with pytest.raises(ValueError, match="one page per row"):
        PDFScan(node_id="scan:p", alias="p", input_id="p", n_docs=1,
                row_mode="page", n_pages=2, visual_tokens=280,
                pages_per_row_max=2)
