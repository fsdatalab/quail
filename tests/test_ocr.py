"""The OCR operator: PDF rows read as LiteParse text through the text path.

The PDFs are written by hand with real text, so LiteParse reads real
pages; no model runs.
"""

import pyarrow as pa
import pytest
from pdfs import make_pdf, make_text_pdf

import quail
from quail.catalog import OcrProvider, model_only_columns
from quail.execution.types import TokenizedInput
from quail.pdf import (
    OcrOptions,
    PdfReadError,
    read_manifest,
    row_texts,
    sample_page_texts,
)
from quail.pdf.ocr import PAGE_SEPARATOR
from quail.physical import AiJoin, OcrScan, PDFScan
from quail.planner.plan import EngineConfig, Refusal
from quail.specs import DIFFUSION_GEMMA_26B_FP8

pytest.importorskip("liteparse")

# parse in this process rather than a LiteParse worker pool
INLINE = dict(processes=0)


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


def _pages(sources, ocr=True, **kwargs):
    provider = quail.DocumentProvider.from_pdfs(
        sources, id_col="doc_id", path_col="path", **kwargs)
    return provider.ocr(**INLINE) if ocr else provider


def test_row_texts_stream_file_by_file_in_row_order(sources):
    by_page = list(row_texts(_pdf_input(sources, "page"), OcrOptions(**INLINE)))
    # one chunk per file, starting at the file's first row
    assert by_page == [(0, ["alpha one", "", "gamma three words here"]),
                       (3, ["beta one", "beta two"])]
    by_file = list(row_texts(_pdf_input(sources, "pdf"), OcrOptions(**INLINE)))
    assert by_file == [
        (0, [PAGE_SEPARATOR.join(["alpha one", "", "gamma three words here"])]),
        (1, [PAGE_SEPARATOR.join(["beta one", "beta two"])])]


def test_listed_rows_that_revisit_a_file_wait_only_for_it(sources):
    from quail.pdf import PDFInput

    manifest = read_manifest(sources.column("path").to_pylist())
    # rows: a.1, b.2, a.3, b.1; the third row needs only file a
    pdf_input = PDFInput.listed(manifest.sources, manifest.pages,
                                [(0, 1), (1, 2), (0, 3), (1, 1)])
    chunks = list(row_texts(pdf_input, OcrOptions(**INLINE)))
    assert chunks == [(0, ["alpha one"]),
                      (1, ["beta two", "gamma three words here", "beta one"])]


def test_sample_reads_the_first_pages_across_sources(sources):
    pdf_input = _pdf_input(sources, "page")
    options = OcrOptions(**INLINE)
    assert sample_page_texts(pdf_input, options, pages=2) == ["alpha one", ""]
    assert sample_page_texts(pdf_input, options, pages=4) == [
        "alpha one", "", "gamma three words here", "beta one"]
    assert len(sample_page_texts(pdf_input, options, pages=99)) == 5


def test_a_source_changed_after_planning_is_refused(tmp_path):
    path = make_text_pdf(tmp_path / "c.pdf", ["one"])
    pdf_input = _pdf_input(pa.table({"path": [path]}), "page")
    make_text_pdf(tmp_path / "c.pdf", ["one", "two"])
    with pytest.raises(PdfReadError, match="changed after planning"):
        list(row_texts(pdf_input, OcrOptions(**INLINE)))


def test_blank_pages_give_empty_text(tmp_path):
    path = make_pdf(tmp_path / "blank.pdf", [(612, 792)] * 2)
    chunks = list(row_texts(_pdf_input(pa.table({"path": [path]}), "pdf"),
                            OcrOptions(**INLINE)))
    assert chunks == [(0, [PAGE_SEPARATOR])]


def test_the_ocr_operator_is_a_text_table_over_the_pdf_rows(sources):
    pdf = _pages(sources, ocr=False, row_mode="page")
    view = pdf.ocr(**INLINE)
    assert isinstance(view, OcrProvider)
    assert view.columns == pdf.columns
    assert view.schema().field("document").type == pa.string()
    assert model_only_columns(view) == frozenset()
    assert view.statistics().row_count == 5
    assert view.metrics() is None
    # a scan without the document column parses nothing
    ids = view.scan(quail.ScanRequest(columns=("doc_id",))).read_all()
    assert ids.column(0).to_pylist() == ["a", "a", "a", "b", "b"]
    assert view.metrics() is None
    reader = view.scan(quail.ScanRequest(
        columns=("doc_id", "page_number", "document"), batch_rows=2))
    batches = list(reader)
    # streamed file by file, then in batch_rows pieces: 2+1 rows, then 2
    assert [len(batch) for batch in batches] == [2, 1, 2]
    rows = pa.Table.from_batches(batches).to_pydict()
    assert rows == {
        "doc_id": ["a", "a", "a", "b", "b"],
        "page_number": [1, 2, 3, 1, 2],
        "document": ["alpha one", "", "gamma three words here",
                     "beta one", "beta two"],
    }
    metrics = view.metrics()
    assert (metrics["rows"], metrics["pages"], metrics["empty_rows"]) == (5, 5, 1)
    assert metrics["chars"] == sum(len(text) for text in rows["document"])
    # the texts are kept: a later scan serves them, limited, from memory
    again = view.scan(quail.ScanRequest(columns=("document",), limit=2))
    assert again.read_all().column(0).to_pylist() == ["alpha one", ""]
    # the identity names the operator and its options as well as the rows
    assert view.content_identity().startswith(pdf.content_identity())
    assert pdf.ocr(language="deu").content_identity() != view.content_identity()
    # the estimate scales a page sample by each row's page count
    whole = _pages(sources, row_mode="pdf")
    counted = []

    def count(texts):
        counted.append(len(texts))
        return [len(fake_tok(text)) for text in texts]

    # 10 words over 5 pages, so 2 tokens a page: 3 pages and 2 pages
    assert whole.estimate_token_lengths(count) == [6, 4]
    assert counted == [5]


def test_a_limited_first_scan_still_keeps_every_row(sources):
    view = _pages(sources, row_mode="page")
    first = view.scan(quail.ScanRequest(columns=("document",), limit=1))
    assert first.read_all().column(0).to_pylist() == ["alpha one"]
    assert view.metrics()["rows"] == 5
    everything = view.scan(quail.ScanRequest(columns=("document",)))
    assert len(everything.read_all()) == 5


def session(model="qwen3-4b-fp8", tokenizer=fake_tok, **config):
    return quail.Session(EngineConfig(model=model, device="h100-sxm",
                                      **config), tokenizer=tokenizer)


FILTER_SQL = """
    SELECT p.doc_id, p.page_number
    FROM pages p
    WHERE AI_FILTER(PROMPT('{0} Does this page mention beta?', p.document))
"""


def test_a_text_model_plans_the_ocr_operator_as_text(sources):
    with session() as s:
        s.register("pages", _pages(sources, row_mode="page"))
        query = s.sql(FILTER_SQL)
        plan = query.plan()
        assert not isinstance(plan, Refusal), plan
        (scan,) = [n for n in plan.nodes if isinstance(n, OcrScan)]
        assert not any(isinstance(n, PDFScan) for n in plan.nodes)
        assert (scan.n_docs, scan.n_pages, scan.row_mode) == (5, 5, "page")
        assert "OCR text" in query.explain()
        request = query._prepare_physical()
        bound = request.inputs[scan.input_id]
        assert isinstance(bound, TokenizedInput) and len(bound) == 5
        # the tokens are the pages' words; the blank page has none
        store = query.token_inputs()["p"]
        assert list(store.lengths) == [2, 0, 4, 2, 2]
        assert store.column("page_number").to_pylist() == [1, 2, 3, 1, 2]
        assert request.plan["graph"]["nodes"][0]["type"] == "quail.ocr_scan"
        # the operator's counters are reported per alias
        metrics = query.ocr_metrics()
        assert set(metrics) == {"p"}
        assert (metrics["p"]["pages"], metrics["p"]["empty_rows"]) == (5, 1)
        assert metrics["p"]["extract_s"] >= 0


def test_a_text_model_refuses_pdf_pages_and_points_at_the_operator(sources):
    with session() as s:
        s.register("pages", _pages(sources, ocr=False, row_mode="page"))
        plan = s.sql(FILTER_SQL).plan()
        assert isinstance(plan, Refusal)
        assert plan.constraint == "model_takes_text_only"
        assert ".ocr()" in plan.reasons[0]


def test_the_operator_on_an_image_model_and_the_round_trip(sources):
    from quail.builtins import built_in_registry
    from quail.physical import decode_graph

    with session(DIFFUSION_GEMMA_26B_FP8.name) as s:
        s.register("pages", _pages(sources, row_mode="pdf"))
        plan = s.sql("""
            SELECT p.doc_id FROM pages p
            WHERE AI_FILTER(PROMPT('{0} Is this an invoice?', p.document))
        """).plan()
        (scan,) = [n for n in plan.nodes if isinstance(n, OcrScan)]
        assert (scan.n_docs, scan.n_pages, scan.pages_per_row_max) == (2, 5, 3)
        envelope = plan.to_envelope(s.registry.codecs)
        graph = decode_graph(envelope["graph"], built_in_registry().codecs)
        (decoded,) = [n for n in graph.nodes if isinstance(n, OcrScan)]
        assert decoded == scan


def test_the_operator_text_is_an_ordinary_column(sources):
    from quail.logical import CompileError

    selected = FILTER_SQL.replace("p.doc_id, p.page_number",
                                  "p.doc_id, p.document")
    with session() as s:
        s.register("pages", _pages(sources, row_mode="page"))
        s.register("images", _pages(sources, ocr=False, row_mode="page"))
        query = s.sql(selected)
        assert "p.document" in query.explain()
        with pytest.raises(CompileError, match="holds the PDF pages"):
            s.sql(selected.replace("FROM pages p", "FROM images p"))


JOIN_SQL = """
    SELECT q.id, p.doc_id FROM questions q JOIN pages p
    ON q.doc_id = p.doc_id
    AND AI.IF(PROMPT('Question: {0}\\nDoes this page answer it?\\n{1}',
                     q.question, p.document))
"""


def test_a_join_over_the_operator_may_anchor_on_either_side(sources):
    with session() as s:
        s.register("pages", _pages(sources, row_mode="page"))
        s.register("questions", quail.DocumentProvider.from_table(
            pa.table({"id": [1, 2, 3],
                      "doc_id": ["a", "a", "b"],
                      "question": ["alpha?", "gamma?", "beta?"]}),
            id_col="id"))
        plan = s.sql(JOIN_SQL, dialect="bq").plan()
        assert not isinstance(plan, Refusal), plan
        assert any(isinstance(n, OcrScan) for n in plan.nodes)
        # the pages are text now, so an explicit question anchor holds
        forced = s.sql(JOIN_SQL.replace(
            "p.document))", "p.document), {'anchor': 'q'})"),
            dialect="bq").plan()
        assert not isinstance(forced, Refusal), forced
        (join,) = [n for n in forced.nodes if isinstance(n, AiJoin)]
        assert join.anchor == "q"


def test_request_backends_take_the_operator_but_not_pages(sources):
    def byte_tok(text):
        return list(text.encode())

    with session(backend="stock_vllm", tokenizer=byte_tok) as s:
        s.register("pages", _pages(sources, row_mode="page"))
        plan = s.sql(FILTER_SQL).plan()
        assert not isinstance(plan, Refusal), plan
        assert any(isinstance(n, OcrScan) for n in plan.nodes)
        assert plan.backend == "stock_vllm"
    with session(DIFFUSION_GEMMA_26B_FP8.name, backend="stock_vllm",
                 tokenizer=byte_tok) as s:
        s.register("pages", _pages(sources, ocr=False, row_mode="page"))
        plan = s.sql(FILTER_SQL).plan()
        assert isinstance(plan, Refusal)
        assert plan.constraint == "pdf_input_unsupported"
        assert ".ocr()" in plan.reasons[0]
