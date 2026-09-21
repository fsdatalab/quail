"""PDF page inputs: provider, planner, compiler rules, and request binding.

No page is rendered here. The tests build small blank PDFs with
PDFium so the page manifest and the token arithmetic are real. The
the OCR operator is covered in test_ocr.py.
"""

import pyarrow as pa
import pytest
from pdfs import LETTER, make_pdf

import quail
from quail.catalog import model_only_columns
from quail.execution.execute import check_scan_input
from quail.execution.types import PhysicalRequest, document_input
from quail.logical import CompileError
from quail.pdf import PDFInput, PdfPageRef, PdfRowRef, PdfSource
from quail.physical import PDFScan, TextScan
from quail.planner.plan import EngineConfig, PdfDocuments, Refusal
from quail.specs import DIFFUSION_GEMMA_26B_FP8, QWEN3_4B_FP8
from quail.specs.vision import render_size, resolve_image_tokens, soft_tokens

pypdfium2 = pytest.importorskip("pypdfium2")

GEMMA = DIFFUSION_GEMMA_26B_FP8


def fake_tok(text):
    return text.split()


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
    assert PDFInput.formed(src, pages, "page").rows == (
        PdfRowRef((0,)), PdfRowRef((1,)), PdfRowRef((2,)))
    good = PDFInput.formed(src, pages, "pdf")
    assert good.rows == (PdfRowRef((0, 1)), PdfRowRef((2,)))
    with pytest.raises(ValueError, match="row_mode must be one of"):
        PDFInput.formed(src, pages, "chapter")
    with pytest.raises(ValueError, match="source 2 has no pages"):
        PDFInput.formed(src + (PdfSource("/c.pdf", 1, 1),), pages, "pdf")
    assert (len(good), good.page_count, good.pages_per_row_max) == (2, 3, 2)
    assert good.row_pages(0) == pages[:2]
    assert good.page_counts == (2, 1) and good.source_rows == (0, 1)
    listed = PDFInput.listed(src, pages, [(1, 1), (0, 2)])
    assert listed.rows == (PdfRowRef((2,)), PdfRowRef((1,)))
    assert listed.row_mode == "page" and listed.source_rows == (1, 0)
    with pytest.raises(ValueError, match="page 3 of '/a.pdf' does not exist"):
        PDFInput.listed(src, pages, [(0, 3)])
    with pytest.raises(ValueError, match="mixes pages of sources"):
        PDFInput(src, pages, (PdfRowRef((1, 2)),), "pdf")
    with pytest.raises(ValueError, match="refers to page 7"):
        PDFInput(src, pages, (PdfRowRef((7,)),), "page")
    with pytest.raises(ValueError, match="page mode row shows one"):
        PDFInput(src, pages, (PdfRowRef((0, 1)),), "page")
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
    pdf_input = provider.pdf_input()
    assert pdf_input.row_mode == "page"
    assert [p.page_index for p in pdf_input.pages] == [0, 1, 2, 0, 1]
    assert pdf_input.pages[4].width_points == 792.0
    # the identity covers the table's values, not only its schema
    renamed = quail.DocumentProvider.from_pdfs(
        sources.set_column(2, "title", pa.array(["x", "y"])),
        id_col="doc_id", path_col="path", row_mode="page")
    assert renamed.content_identity() != provider.content_identity()


def test_pdf_mode_provider_rows(sources):
    provider = quail.DocumentProvider.from_pdfs(
        sources, id_col="doc_id", path_col="path", row_mode="pdf")
    assert provider.columns == ("doc_id", "path", "title", "page_count", "document")
    rows = provider.scan(quail.ScanRequest(
        columns=("doc_id", "page_count"))).read_all().to_pydict()
    assert rows == {"doc_id": ["a", "b"], "page_count": [3, 2]}
    pdf_input = provider.pdf_input()
    assert [r.page_ids for r in pdf_input.rows] == [(0, 1, 2), (3, 4)]
    with pytest.raises(CompileError, match="row_mode"):
        quail.DocumentProvider.from_pdfs(
            sources, id_col="doc_id", path_col="path", row_mode="chapter")
    with pytest.raises(CompileError, match="not both"):
        quail.catalog.PDFProvider(sources, "doc_id", "path")
    with pytest.raises(CompileError, match="clash with the columns"):
        quail.DocumentProvider.from_pdfs(
            sources.append_column("page_count", pa.array([1, 2])),
            id_col="doc_id", path_col="path", row_mode="pdf")


def test_listed_pages_provider_keeps_the_rows_and_ids_it_is_given(sources):
    a_path, b_path = sources.column("path").to_pylist()
    listed = pa.table({
        "page_id": ["b2", "a3", "a1"],
        "path": [b_path, a_path, a_path],
        "page_number": pa.array([2, 3, 1], pa.int32()),
        "label": ["x", "y", "z"],
    })
    provider = quail.DocumentProvider.from_pdf_pages(
        listed, id_col="page_id", path_col="path", page_col="page_number")
    assert provider.columns == (
        "page_id", "path", "page_number", "label", "page_count", "document")
    assert model_only_columns(provider) == {"document"}
    assert provider.statistics().row_count == 3
    rows = provider.scan(quail.ScanRequest(
        columns=("page_id", "page_number", "page_count", "label"))).read_all()
    assert rows.to_pydict() == {
        "page_id": ["b2", "a3", "a1"], "page_number": [2, 3, 1],
        "page_count": [2, 3, 3], "label": ["x", "y", "z"]}
    pdf_input = provider.pdf_input()
    assert pdf_input.row_mode == "page"
    assert [(p.source_index, p.page_number) for p in pdf_input.row_pages(0)] == [
        (0, 2)]
    assert [(p.source_index, p.page_number) for p in pdf_input.row_pages(1)] == [
        (1, 3)]
    # the identity tells listed pages apart from the same files' full rows
    other = quail.DocumentProvider.from_pdf_pages(
        listed.slice(0, 2), id_col="page_id", path_col="path",
        page_col="page_number")
    assert provider.content_identity() != other.content_identity()

    with pytest.raises(CompileError, match="page column 'page'"):
        quail.DocumentProvider.from_pdf_pages(
            listed, id_col="page_id", path_col="path", page_col="page")
    with pytest.raises(CompileError, match="integer page numbers"):
        quail.DocumentProvider.from_pdf_pages(
            listed, id_col="page_id", path_col="path", page_col="label")
    beyond = listed.set_column(2, "page_number", pa.array([2, 4, 1], pa.int32()))
    with pytest.raises(CompileError, match="page 4 of .* has 3 pages"):
        quail.DocumentProvider.from_pdf_pages(
            beyond, id_col="page_id", path_col="path",
            page_col="page_number").statistics()


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


def test_the_estimate_prices_every_page_image_in_full(sources):
    """Pages share placeholder ids, not KV, so no page gets a prefix credit."""
    from quail.planner.prefixes import prefix_credits

    with gemma_session() as session:
        session.register("pages", quail.DocumentProvider.from_pdfs(
            sources, id_col="doc_id", path_col="path", row_mode="page"))
        query = session.sql(FILTER_SQL)
        assert prefix_credits(query.token_inputs()["p"]) == [0] * 5
        estimate = quail.speed_of_light_estimate(
            query, lambda prompt, assignment: True)
        per_page = letter_soft(280) + GEMMA.image_frame_tokens
        question = query.logical.operators().filters["p"][0].prompt.tail_tokens
        preamble = query.logical.operators().filters["p"][0].prompt.preamble_tokens
        assert estimate.fresh_tokens == 5 * (
            preamble + per_page + question + GEMMA.canvas_tokens)
        assert estimate.documents_by_alias == {"p": 5}
        # the landscape page is letter rotated, so every page has the
        # same soft token count; attention stays inside each page
        patches = letter_soft(280) * GEMMA.image_pool_kernel ** 2
        assert estimate.work.image_patches == 5 * patches
        assert estimate.work.image_pairs == 5 * patches * patches
        assert estimate.work.image_soft_tokens == 5 * letter_soft(280)
        tower = GEMMA.vision_tower
        attention = estimate.latency.component("vision_attention")
        assert attention.precision == "bf16"
        assert attention.flops == (
            4 * tower.heads * tower.head_dim * tower.layers
            * estimate.work.image_pairs)
        assert [c.name for c in estimate.latency.components][:5] == [
            "vision_embed", "vision_attn_proj", "vision_mlp",
            "vision_attention", "vision_project"]


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


def test_text_model_refuses_pdf_pages(sources):
    with quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
                       tokenizer=fake_tok) as session:
        session.register("pages", quail.DocumentProvider.from_pdfs(
            sources, id_col="doc_id", path_col="path", row_mode="page"))
        plan = session.sql(FILTER_SQL).plan()
        assert isinstance(plan, Refusal)
        assert plan.constraint == "model_takes_text_only"


def test_pdf_rows_need_one_gpu(sources):
    with quail.Session(EngineConfig(model=GEMMA.name, device="h100-sxm",
                                    gpus=2), tokenizer=fake_tok) as session:
        session.register("pages", quail.DocumentProvider.from_pdfs(
            sources, id_col="doc_id", path_col="path", row_mode="page"))
        plan = session.sql(FILTER_SQL).plan()
        assert isinstance(plan, Refusal)
        assert plan.constraint == "pdf_rows_need_one_gpu"


JOIN_SQL = """
    SELECT q.id, p.doc_id FROM questions q JOIN pages p
    ON q.doc_id = p.doc_id
    AND AI.IF(PROMPT('Question: {0}\\nDoes this page answer it?\\n{1}',
                     q.question, p.document))
"""


def _register_join_tables(session, sources):
    session.register("pages", quail.DocumentProvider.from_pdfs(
        sources, id_col="doc_id", path_col="path", row_mode="page"))
    session.register("questions", quail.DocumentProvider.from_table(
        pa.table({"id": [1, 2, 3],
                  "doc_id": ["a", "a", "b"],
                  "question": ["capex?", "revenue?", "net income?"]}),
        id_col="id"))


def test_join_over_pdf_pages_anchors_on_the_pages(sources):
    from quail.physical import AiJoin

    with gemma_session() as session:
        _register_join_tables(session, sources)
        plan = session.sql(JOIN_SQL, dialect="bq").plan()
        assert not isinstance(plan, Refusal), plan
        (scan,) = [n for n in plan.nodes if isinstance(n, PDFScan)]
        assert scan.alias == "p"
        (join,) = [n for n in plan.nodes if isinstance(n, AiJoin)]
        assert join.anchor == "p"
        assert all(stage.partners == ("q",) for stage in join.stages)
        # an explicit text anchor cannot hold the pages
        forced = session.sql(JOIN_SQL.replace(
            "p.document))", "p.document), {'anchor': 'q'})"),
            dialect="bq").plan()
        assert isinstance(forced, Refusal)
        assert forced.constraint == "pdf_join_partner_unsupported"


def test_join_inputs_carry_an_image_source_for_pdf_anchors(sources):
    from test_quail_backend import graph_state

    from quail.backends.quail.executor.images import PageImages
    from quail.backends.quail.graph import execute_single_graph
    from quail.backends.quail.worker import (
        payload_documents,
        quail_runtime_payload,
    )
    from quail.builtins import built_in_registry
    from quail.execution.runner import NodeMetrics, NodeResult
    from quail.physical import AiJoin, decode_graph

    with gemma_session() as session:
        _register_join_tables(session, sources)
        query = session.sql(JOIN_SQL, dialect="bq")
        query.plan()
        request = query._prepare_physical()
    graph = decode_graph(request.plan["graph"], built_in_registry().codecs)
    payload = quail_runtime_payload(request, graph)
    payload["pre_ids"] = [1, 2]
    docs = payload_documents(payload, GEMMA)
    (join,) = [n for n in graph.nodes if isinstance(n, AiJoin)]
    seen = {}

    class CapturingExecution:
        def execute(self, node, inputs):
            if isinstance(node, AiJoin):
                seen.update(inputs)
                anchors = inputs["anchor_ids"]
                return NodeResult(
                    {f"ids:{node.anchor}": list(anchors),
                     "join_answers:0": {
                         "rows": {}, "anchor_index": list(anchors),
                         "partner_index": inputs["partner_indices"][0],
                         "anchor": node.anchor, "partners": ["q"],
                         "semantics": "full", "selectivity": None,
                         "written_pos": 0}},
                    NodeMetrics(fresh_tokens=0, extension={"answers": [{}]}))
            raise AssertionError(f"unexpected node {node}")

    state = graph_state(CapturingExecution(), docs)
    state["columns"] = payload["columns"]
    execute_single_graph(state, payload, graph)
    images = seen["images"]
    assert isinstance(images, PageImages)
    assert images.document_ids == list(seen["anchor_ids"]) == [0, 1, 2, 3, 4]
    assert images.pre_tokens == 2
    prefix = seen["prefixes"][0]
    assert prefix[:3] == [1, 2, GEMMA.image_start_id]
    assert len(prefix) == 2 + docs["p"].lengths[0]


def test_worker_lays_pdf_rows_out_as_page_prompts_with_an_image_source(sources):
    from quail.backends.quail.executor.images import PageImages
    from quail.backends.quail.graph import filter_inputs
    from quail.backends.quail.worker import (
        execute_quail_multi,
        payload_documents,
        quail_runtime_payload,
    )
    from quail.builtins import built_in_registry
    from quail.pdf.prompt import PagePrompts
    from quail.physical import AiFilter, decode_graph

    with gemma_session() as session:
        session.register("pages", quail.DocumentProvider.from_pdfs(
            sources, id_col="doc_id", path_col="path", row_mode="page"))
        query = session.sql(FILTER_SQL)
        plan = query.plan()
        request = query._prepare_physical()
    graph = decode_graph(request.plan["graph"], built_in_registry().codecs)
    payload = quail_runtime_payload(request, graph)
    assert payload["docs"] == {}
    bound, budget = payload["pdf_inputs"]["p"]
    assert isinstance(bound, PDFInput) and budget == 280
    docs = payload_documents(payload, GEMMA)
    prompts = docs["p"]
    assert isinstance(prompts, PagePrompts) and len(prompts) == 5
    (scan,) = [n for n in plan.nodes if isinstance(n, PDFScan)]
    assert sum(prompts.lengths) == scan.total_tokens
    assert prompts[0][0] == GEMMA.image_start_id
    assert prompts[0][-1] == GEMMA.image_end_id
    (node,) = [n for n in graph.nodes if isinstance(n, AiFilter)]
    state = {"docs": docs, "pre": [1, 2, 3], "filter_limit": None}
    inputs = filter_inputs(state, node, [4, 0])
    assert isinstance(inputs["images"], PageImages)
    assert inputs["images"].document_ids == [4, 0]
    assert inputs["images"].pre_tokens == 3
    assert len(inputs["documents"]) == 2
    assert list(inputs["documents"][0][:4]) == [1, 2, 3, GEMMA.image_start_id]
    text_state = {"docs": {"p": [[7, 8]]}, "pre": [], "filter_limit": None}
    assert filter_inputs(text_state, node, [0])["images"] is None
    with pytest.raises(ValueError, match="one GPU"):
        execute_quail_multi(payload, None, graph)


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
        session.register("more", quail.DocumentProvider.from_pdfs(
            sources, id_col="doc_id", path_col="path", row_mode="page"))
        with pytest.raises(CompileError, match="pages of one table"):
            session.sql("""
                SELECT p.doc_id FROM pages p JOIN more m
                ON AI_FILTER(PROMPT('{0} matches {1}?', p.document, m.document))
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
    pdf_input = PDFInput(src, pages, (PdfRowRef((0,)),), "page")
    check_scan_input(pdf, pdf_input)
    check_scan_input(text, document_input([[1, 2]]))
    with pytest.raises(TypeError, match="needs PDFInput"):
        check_scan_input(pdf, document_input([[1, 2]]))
    with pytest.raises(TypeError, match="needs TokenizedInput"):
        check_scan_input(text, pdf_input)
    with pytest.raises(ValueError, match="plans 1 documents but its input holds 2"):
        check_scan_input(text, document_input([[1], [2]]))
    with pytest.raises(ValueError, match="row_mode='page', but its input"):
        check_scan_input(pdf, PDFInput(src, pages, (PdfRowRef((0,)),), "pdf"))
    with pytest.raises(ValueError, match="one page per row"):
        PDFScan(node_id="scan:p", alias="p", input_id="p", n_docs=1,
                row_mode="page", n_pages=2, visual_tokens=280,
                pages_per_row_max=2)
    with pytest.raises(ValueError, match="positive visual token budget"):
        PDFScan(node_id="scan:p", alias="p", input_id="p", n_docs=1,
                row_mode="page", n_pages=1, pages_per_row_max=1)
    with pytest.raises(ValueError, match="only it, has a budget"):
        PdfDocuments(reading="ocr", row_mode="page", n_pages=1,
                     pages_per_row_max=1, visual_tokens=280)
