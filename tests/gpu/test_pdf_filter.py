"""A filter over PDF rows through the whole Quail path, on a GPU.

Six two-page contracts made with PDFium name a governing state in
large type on their second page. The session registers them as a PDF
provider in pdf row mode, plans the query on DiffusionGemma, renders
the pages in a process pool, embeds them through the vision tower,
runs block attention over each page, and answers one filter. Skipped
without a CUDA device; runs through
`uv run modal run experiments/run_gpu_tests.py --keyword pdf`.
"""

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs a CUDA device")

MODEL = "diffusion-gemma-26b-a4b-fp8"
STATES = ("Delaware", "New York", "Delaware", "California", "Texas", "Delaware")
QUESTION = ("Judge strictly from the contract {0}. Is this agreement governed "
            "by the laws of the State of Delaware? Answer TRUE or FALSE.")


def _write_contract(path, state):
    """A two-page agreement whose second page names its governing law."""
    from pdfs import make_text_pdf

    make_text_pdf(path, (
        ["SERVICES AGREEMENT", "between Alpha Corp. and Beta LLC.",
         "1. Services. Alpha provides the services in Exhibit A."],
        ["2. Governing Law.", "This Agreement is governed by the laws",
         f"of the State of {state}.", "3. Term. Two years from signing."],
    ), font_size=28, line_height=60)


def test_filter_over_pdf_rows_reads_the_pages(tmp_path):
    import pyarrow as pa

    import quail
    from quail.planner.plan import EngineConfig, Refusal

    paths = []
    for index, state in enumerate(STATES):
        path = tmp_path / f"ct{index}.pdf"
        _write_contract(path, state)
        paths.append(str(path))
    sources = pa.table({"id": [f"ct{i}" for i in range(len(STATES))],
                        "path": paths})
    with quail.Session(EngineConfig(
            gpus=1, model=MODEL, backend="quail", device="h100-sxm")) as sess:
        sess.register("contracts", quail.DocumentProvider.from_pdfs(
            sources, id_col="id", path_col="path", row_mode="pdf"))
        query = (sess.docs("contracts").alias("c")
                 .ai_filter(quail.prompt(QUESTION, quail.col("c.document")),
                            selectivity=0.5)
                 .select("c.id"))
        plan = query.plan()
        assert not isinstance(plan, Refusal), plan
        explained = query.explain()
        assert "row_mode=pdf" in explained and "pages=12" in explained
        result = query.run()
        rows = sorted(row[0] for row in result.to_rows())

    planted = sorted(f"ct{i}" for i, state in enumerate(STATES)
                     if state == "Delaware")
    print("planted", planted, "got", rows, flush=True)
    images = result.report["backend_metrics"]["images"]
    print("image metrics", images, flush=True)
    assert images["c"]["pages_rendered"] == 12
    assert result.report["fresh_tokens"] >= 12 * 266
    agree = len(set(rows) & set(planted)) + (
        len(STATES) - len(set(rows) | set(planted)))
    assert agree >= 5, (rows, planted)


JOIN_PROMPT = ("Judge strictly from the page.\n\nQuestion: {0}\n\n"
               "Page: {1}\n\nDoes this page answer the question? "
               "Answer TRUE or FALSE.")


def test_join_over_pdf_pages_finds_the_page_that_answers(tmp_path):
    """Each contract's questions against its own pages: one page answers."""
    import pyarrow as pa

    import quail
    from quail.planner.plan import EngineConfig, Refusal

    paths = []
    for index, state in enumerate(STATES):
        path = tmp_path / f"ct{index}.pdf"
        _write_contract(path, state)
        paths.append(str(path))
    sources = pa.table({"id": [f"ct{i}" for i in range(len(STATES))],
                        "path": paths})
    questions = pa.table({
        "id": [f"q{i}" for i in range(len(STATES))],
        "contract": [f"ct{i}" for i in range(len(STATES))],
        "question": ["Which state's law governs this agreement?"] * len(STATES),
    })
    with quail.Session(EngineConfig(
            gpus=1, model=MODEL, backend="quail", device="h100-sxm")) as sess:
        sess.register("pages", quail.DocumentProvider.from_pdfs(
            sources, id_col="id", path_col="path", row_mode="page"))
        sess.register("questions", quail.DocumentProvider.from_table(
            questions, id_col="id"))
        query = sess.sql("""
            SELECT q.id, p.page_number FROM questions q JOIN pages p
            ON q.contract = p.id
            AND AI.IF(PROMPT(%r, q.question, p.document))
        """ % JOIN_PROMPT, dialect="bq")
        plan = query.plan()
        assert not isinstance(plan, Refusal), plan
        explained = query.explain()
        assert "row_mode=page" in explained and "pages=12" in explained
        result = query.run()
        pairs = sorted(tuple(row) for row in result.to_rows())

    print("pairs", pairs, flush=True)
    images = result.report["backend_metrics"]["images"]
    print("image metrics", images, flush=True)
    assert images["p"]["pages_rendered"] == 12
    # the governing law is on page 2 of every contract
    got = {}
    for question, page_number in pairs:
        got.setdefault(question, set()).add(page_number)
    right = sum(got.get(f"q{i}") == {2} for i in range(len(STATES)))
    assert right >= 4, pairs
