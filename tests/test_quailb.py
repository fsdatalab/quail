"""CPU checks that every QUAIL-B query builds and plans on Quail."""

import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
from pdfs import make_text_pdf

import quail
from quail.bench.quailb import queries, register_tables
from quail.physical import PdfTextScan
from quail.planner.plan import EngineConfig, Refusal
from quail_b.data import ASPECTS, SCENARIOS
from quail_b.queries import QUERY_ORDER

CUAD_QUERIES = {f"CUAD-{i}" for i in range(1, 6)}
FIN_QUERIES = {"FIN-1", "FIN-2"}
PDF_QUERIES = CUAD_QUERIES | FIN_QUERIES


def _standin_sets(tmp_path):
    """Write small parquet files with the benchmark table schemas."""
    def write(name, col, values):
        pq.write_table(pa.table({
            "id": [f"{name}{i}" for i in range(len(values))],
            col: values,
        }), tmp_path / f"{name}.parquet")

    write("reviews", "body", [f"review text {i}" for i in range(12)])
    write("aspects", "aspect", ASPECTS)
    write("reports", "report", [f"medical report {i}" for i in range(8)])
    write("terms", "term", [f"reaction {i}" for i in range(6)])
    pq.write_table(pa.table({
        "id": [f"cl{i}" for i in range(6)],
        "claim": [f"claim {i}" for i in range(6)],
        "label": ["SUPPORTS" if i % 2 == 0 else "REFUTES"
                  for i in range(6)],
        # a claim's page is an evidence id, as in the real corpus
        "evidence_wiki_url": [f"evidence{i % 4}" for i in range(6)],
    }), tmp_path / "claims.parquet")
    write("evidence", "text", [f"Wikipedia passage {i}" for i in range(6)])
    pq.write_table(pa.table({
        "id": [f"lc{i}" for i in range(6)],
        "destination_context": [f"citation excerpt {i}" for i in range(6)],
        "cited_passage_ids": [[f"passage-{i}"] for i in range(6)],
    }), tmp_path / "citation_contexts.parquet")
    pq.write_table(pa.table({
        "id": [f"lp{i}" for i in range(6)],
        "passage_text": [f"cited passage {i}" for i in range(6)],
        "passage_ids": [[f"passage-{i}"] for i in range(6)],
    }), tmp_path / "citation_passages.parquet")
    pq.write_table(pa.table({
        "id": [f"at{i}-t005" for i in range(6)],
        "trace": [f"agent trace {i}" for i in range(6)],
        "trajectory_id": [f"at{i}" for i in range(6)],
        "turn_index": [5] * 6,
        "token_count": [3] * 6,
    }), tmp_path / "agent_traces.parquet")
    write("policies", "policy_text",
          [f"privacy policy text {i}" for i in range(8)])
    pq.write_table(pa.table({
        "id": [f"sc{i}" for i in range(len(SCENARIOS))],
        "scenario": SCENARIOS,
    }), tmp_path / "scenarios.parquet")
    _standin_contracts(tmp_path, page_counts=(2, 1, 40))
    _standin_filings(tmp_path, page_counts=(3, 2))
    return tmp_path


def _text_pdf(path, page_count) -> str:
    """Write a PDF of letter pages with a line of text each; return its sha256."""
    make_text_pdf(path, [f"{path.stem} page {page} says hello"
                         for page in range(1, page_count + 1)])
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _standin_filings(tmp_path, page_counts):
    """Filing PDFs under files/, one question per filing."""
    files = tmp_path / "files"
    files.mkdir(exist_ok=True)
    questions, pages = [], []
    for index, count in enumerate(page_counts):
        digest = _text_pdf(files / f"f{index}.pdf", count)
        questions.append({
            "id": f"fq{index}", "financebench_id": f"financebench_id_{index:05d}",
            "filing": f"f{index}", "doc_name": f"FILING{index}", "company": "Co",
            "question_type": "metrics-generated",
            "question": f"question {index}", "answer": "42",
            "evidence_pages": [1]})
        pages += [{
            "id": f"f{index}p{page}", "filing": f"f{index}",
            "doc_name": f"FILING{index}", "page_number": page,
            "page_count": count, "pdf_sha256": digest,
            "document": f"files/f{index}.pdf#page={page}"}
            for page in range(1, count + 1)]
    pq.write_table(pa.Table.from_pylist(questions),
                   tmp_path / "filing_questions.parquet")
    pq.write_table(pa.Table.from_pylist(pages), tmp_path / "filing_pages.parquet")


def _standin_contracts(tmp_path, page_counts):
    """Contract PDFs under files/, with both file-backed tables."""
    files = tmp_path / "files"
    files.mkdir(exist_ok=True)
    contracts, pages = [], []
    for index, count in enumerate(page_counts):
        digest = _text_pdf(files / f"ct{index}.pdf", count)
        contracts.append({
            "id": f"ct{index}", "title": f"contract {index}",
            "page_count": count, "pdf_sha256": digest,
            "document": f"files/ct{index}.pdf", "clauses": ["Exclusivity"]})
        pages += [{
            "id": f"ct{index}p{page}", "contract_id": f"ct{index}",
            "page_number": page, "pdf_sha256": digest,
            "document": f"files/ct{index}.pdf#page={page}", "clauses": []}
            for page in range(1, count + 1)]
    pq.write_table(pa.Table.from_pylist(contracts), tmp_path / "contracts.parquet")
    pq.write_table(pa.Table.from_pylist(pages), tmp_path / "contract_pages.parquet")


def test_all_queries_compile_and_plan(tmp_path):
    for backend in [
    "quail", "stock_vllm", "pipelined_vllm", "pipelined_sglang",
]:
        _standin_sets(tmp_path)
        sess = quail.Session(
            EngineConfig(
                gpus=1,
                model="qwen3-4b-fp8",
                backend=backend,
                device="h100-sxm",
            ),
            tokenizer=lambda text: list(text.encode()),
        )
        register_tables(sess, tmp_path)
        qdefs = queries(sess)
        expected = {
            *(f"IMDB-{i}" for i in range(1, 11)),
            *(f"BIO-{i}" for i in range(1, 5)),
            *(f"FEV-{i}" for i in range(1, 11)),
            *(f"LEP-{i}" for i in range(1, 6)),
            "AGENT-1", "AGENT-2",
            "PRIV-1", "PRIV-2",
            *PDF_QUERIES,
        }
        assert set(qdefs) == expected
        assert set(QUERY_ORDER) == expected - {"PRIV-1", "PRIV-2"}
        for qid, (_, build) in qdefs.items():
            query = build()
            if qid in PDF_QUERIES:
                # a text model reads the pages as extracted text, on
                # every backend, with the same query
                plan = query.plan()
                assert not isinstance(plan, Refusal), f"{qid} refused: {plan}"
                assert any(isinstance(node, PdfTextScan)
                           for node in plan.nodes), qid
                assert "read as text, ocr=off" in query.explain(), qid
                continue
            operators = query.logical.operators()
            filters, joins = operators.filters, operators.joins
            predicates = [predicate for chain in filters.values()
                          for predicate in chain]
            if qid.startswith("PRIV-"):
                assert all(predicate.selectivity is None
                           for predicate in predicates), qid
                assert all(join.selectivity is None for join in joins), qid
            else:
                assert all(predicate.selectivity is not None
                           for predicate in predicates), qid
                assert all(join.selectivity is not None for join in joins), qid
            plan = query.plan()
            assert not isinstance(plan, Refusal), f"{qid} refused: {plan}"
            assert plan.settings["order_rule"] == "by_cost", qid
            assert "physical:" in query.explain(), qid


def test_cuad_queries_plan_on_an_image_model_over_bounded_pdf_rows(tmp_path):
    _standin_sets(tmp_path)
    sess = quail.Session(
        EngineConfig(
            gpus=1,
            model="diffusion-gemma-26b-a4b-fp8",
            backend="quail",
            device="h100-sxm",
        ),
        tokenizer=lambda text: list(text.encode()),
    )
    register_tables(sess, tmp_path)
    # the bounded contracts are their own provider: the 40-page one is out
    assert "contracts[page_count<=32]" in sess.catalog
    assert "contracts" not in sess.catalog
    assert sess.catalog.get("contracts[page_count<=32]").statistics().row_count == 2
    assert sess.catalog.get("contract_pages").statistics().row_count == 43
    # the page rows keep the benchmark's join columns beside the pages
    assert "filing" in sess.catalog.get("filing_pages").columns
    for qid in sorted(PDF_QUERIES):
        query = queries(sess)[qid][1]()
        plan = query.plan()
        assert not isinstance(plan, Refusal), f"{qid} refused: {plan}"
        explained = query.explain()
        assert "physical:" in explained, qid
        mode = "pdf" if qid in ("CUAD-3", "CUAD-4", "CUAD-5") else "page"
        assert f"row_mode={mode}" in explained, qid
        if qid in FIN_QUERIES:
            # the pages anchor the join; the questions are the partners
            assert "anchor=p" in explained, qid


def test_page_rows_keep_their_benchmark_ids_through_a_symlinked_directory(
        tmp_path):
    """Page rows carry the table's ids; a mount's real path changes nothing."""
    from quail.bench.quailb import pdf_provider, read_tables
    from quail.catalog import ScanRequest

    data = tmp_path / "data"
    data.mkdir()
    _standin_contracts(data, page_counts=(2, 3))
    link = tmp_path / "mount"
    link.symlink_to(data, target_is_directory=True)
    tables = read_tables(link, ["contract_pages"])
    # any subset of pages, in the table's order
    provider = pdf_provider(tables["contract_pages"].take([4, 0, 3]))
    assert provider.statistics().row_count == 3
    rows = provider.scan(ScanRequest(columns=("id", "page", "page_count"),
                                     filter=None, limit=None)).read_all()
    assert rows.column("id").to_pylist() == ["ct1p3", "ct0p1", "ct1p2"]
    assert rows.column("page").to_pylist() == [3, 1, 2]
    assert rows.column("page_count").to_pylist() == [3, 2, 3]
    assert [page.page_number for page in provider.pdf_input().row_pages(0)] == [3]
