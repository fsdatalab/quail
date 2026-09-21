"""CPU checks that every QUAIL-B query builds and plans on Quail."""

import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pypdfium2
import pytest

import quail
from quail.bench.quailb import queries, register_tables
from quail.planner.plan import EngineConfig, Refusal
from quail_b.data import ASPECTS, SCENARIOS
from quail_b.queries import QUERY_ORDER

CUAD_QUERIES = {f"CUAD-{i}" for i in range(1, 6)}


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
    return tmp_path


def _standin_contracts(tmp_path, page_counts):
    """Blank contract PDFs under files/, with both file-backed tables."""
    files = tmp_path / "files"
    files.mkdir(exist_ok=True)
    contracts, pages = [], []
    for index, count in enumerate(page_counts):
        document = pypdfium2.PdfDocument.new()
        for _ in range(count):
            document.new_page(612, 792)
        path = files / f"ct{index}.pdf"
        document.save(str(path))
        document.close()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
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
            *CUAD_QUERIES,
        }
        assert set(qdefs) == expected
        assert set(QUERY_ORDER) == expected - {"PRIV-1", "PRIV-2"}
        for qid, (_, build) in qdefs.items():
            query = build()
            if qid in CUAD_QUERIES:
                # PDF rows need a model that takes images
                plan = query.plan()
                assert isinstance(plan, Refusal), qid
                assert "takes text only" in " ".join(plan.reasons), qid
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
    for qid in sorted(CUAD_QUERIES):
        query = queries(sess)[qid][1]()
        plan = query.plan()
        assert not isinstance(plan, Refusal), f"{qid} refused: {plan}"
        explained = query.explain()
        assert "physical:" in explained, qid
        mode = "page" if qid in ("CUAD-1", "CUAD-2") else "pdf"
        assert f"row_mode={mode}" in explained, qid


def test_page_rows_register_through_a_symlinked_data_directory(tmp_path):
    """A mounted volume's real path differs from the reference; rows still match."""
    from quail.bench.quailb import pdf_provider, read_tables

    data = tmp_path / "data"
    data.mkdir()
    _standin_contracts(data, page_counts=(2, 3))
    link = tmp_path / "mount"
    link.symlink_to(data, target_is_directory=True)
    tables = read_tables(link, ["contract_pages"])
    provider = pdf_provider(tables["contract_pages"])
    assert provider.statistics().row_count == 5
    # a table missing one page of a file no longer matches the files' rows
    with pytest.raises(ValueError, match="every page of each file"):
        pdf_provider(tables["contract_pages"].slice(0, 4))
