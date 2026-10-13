"""CPU checks that every QUAIL-B query builds and plans on Quail."""

from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
from fakes import letter_tokens

import quail
from quail.bench.quailb import (
    _submission_to_answer_s,
    queries,
    register_tables,
)
from quail.logical import Aggregate
from quail.planner.plan import EngineConfig, Refusal
from quail_b import data
from quail_b.data import ASPECTS, SCENARIOS
from quail_b.queries import QUERY_ORDER


def _standin_sets(tmp_path):
    """Write small parquet files with the benchmark table schemas."""
    six = range(6)
    tables = {
        "reviews": {"body": [f"review text {i}" for i in range(12)]},
        "aspects": {"aspect": ASPECTS},
        "reports": {"report": [f"medical report {i}" for i in range(8)]},
        "terms": {"term": [f"reaction {i}" for i in six]},
        "claims": {
            "claim": [f"claim {i}" for i in six],
            "label": ["SUPPORTS" if i % 2 == 0 else "REFUTES" for i in six],
            # a claim's page is an evidence id, as in the real corpus
            "evidence_wiki_url": [f"evidence{i % 4}" for i in six],
        },
        "evidence": {"text": [f"Wikipedia passage {i}" for i in six]},
        "citation_contexts": {
            "destination_context": [f"citation excerpt {i}" for i in six],
            "cited_passage_ids": [[f"passage-{i}"] for i in six],
        },
        "citation_passages": {
            "passage_text": [f"cited passage {i}" for i in six],
            "passage_ids": [[f"passage-{i}"] for i in six],
        },
        "agent_traces": {
            "trace": [f"agent trace {i}" for i in six],
            "trajectory_id": [f"at{i}" for i in six],
            "turn_index": [5] * 6,
            "token_count": [3] * 6,
        },
        "policies": {"policy_text": [f"privacy policy text {i}" for i in range(8)]},
        "scenarios": {"scenario": SCENARIOS},
    }
    for name, columns in tables.items():
        rows = len(next(iter(columns.values())))
        ids = [f"{name}{i}" for i in range(rows)]
        pq.write_table(pa.table({"id": ids, **columns}), tmp_path / f"{name}.parquet")
    # the trace, sales, and contract tables, with every column of their
    # published schema; a row's key columns name other rows of the same
    # stand-in, as in the real corpus
    schemas = {
        "support_traces": data.SUPPORT_TRACE_SCHEMA,
        "support_messages": data.MESSAGE_SCHEMA,
        "issue_runs": data.ISSUE_RUN_SCHEMA,
        "issue_messages": data.MESSAGE_SCHEMA,
        "wrench_runs": data.WRENCH_RUN_SCHEMA,
        "wrench_steps": data.WRENCH_STEP_SCHEMA,
        "sales_calls": data.SALES_CALL_SCHEMA,
        "contracts": data.CONTRACT_SCHEMA,
        "clause_types": data.CLAUSE_TYPE_SCHEMA,
    }
    for name, schema in schemas.items():
        columns = {}
        for field in schema:
            if field.name == "id":
                columns["id"] = [f"{name}{i}" for i in six]
            elif pa.types.is_string(field.type):
                columns[field.name] = [f"{name} {field.name} {i}" for i in six]
            elif pa.types.is_floating(field.type):
                columns[field.name] = [float(i) for i in six]
            else:
                columns[field.name] = [i % 2 for i in six]
        pq.write_table(pa.table(columns, schema=schema),
                       tmp_path / f"{name}.parquet")


def test_all_queries_compile_and_plan_and_answer_timing_adds_common_work(tmp_path):
    _standin_sets(tmp_path)
    expected = {
        *(f"IMDB-{i}" for i in range(1, 11)),
        *(f"BIO-{i}" for i in range(1, 5)),
        *(f"FEV-{i}" for i in range(1, 11)),
        *(f"LEP-{i}" for i in range(1, 6)),
        "AGENT-1", "AGENT-2",
        "SUPPORT-2", "SUPPORT-5", "RUNS-1", "RUNS-3", "WRENCH-1",
        "SALES-1", "SALES-3", "SALES-4", "CUAD-5", "CUAD-6", "CUAD-13",
        "CUAD-14",
        "PRIV-1", "PRIV-2",
    }
    classify = {*(f"IMDB-{i}" for i in range(11, 16)), "BIO-5", "BIO-6",
                "FEV-11", "LEP-6", "AGENT-3", "AGENT-4", "AGENT-5",
                "SUPPORT-3", "WRENCH-3"}
    extract = {f"CUAD-{i}" for i in (1, 2, 3, 4, 7, 8, 9, 10)}
    # relational steps after the AI operators; the request backends
    # refuse the ones with ORDER BY, DISTINCT, OFFSET, or GROUP BY
    relational = {
        *(f"REL-AGENT-{i}" for i in range(1, 8)),
        "SUPPORT-1", "SUPPORT-4", "SUPPORT-6", "RUNS-2", "RUNS-4",
        "WRENCH-2", "WRENCH-4", "WRENCH-5", "SALES-2", "SALES-5",
        "CUAD-11", "CUAD-12", "CUAD-15",
    }
    assert set(QUERY_ORDER) == (
        expected | classify | extract | relational) - {"PRIV-1", "PRIV-2"}
    for backend in ("quail", "stock_vllm", "pipelined_vllm", "pipelined_sglang"):
        with quail.Session(
            EngineConfig(gpus=1, model="qwen3-4b-fp8", backend=backend,
                         device="h100-sxm"),
            tokenizer=letter_tokens,
        ) as sess:
            register_tables(sess, tmp_path)
            qdefs = queries(sess)
            # every backend lists every query; SGLang refuses the
            # classification and extraction queries at plan time, since
            # its engine returns no decoded answer text.
            assert set(qdefs) == expected | classify | extract | relational, (
                backend)
            for qid, (_, build) in qdefs.items():
                case = f"{backend} {qid}"
                query = build()
                operators = query.logical.operators()
                selectivities = [
                    predicate.selectivity for chain in operators.filters.values()
                    for predicate in chain]
                selectivities += [join.selectivity for join in operators.joins]
                assert all((s is None) == qid.startswith("PRIV-")
                           for s in selectivities), case
                plan = query.plan()
                root = query.logical.result
                sorts = bool(root.order or root.distinct or root.offset
                             or isinstance(root.input, Aggregate))
                assert sorts <= (qid in relational), case
                if backend != "quail" and sorts:
                    assert isinstance(plan, Refusal), case
                    assert plan.constraint == "sort_needs_quail_backend", case
                    continue
                if backend == "pipelined_sglang" and qid in classify:
                    assert isinstance(plan, Refusal), case
                    assert plan.constraint == "classify_needs_quail_backend", case
                    continue
                if backend == "pipelined_sglang" and qid in extract:
                    assert isinstance(plan, Refusal), case
                    assert plan.constraint == "extract_needs_quail_backend", case
                    continue
                assert not isinstance(plan, Refusal), f"{case} refused: {plan}"
                assert plan.settings["order_rule"] == "by_cost", case
                assert "physical:" in query.explain(), case

    report = {
        "model_wall_s": 10.0,
        "finish_s": 0.5,
        "input_ready_s": 2.0,
        "physical_prepare_s": 0.25,
    }
    assert _submission_to_answer_s(
        "quail", report, frontend_s=1.0, answer_prepare_s=0.75
    ) == 14.5
    assert _submission_to_answer_s(
        "pipelined_vllm", report, frontend_s=1.0, answer_prepare_s=0.75
    ) == 11.25


def test_kernel_cache_files_counts_each_cache_directory(tmp_path, monkeypatch):
    from quail.bench.quailb import kernel_cache_files

    monkeypatch.setenv("QUAIL_CACHE_DIR", str(tmp_path / "missing"))
    assert kernel_cache_files() == {}
    (tmp_path / "triton" / "a").mkdir(parents=True)
    (tmp_path / "triton" / "a" / "k.cubin").write_bytes(b"")
    (tmp_path / "triton" / "b").mkdir()
    (tmp_path / "deep_gemm").mkdir()
    (tmp_path / "marker.json").write_text("{}")
    monkeypatch.setenv("QUAIL_CACHE_DIR", str(tmp_path))
    assert kernel_cache_files() == {"deep_gemm": 0, "triton": 2}


def test_request_benchmark_counts_prompt_tokens_without_decode_tokens(monkeypatch):
    from quail.bench import quailb

    result = SimpleNamespace(report={
        "model_wall_s": 1.0, "finish_s": 0.0,
        "fresh_tokens": 110, "cached_tokens": 20,
        "backend_metrics": {"prompt_tokens": 100},
    })
    query = SimpleNamespace(run=lambda: result)
    output = SimpleNamespace(runtime_s=0.0, measurements={})
    monkeypatch.setattr(quailb, "kernel_cache_files", lambda: {})
    monkeypatch.setattr(quailb, "_build", lambda *args: query)
    monkeypatch.setattr(quailb, "run_output", lambda *args: output)
    monkeypatch.setattr(quailb, "join_anchors", lambda *args: {})
    monkeypatch.setattr(quailb, "prompt_pieces", lambda *args: {})
    session = SimpleNamespace(config=SimpleNamespace(backend="stock_vllm"))
    actual = quailb.run_query(session, SimpleNamespace(info=None), {})
    assert actual.measurements["input_tokens"] == 100
