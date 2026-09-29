"""CPU checks that every QUAIL-B query builds and plans on Quail."""

import pyarrow as pa
import pyarrow.parquet as pq
from fakes import letter_tokens

import quail
from quail.bench.quailb import (
    _submission_to_answer_s,
    queries,
    register_tables,
)
from quail.planner.plan import EngineConfig, Refusal
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


def test_all_queries_compile_and_plan_and_answer_timing_adds_common_work(tmp_path):
    _standin_sets(tmp_path)
    expected = {
        *(f"IMDB-{i}" for i in range(1, 11)),
        *(f"BIO-{i}" for i in range(1, 5)),
        *(f"FEV-{i}" for i in range(1, 11)),
        *(f"LEP-{i}" for i in range(1, 6)),
        "AGENT-1", "AGENT-2",
        "PRIV-1", "PRIV-2",
    }
    classify = {*(f"IMDB-{i}" for i in range(11, 16)), "BIO-5", "BIO-6",
                "FEV-11", "LEP-6", "AGENT-3", "AGENT-4"}
    assert set(QUERY_ORDER) == (expected | classify) - {"PRIV-1", "PRIV-2"}
    for backend in ("quail", "stock_vllm", "pipelined_vllm", "pipelined_sglang"):
        with quail.Session(
            EngineConfig(gpus=1, model="qwen3-4b-fp8", backend=backend,
                         device="h100-sxm"),
            tokenizer=letter_tokens,
        ) as sess:
            register_tables(sess, tmp_path)
            qdefs = queries(sess)
            # every backend lists every query; SGLang refuses the
            # classification queries at plan time, since it returns
            # no named tokens' log probabilities. A lone classification
            # chain plans on Quail's classify planner, in written order
            runnable = {"IMDB-11", "IMDB-14", "BIO-5", "FEV-11", "AGENT-4"}
            assert set(qdefs) == expected | classify, backend
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
                if backend == "pipelined_sglang" and qid in classify:
                    assert isinstance(plan, Refusal), case
                    assert plan.constraint == "classify_needs_quail_backend", case
                    continue
                assert not isinstance(plan, Refusal), f"{case} refused: {plan}"
                assert plan.settings["order_rule"] == (
                    "written order" if qid in runnable and backend == "quail"
                    else "by_cost"), case
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
