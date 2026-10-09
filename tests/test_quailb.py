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
from quail_b.queries import QUERY_ORDER, pending_query_ids
from quail_b.queries import queries as query_specs


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
        "support_traces": {
            "request": [f"request {i}" for i in six],
            "transcript": [f"conversation {i}" for i in six],
            "message_count": pa.array([5] * 6, pa.int32()),
            "task_id": [f"task{i // 2}" for i in six],
            "domain": ["airline"] * 6, "model": ["gpt-4o"] * 6,
            "trial": pa.array([0] * 6, pa.int32()), "reward": [1.0, 0.0] * 3,
        },
        "support_messages": _messages("support_messages"),
        "issue_runs": {
            "request": [f"issue {i}" for i in six],
            "transcript": [f"run {i}" for i in six],
            "message_count": pa.array([5] * 6, pa.int32()),
            "instance_id": [f"issue{i // 2}" for i in six], "repo": ["r"] * 6,
            "resolved": pa.array([1, 0] * 3, pa.int32()),
            "token_count": pa.array([3] * 6, pa.int32()),
        },
        "issue_messages": _messages("issue_messages"),
        "wrench_runs": {
            "task_id": [f"task{i // 2}" for i in six],
            "model": ["gpt-5.4", "gemini-3.1-pro"] * 3,
            "mode": ["hack", "baseline"] * 3,
            "transcript": [f"agent run {i}" for i in six],
            "step_count": pa.array([3] * 6, pa.int32()),
            "token_count": pa.array([40] * 6, pa.int32()),
        },
        "wrench_steps": {
            "run_id": [f"wrench_runs{i // 2}" for i in six],
            "step_index": pa.array([1, 2] * 3, pa.int32()),
            "model": ["gpt-5.4"] * 6,
            "text": [f"agent step {i}" for i in six],
        },
    }
    for name, columns in tables.items():
        rows = len(next(iter(columns.values())))
        ids = [f"{name}{i}" for i in range(rows)]
        pq.write_table(pa.table({"id": ids, **columns}), tmp_path / f"{name}.parquet")


def _messages(table):
    """A messages table in the layout `quail.trace_tables` writes."""
    six = range(6)
    before = [None if i == 0 else f"{table}{i - 1}" for i in six]
    return {
        "trace_id": [f"trace{i // 3}" for i in six],
        "turn_index": pa.array(list(six), pa.int32()),
        "role": ["assistant" if i % 2 == 0 else "user" for i in six],
        "content": [f"message {i}" for i in six],
        "tool_call_id": pa.array([None] * 6, pa.string()),
        "prev_id": before,
        "prev_user_id": pa.array([None] * 6, pa.string()),
        "prev_assistant_id": before,
    }


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
                "FEV-11", "LEP-6", "AGENT-3", "AGENT-4", "AGENT-5"}
    relational = {f"REL-AGENT-{i}" for i in range(1, 8)}
    assert set(QUERY_ORDER) == (
        expected | classify | relational) - {"PRIV-1", "PRIV-2"}
    # queries whose labels are pending are listed and built like the rest
    specs = query_specs(include_privacy=True, include_pending=True)
    pending = set(pending_query_ids())
    for qid in pending:
        info = specs[qid].info
        if info.relational:
            relational.add(qid)
        elif info.classifies:
            classify.add(qid)
        else:
            expected.add(qid)
    for backend in ("quail", "stock_vllm", "pipelined_vllm", "pipelined_sglang"):
        with quail.Session(
            EngineConfig(gpus=1, model="qwen3-4b-fp8", backend=backend,
                         device="h100-sxm"),
            tokenizer=letter_tokens,
        ) as sess:
            register_tables(sess, tmp_path)
            qdefs = queries(sess)
            # every backend lists every query; SGLang refuses the
            # classification queries at plan time, since its engine
            # returns no decoded answer text.
            assert set(qdefs) == expected | classify | relational, backend
            for qid, (_, build) in qdefs.items():
                case = f"{backend} {qid}"
                query = build()
                operators = query.logical.operators()
                selectivities = [
                    predicate.selectivity for chain in operators.filters.values()
                    for predicate in chain]
                selectivities += [join.selectivity for join in operators.joins]
                assert all((s is None) == (qid.startswith("PRIV-")
                                           or qid in pending)
                           for s in selectivities), case
                plan = query.plan()
                # a regular predicate alone runs on every backend; a sort or
                # an aggregate runs on Quail only
                if backend != "quail" and specs[qid].info.tail:
                    assert isinstance(plan, Refusal), case
                    assert plan.constraint == "sort_needs_quail_backend", case
                    continue
                if backend == "pipelined_sglang" and qid in classify:
                    assert isinstance(plan, Refusal), case
                    assert plan.constraint == "classify_needs_quail_backend", case
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
