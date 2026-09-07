"""CPU checks for the QUAIL-B query catalog and table schemas."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quail
from quail.bench.quailb import queries, register_privacy_sets, register_sets
from quail.planner.decide import collect_operators
from quail.planner.plan import EngineConfig, Refusal
from quailb.data import (
    AGENT_TRACE_DOCUMENTS,
    AGENT_TRACE_MAX_TOKENS,
    AGENT_TRACE_TURN_INTERVAL,
    ASPECTS,
    LEPARD_POSITIVE_PAIRS,
    SCENARIOS,
    SETS,
    _agent_snapshot_boundaries,
    _agent_trace_rows,
    _lepard_documents,
    _n_agent_documents,
    _n_lepard_pairs,
    _sample_lepard_pairs,
)
from quailb.queries import (
    QUERY_FAMILY_WORKLOADS,
    QUERY_ORDER,
    query_family_name,
    split_query_families,
    split_query_ids,
)


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
        "evidence_wiki_url": [f"Page_{i}" for i in range(6)],
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
    return tmp_path


@pytest.mark.parametrize("backend", [
    "quail", "stock_vllm", "pipelined_vllm", "pipelined_sglang",
])
def test_all_queries_compile_and_plan(tmp_path, backend):
    _standin_sets(tmp_path)
    sess = quail.Session(EngineConfig(gpus=1, backend=backend),
                         tokenizer=lambda text: list(text.encode()))
    register_sets(sess, tmp_path)
    register_privacy_sets(sess, tmp_path)
    qdefs = queries(sess)
    expected = {
        *(f"IMDB-{i}" for i in range(1, 11)),
        *(f"BIO-{i}" for i in range(1, 4)),
        *(f"FEV-{i}" for i in range(1, 10)),
        *(f"LEP-{i}" for i in range(1, 9)),
        "AGENT-1", "AGENT-2",
        "PRIV-1", "PRIV-2",
    }
    assert set(qdefs) == expected
    assert set(QUERY_ORDER) == expected - {"PRIV-1", "PRIV-2"}
    for qid, (_, build) in qdefs.items():
        query = build()
        _, filters, joins = collect_operators(query.logical)
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
        expected_order = (
            "as_written" if qid.startswith("PRIV-")
            else "by_cost"
        )
        assert plan.settings["order_rule"] == expected_order, qid
        assert "physical:" in query.explain(), qid


def test_set_table_matches_design():
    assert SETS == {
        "reviews": 50_000,
        "reports": 5_000,
        "claims": 5_000,
        "agent_traces": AGENT_TRACE_DOCUMENTS,
        "policies": 1_000_000,
    }
    assert LEPARD_POSITIVE_PAIRS == 5_000
    assert _n_lepard_pairs(0.1) == 500
    assert len(ASPECTS) == 12
    assert len(SCENARIOS) == 100
    assert AGENT_TRACE_TURN_INTERVAL == 5
    assert AGENT_TRACE_MAX_TOKENS == 24_000
    assert _n_agent_documents(1.0) == 17_718
    assert _n_agent_documents(0.1) == 1_772


def test_agent_snapshots_include_every_fifth_turn_and_following_tool():
    messages = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "issue"},
    ]
    for turn in range(1, 12):
        messages.append({"role": "assistant", "content": f"action {turn}"})
        messages.append({"role": "tool", "content": f"result {turn}"})

    text, boundaries = _agent_snapshot_boundaries(messages)
    snapshots = [(turn, text[:end]) for turn, end in boundaries]

    assert [turn for turn, _snapshot in snapshots] == [5, 10]
    assert snapshots[0][1].endswith("[TOOL]\nresult 5")
    assert snapshots[1][1].startswith(snapshots[0][1])
    assert snapshots[1][1].endswith("[TOOL]\nresult 10")


class _WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()


def test_agent_trace_rows_have_unique_ids_and_cumulative_text():
    messages = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "issue"},
    ]
    for turn in range(1, 11):
        messages.append({"role": "assistant", "content": f"action {turn}"})
        messages.append({"role": "tool", "content": f"result {turn}"})

    rows = _agent_trace_rows(7, messages, _WordTokenizer())

    assert [row["id"] for row in rows] == ["at0007-t005", "at0007-t010"]
    assert [row["turn_index"] for row in rows] == [5, 10]
    assert rows[1]["trace"].startswith(rows[0]["trace"])
    assert rows[0]["token_count"] < rows[1]["token_count"]


def test_agent_trace_rows_require_a_user_issue():
    messages = [
        {"role": "system", "content": "Solve the issue."},
        {"role": "assistant", "content": "action"},
    ]

    assert _agent_trace_rows(7, messages, _WordTokenizer()) == []


def test_lepard_samples_pairs_before_deduplicating_documents():
    context_a = "A" * 60
    context_b = "B" * 60
    context_c = "C" * 60
    rows = [
        ("d1", context_a, "p1"),
        ("d1", context_a, "p1"),
        ("d1", context_a, "p2"),
        ("d2", context_b, "p1"),
        ("d3", context_c, "missing"),
    ]
    passages = {"p1": "shared passage", "p2": "shared passage"}

    pairs = _sample_lepard_pairs(rows, passages, 10)
    contexts, passage_rows = _lepard_documents(pairs)

    assert len(pairs) == 3
    assert {row["destination_context"]: row["cited_passage_ids"]
            for row in contexts} == {
        context_a: ["p1", "p2"],
        context_b: ["p1"],
    }
    assert passage_rows == [
        {"id": "lp0", "passage_text": "shared passage",
         "passage_ids": ["p1", "p2"]},
    ]


def test_lepard_pair_sample_is_stable_and_nested():
    rows = [(f"d{i}", f"context {i} " + "x" * 50, f"p{i}")
            for i in range(20)]
    passages = {f"p{i}": f"passage {i}" for i in range(20)}

    small = _sample_lepard_pairs(rows, passages, 5)
    large = _sample_lepard_pairs(reversed(rows), passages, 10)

    assert small == large[:5]


def test_parallel_query_split_matches_stock_vllm():
    assert split_query_ids(QUERY_ORDER, 4) == (
        QUERY_ORDER[0:8],
        QUERY_ORDER[8:16],
        QUERY_ORDER[16:24],
        QUERY_ORDER[24:32],
    )


def test_query_family_split_matches_benchmark_catalog():
    assert QUERY_FAMILY_WORKLOADS == {
        "IMDB": "imdb",
        "BIO": "biodex",
        "FEV": "fever",
        "LEP": "lepard",
        "AGENT": "agent",
    }
    assert split_query_families(QUERY_ORDER) == (
        QUERY_ORDER[0:10],
        QUERY_ORDER[10:13],
        QUERY_ORDER[13:22],
        QUERY_ORDER[22:30],
        QUERY_ORDER[30:32],
    )
    assert query_family_name(QUERY_ORDER[0:10]) == "imdb"


def test_query_family_rejects_mixed_or_unknown_queries():
    with pytest.raises(ValueError, match="expected one query family"):
        query_family_name(("IMDB-1", "BIO-1"))
    with pytest.raises(ValueError, match="unknown query family"):
        split_query_families(("OTHER-1",))
