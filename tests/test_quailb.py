"""CPU checks for the QUAIL-B query catalog and table schemas."""

import pyarrow as pa
import pyarrow.parquet as pq

import quail
from quail.bench.quailb import (
    ASPECTS,
    LEPARD_POSITIVE_PAIRS,
    QUERY_ORDER,
    SCENARIOS,
    SETS,
    _lepard_documents,
    _n_lepard_pairs,
    _sample_lepard_pairs,
    queries,
    register_privacy_sets,
    register_sets,
    split_query_ids,
)
from quail.planner.decide import _collect
from quail.planner.plan import EngineConfig, Refusal


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
    write("policies", "policy_text",
          [f"privacy policy text {i}" for i in range(8)])
    pq.write_table(pa.table({
        "id": [f"sc{i}" for i in range(len(SCENARIOS))],
        "scenario": SCENARIOS,
    }), tmp_path / "scenarios.parquet")
    return tmp_path


def test_all_queries_compile_and_plan(tmp_path):
    _standin_sets(tmp_path)
    sess = quail.Session(EngineConfig(gpus=1), tokenizer=str.split)
    register_sets(sess, tmp_path)
    register_privacy_sets(sess, tmp_path)
    qdefs = queries(sess)
    expected = {
        *(f"IMDB-{i}" for i in range(1, 11)),
        *(f"BIO-{i}" for i in range(1, 4)),
        *(f"FEV-{i}" for i in range(1, 10)),
        *(f"LEP-{i}" for i in range(1, 9)),
        "PRIV-1", "PRIV-2",
    }
    assert set(qdefs) == expected
    assert set(QUERY_ORDER) == expected - {"PRIV-1", "PRIV-2"}
    for qid, (_, build) in qdefs.items():
        query = build()
        _, filters, joins = _collect(query.logical)
        predicates = [predicate for chain in filters.values()
                      for predicate in chain]
        assert all(predicate.selectivity is not None
                   for predicate in predicates), qid
        assert all(join.selectivity is not None for join in joins), qid
        plan = query.plan()
        assert not isinstance(plan, Refusal), f"{qid} refused: {plan}"
        assert plan.order_rule == "by_cost", qid
        assert "physical:" in query.explain(), qid


def test_set_table_matches_design():
    assert SETS == {
        "reviews": 50_000,
        "reports": 5_000,
        "claims": 5_000,
        "policies": 1_000_000,
    }
    assert LEPARD_POSITIVE_PAIRS == 5_000
    assert _n_lepard_pairs(0.1) == 500
    assert len(ASPECTS) == 12
    assert len(SCENARIOS) == 100


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
        QUERY_ORDER[16:23],
        QUERY_ORDER[23:30],
    )
