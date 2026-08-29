"""CPU checks for the QUAIL-B query catalog and table schemas."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quail
from quail.bench.quailb import (
    ASPECTS,
    QUERY_FAMILY_WORKLOADS,
    QUERY_ORDER,
    SCENARIOS,
    SETS,
    queries,
    query_family_name,
    register_privacy_sets,
    register_sets,
    split_query_ids,
    split_query_families,
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
        "id": [f"lp{i}" for i in range(6)],
        "destination_context": [f"citation excerpt {i}" for i in range(6)],
        "passage_text": [f"cited passage {i}" for i in range(6)],
        "passage_id": [f"passage-{i}" for i in range(6)],
    }), tmp_path / "citations.parquet")
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
        expected_order = "as_written" if qid.startswith("PRIV-") else "by_cost"
        assert plan.order_rule == expected_order, qid
        assert "physical:" in query.explain(), qid


def test_set_table_matches_design():
    assert SETS == {
        "reviews": 50_000,
        "reports": 5_000,
        "claims": 5_000,
        "citations": 2_000,
        "policies": 1_000_000,
    }
    assert len(ASPECTS) == 12
    assert len(SCENARIOS) == 100


def test_parallel_query_split_matches_stock_vllm():
    assert split_query_ids(QUERY_ORDER, 4) == (
        QUERY_ORDER[0:8],
        QUERY_ORDER[8:16],
        QUERY_ORDER[16:23],
        QUERY_ORDER[23:30],
    )


def test_query_family_split_matches_benchmark_catalog():
    assert QUERY_FAMILY_WORKLOADS == {
        "IMDB": "imdb",
        "BIO": "biodex",
        "FEV": "fever",
        "LEP": "lepard",
    }
    assert split_query_families(QUERY_ORDER) == (
        QUERY_ORDER[0:10],
        QUERY_ORDER[10:13],
        QUERY_ORDER[13:22],
        QUERY_ORDER[22:30],
    )
    assert query_family_name(QUERY_ORDER[0:10]) == "imdb"


def test_query_family_rejects_mixed_or_unknown_queries():
    with pytest.raises(ValueError, match="expected one query family"):
        query_family_name(("IMDB-1", "BIO-1"))
    with pytest.raises(ValueError, match="unknown query family"):
        split_query_families(("OTHER-1",))
