"""CPU checks for the QUAIL-B query catalog and table schemas."""

import pyarrow as pa
import pyarrow.parquet as pq

import quail
from quail.bench.quailb import (
    ASPECTS, SCENARIOS, SETS, queries, register_privacy_sets, register_sets,
)
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
        *(f"BIO-{i}" for i in range(1, 9)),
        *(f"FEV-{i}" for i in range(1, 10)),
        *(f"LEP-{i}" for i in range(1, 9)),
        "PRIV-1", "PRIV-2",
    }
    assert set(qdefs) == expected
    for qid, (_, build) in qdefs.items():
        query = build()
        plan = query.plan()
        assert not isinstance(plan, Refusal), f"{qid} refused: {plan}"
        assert "physical:" in query.explain(), qid


def test_set_table_matches_design():
    assert SETS == {
        "reviews": 50_000,
        "reports": 10_000,
        "claims": 100_000,
        "citations": 2_000,
        "policies": 1_000_000,
    }
    assert len(ASPECTS) == 12
    assert len(SCENARIOS) == 100
