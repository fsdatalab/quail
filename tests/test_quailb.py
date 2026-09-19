"""CPU checks that every QUAIL-B query builds and plans on Quail."""

import pyarrow as pa
import pyarrow.parquet as pq

import quail
from quail.bench.quailb import queries, register_tables
from quail.planner.plan import EngineConfig, Refusal
from quail_b.data import ASPECTS, SCENARIOS
from quail_b.queries import QUERY_ORDER


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
    return tmp_path


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
            *(f"BIO-{i}" for i in range(1, 4)),
            *(f"FEV-{i}" for i in range(1, 11)),
            *(f"LEP-{i}" for i in range(1, 6)),
            "AGENT-1", "AGENT-2",
            "PRIV-1", "PRIV-2",
        }
        assert set(qdefs) == expected
        assert set(QUERY_ORDER) == expected - {"PRIV-1", "PRIV-2"}
        for qid, (_, build) in qdefs.items():
            query = build()
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
