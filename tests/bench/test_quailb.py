"""CPU checks that every QUAIL-B query builds and plans on Quail."""


from helpers.planning import operators_of
from helpers.quailb_data import standin_sets
from helpers.sessions import TOKENIZER

import quail
from bench.quailb import (
    _submission_to_answer_s,
    queries,
    register_tables,
)
from quail.api.config import EngineConfig
from quail.physical.plan import Refusal
from quail_b.queries import QUERY_ORDER


def test_all_queries_compile_and_plan_and_answer_timing_adds_common_work(tmp_path):
    standin_sets(tmp_path)
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
    for backend in ("quail", "stock_vllm", "pipelined_vllm"):
        with quail.Session(
            EngineConfig(gpus=1, model="qwen3-4b-fp8", backend=backend,
                         device="h100-sxm"),
            tokenizer=TOKENIZER,
        ) as sess:
            register_tables(sess, tmp_path)
            qdefs = queries(sess)
            # every backend lists every query
            assert set(qdefs) == expected | classify | extract | relational, (
                backend)
            for qid, (_, build) in qdefs.items():
                case = f"{backend} {qid}"
                query = build()
                found = operators_of(query)
                selectivities = [
                    predicate.selectivity
                    for _, predicates in found.filters for predicate in predicates]
                selectivities += [join.selectivity for join in found.joins]
                assert all((s is None) == qid.startswith("PRIV-")
                           for s in selectivities), case
                plan = query.plan()
                sorts = bool(found.result.order or found.result.distinct
                             or found.result.offset)
                assert sorts <= (qid in relational), case
                if backend != "quail" and sorts:
                    # a request backend sorts returned rows on the coordinator
                    if isinstance(plan, Refusal):
                        # refused for its AI functions, not its sort
                        assert plan.constraint != "sort_needs_quail_backend", case
                    else:
                        assert any(node.type_name == "quail.sort"
                                   for node in plan.graph.nodes), case
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
    # DataFusion's time around the model region counts for every backend
    report["datafusion_s"] = 0.5
    assert _submission_to_answer_s(
        "quail", report, frontend_s=1.0, answer_prepare_s=0.75
    ) == 15.0
