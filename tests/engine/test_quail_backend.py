"""Quail backend execution tests without a GPU."""

import json
from types import SimpleNamespace

import pyarrow as pa
import pytest
from helpers.fakes import (
    FakeArena,
    fake_pipeline,
    fake_readout,
    fake_torch,
    run_graph_on_arena,
    two_alias_graph,
)

from quail.backends.quail import QuailModelExecution
from quail.backends.quail.distributed import execute_distributed_graph
from quail.builtins import built_in_registry
from quail.execution.result import filter_answer_table
from quail.physical import (
    AiFilter,
    PhysicalGraph,
    PortRef,
    Scan,
    input_ports,
)
from quail.specs import DEVICES, MODELS


def test_filter_execution_and_retention_inputs(monkeypatch):

    filtered = AiFilter(
        node_id="filter:d", inputs=input_ports((PortRef("input:d", "rows:d"),)),
        alias="d", question_token_ids=(9,), written_position=0, question_tokens=1,
        preamble_tokens=1, selectivity=None, expected_rows=4)
    graph = PhysicalGraph((Scan(node_id="input:d", alias="d", input_id="d"),
                           filtered), PortRef("filter:d", "rows:d"))

    def round_fn(kind, subs):
        assert kind == "execute"
        outputs = []
        for sub in subs:
            indices = sub["inputs"]["filter:d"]["input:0"].ids["d"].to_pylist()
            from quail.execution.metrics import NodeMetrics, NodeResult

            result = NodeResult({
                "rows:d": [index for index in indices if index % 2 == 0],
                "filter_answers:d": filter_answer_table(
                    "d", indices, [0] * len(indices),
                    [index % 2 == 0 for index in indices])},
                NodeMetrics(fresh_tokens=len(indices)))
            outputs.append({
                "results": {"filter:d": result}, "retained": (),
                "kv_manager": {"evicted_keys": 1},
                "memory": {"peak_gib": 1.0,
                           "memory_capacity": {"pages": 16, "slots": 2}},
            })
        return outputs

    registry = built_in_registry()
    payload = {
        "model": "qwen3-4b-fp8", "chunk_tokens": 8192, "true_ids": [1],
        "false_ids": [2], "filter_limit": None,
        "order_rule": "by_cost", "documents": {"d": [[1], [2], [3], [4]]},
        "joins": [], "filters": {}, "filter_arena_writes": {},
        "shards": {"d": ((0, 2), (1, 3))},
        "physical_plan": {
            "backend": "quail", "model": "qwen3-4b-fp8", "device": "h100-sxm",
            "gpus": 2, "settings": {}, "graph": json.loads(graph.to_json())},
    }
    result = execute_distributed_graph(
        payload, graph, 2, round_fn, DEVICES["h100-sxm"], registry.runtimes,
        registry)
    answers = result["filters"]["d"].sort_by("d")
    assert answers.column(2).to_pylist() == [True, False, True, False]
    assert result["fresh_tokens"] == 4
    assert result["kv_manager"]["evicted_keys"] == 2 and result["peak_gib"] == 1.0
    assert result["memory_capacity"] == {"pages": 32, "slots": 4}
    assert len(result["memory_workers"]) == 2

    arena = FakeArena()
    arena.owned.add(("d", 10))

    def fake_run_stages(*args, on_settled, **kwargs):
        on_settled(0, True, [True])
        return [{0: [True]}], [], 3

    monkeypatch.setattr("quail.backends.quail.executor.parts.run_stages",
                        fake_run_stages)
    execution = QuailModelExecution(SimpleNamespace(model=MODELS["qwen3-4b-fp8"]))
    execution.bind_loaded_model(
        model=object(), arena=arena, pipeline=fake_pipeline())
    execution.bind_query(torch=fake_torch(), async_answers=fake_readout(),
                         answer_rows=object(), chunk_tokens=8192)
    result = execution.execute(filtered, {
        "documents": [[1]], "document_ids": [10]})
    assert not arena.owned
    assert result.outputs["rows:d"].ids["d"].to_pylist() == [10]


def _key_columns(r_keys, p_keys):
    return {alias: pa.table({alias: pa.array(range(len(keys)), pa.int32()),
                             "key": pa.array(keys)})
            for alias, keys in (("r", r_keys), ("p", p_keys))}


def _pairs(table):
    return sorted(zip(table.column("r").to_pylist(), table.column("p").to_pylist()))


def test_pair_join_and_gpu_timing_run_through_the_quail_graph(monkeypatch):
    # document d has key d % 4; partners 0 and 3 share key 0, so a
    # document with key 0 pairs with both and one with key 3 with none
    columns = _key_columns([d % 4 for d in range(14)], [0, 1, 2, 0])
    allowed = {d: [i for i in range(4) if d % 4 == i % 3] for d in range(14)}
    for pipelined in (True, False):
        result, _, filter_truth, join_truth = run_graph_on_arena(
            monkeypatch, two_alias_graph(pipelined, hash_join=True),
            columns=columns)
        cross, _, _, _ = run_graph_on_arena(
            monkeypatch, two_alias_graph(pipelined))
        survivors = [d for d, truth in enumerate(filter_truth) if all(truth)]
        stage = result["joins"][0]
        assert sorted(stage["anchor_index"]) == survivors
        members = stage["anchor_partners"]
        for local, document in enumerate(stage["anchor_index"]):
            mine = sorted(allowed[document])
            assert list(members[local]) == mine
            assert stage["rows"].get(local, []) == [
                join_truth[("r", document)][i] for i in mine]
        table = result["_outputs"][PortRef("group:0", "join_answers:0")]
        assert _pairs(table) == sorted(
            (d, i) for d in survivors for i in allowed[d])
        assert table.column("answer").to_pylist() == [
            bool(join_truth[("r", d)][i])
            for d, i in zip(table.column("r").to_pylist(),
                            table.column("p").to_pylist())]
        metrics = result["node_metrics"]["group:0"]
        assert metrics["evaluated_document_pairs"] == sum(
            len(allowed[d]) for d in survivors)
        assert metrics["fresh_tokens"] < (
            cross["node_metrics"]["group:0"]["fresh_tokens"])
        assert result["node_metrics"]["hash_join:r-p"]["output_rows"] == sum(
            len(mine) for mine in allowed.values())
        # anchors whose pairs all answered FALSE are gone
        kept = [d for d in survivors
                if any(join_truth[("r", d)][i] for i in allowed[d])]
        root = result["_outputs"][PortRef("group:0", "rows:r")]
        assert sorted(root.column("r").to_pylist()) == kept

    # GPU timing sums the chunk events only when asked
    off, _, _, _ = run_graph_on_arena(
        monkeypatch, two_alias_graph(True), pages=48)
    assert "gpu_s" not in off
    assert all(metrics["gpu_s"] == 0.0
               for metrics in off["node_metrics"].values())
    on, model, _, _ = run_graph_on_arena(
        monkeypatch, two_alias_graph(True), pages=48, gpu_timing=True)
    # join chunks run before the filter sequence finishes: one pipeline
    kinds = [kind for kind, _ in model.launched]
    assert kinds.index("join") < len(kinds) - 1 - kinds[::-1].index("filter")
    # the fake torch reports 2 ms per event pair, one pair per chunk; a
    # chunk that packed rows of two nodes counts for each, and its time
    # is split between them by rows, so the nodes' seconds sum to the
    # pipeline's and their chunks to at least its chunk count
    per_node = {name: metrics for name, metrics in on["node_metrics"].items()
                if metrics["chunks"]}
    assert on["gpu_s"] == pytest.approx(
        sum(metrics["gpu_s"] for metrics in per_node.values()), abs=1e-3)
    assert on["gpu_s"] > 0
    assert len(model.launched) == round(on["gpu_s"] / 0.002)
    assert on["chunks"] >= len(model.launched)
    assert all(0 < metrics["gpu_s"] <= metrics["chunks"] * 0.002 + 1e-9
               for metrics in per_node.values())
