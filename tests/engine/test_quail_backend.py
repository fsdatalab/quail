"""Quail backend execution tests without a GPU."""

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
from quail.physical import (
    AiFilter,
    PortRef,
    input_ports,
)
from quail.specs import MODELS


def test_filter_execution_and_retention_inputs(monkeypatch):

    filtered = AiFilter(
        node_id="filter:d", inputs=input_ports((PortRef("input:d", "rows:d"),)),
        alias="d", question_token_ids=(9,), written_position=0, question_tokens=1,
        preamble_tokens=1, selectivity=None, expected_rows=4)
    arena = FakeArena()
    arena.owned.add(("d", 10))

    class SettlingExecutor:
        def __init__(self, *args, on_settled, **kwargs):
            self.on_settled = on_settled

        def run(self):
            self.on_settled(0, True, [True])
            return [{0: [True]}], [], 3

    monkeypatch.setattr(
        "quail.backends.quail.executor.stage_executor.StageExecutor",
        SettlingExecutor)
    execution = QuailModelExecution(SimpleNamespace(model=MODELS["qwen3-4b-fp8"]))
    execution.bind_loaded_model(
        model=object(), arena=arena, pipeline=fake_pipeline())
    execution.bind_query(torch=fake_torch(), async_answers=fake_readout(),
                         answer_rows=object(), chunk_tokens=8192)
    context = SimpleNamespace(sources={}, state={
        "documents": {"d": [[1]] * 11}, "filter_limit": None})
    result = execution.execute_pipeline(
        (filtered,), {"filter:d": {"input:0": [10]}}, context)["filter:d"]
    assert not arena.owned
    assert result.outputs["rows:d"].ids["d"].to_pylist() == [10]


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


def _key_columns(r_keys, p_keys):
    return {alias: pa.table({alias: pa.array(range(len(keys)), pa.int32()),
                             "key": pa.array(keys)})
            for alias, keys in (("r", r_keys), ("p", p_keys))}


def _pairs(table):
    return sorted(zip(table.column("r").to_pylist(), table.column("p").to_pylist()))
