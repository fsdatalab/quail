"""Quail backend execution tests without a GPU."""

from types import SimpleNamespace

import pyarrow as pa
import pytest
from fakes import (
    expected_filter_rows,
    fake_torch,
    keep_even,
    run_graph_on_arena,
    same_key,
    two_alias_graph,
)

from quail.backends.quail import QuailModelExecution
from quail.backends.quail.distributed import execute_distributed_graph
from quail.backends.quail.graph import execute_single_graph
from quail.builtins import built_in_registry
from quail.physical import (
    AiFilter,
    AiJoin,
    FilterStage,
    JoinStage,
    PhysicalGraph,
    PortRef,
    Scan,
)
from quail.physical.base import input_ports
from quail.runtime.runner import NodeMetrics, NodeResult
from quail.specs import DEVICES, MODELS


class FakeAccounting:
    def __init__(self):
        self.owned = set()
        self.retained = set()
        self.n_pages = 100
        self.page_tokens = 16
        self.retention_cap_pages = None

    def configure_retention(self, policy, cap_pages):
        self.retention_policy = policy
        self.retention_cap_pages = cap_pages

    def pages_needed(self, tokens):
        return -(-tokens // self.page_tokens)

    @property
    def retained_pages(self):
        return len(self.retained)

    @property
    def retained_prefix_tokens(self):
        return 16 * len(self.retained)


class FakeArena:
    def __init__(self):
        self.accounting = FakeAccounting()
        self.evicted_keys = 0
        self.evicted_pages = 0
        self.evicted_prefix_tokens = 0

    def free_key(self, key):
        self.accounting.owned.discard(key)
        self.accounting.retained.discard(key)

    def retain(self, key, prefix_tokens, priority=None):
        self.accounting.owned.add(key)
        self.accounting.retained.add(key)

    def evict_retained(self, pages):
        return ()

    def reset_stats(self):
        pass


def graph_state(model_execution, docs):
    registry = built_in_registry()
    return {
        "torch": fake_torch(),
        "arena": FakeArena(),
        "pipeline": SimpleNamespace(attention_mode=None),
        "model_execution": model_execution,
        "runtimes": registry.runtimes,
        "model_spec": MODELS["qwen3-4b-fp8"],
        "device": DEVICES["h100-sxm"],
        "chunk_tokens": 8192,
        "docs": docs,
    }


def test_fixed_join_executes_without_optimizer(monkeypatch):
    def unexpected_search(*args, **kwargs):
        raise AssertionError("execution called the join optimizer")

    monkeypatch.setattr("quail.planner.joins.search_joins", unexpected_search)
    scan_r = Scan(
        node_id="input:r", alias="r", input_id="r"
    )
    scan_p = Scan(
        node_id="input:p", alias="p", input_id="p"
    )
    join = AiJoin(
        node_id="join", anchor="r",
        inputs=input_ports((PortRef("input:r", "ids:r"),
                            PortRef("input:p", "ids:p"))),
        stages=(JoinStage(
            written_pos=0, exec_idx=0, anchor="r", partners=("p",),
            semantics="full", selectivity=0.5, expected_tuples=4,
            anchor_frame_tokens=1, pair_tail_tokens=1,
            anchor_resident="none", tuple_tokens=0,
            frame_token_ids=(7,), label_token_ids=(("p", (10,)),),
            tail_token_ids=(11,),
        ),),
    )
    graph = PhysicalGraph(
        (scan_r, scan_p, join), PortRef("join", "ids:r")
    )

    class FixedJoinExecution:
        def execute(self, node, inputs):
            assert isinstance(node, AiJoin)
            answers = [{0: [True, False], 1: [False, True]}]
            return NodeResult(
                {
                    "ids:r": [0, 1],
                    "join_answers:0": {
                        "rows": answers[0],
                        "anchor_index": [0, 1],
                        "partner_index": [[0], [1]],
                        "anchor": "r",
                        "partners": ["p"],
                        "semantics": "full",
                        "selectivity": 0.5,
                        "written_pos": 0,
                    },
                },
                NodeMetrics(
                    fresh_tokens=20,
                    extension={"answers": answers},
                ),
            )

    result = execute_single_graph(
        graph_state(
            FixedJoinExecution(),
            {"r": [[1], [2]], "p": [[3], [4]]},
        ),
        {"joins": [{}], "filter_limit": None, "pre_ids": [],
         "order_rule": "by_cost"},
        graph,
    )

    assert result["fresh_tokens"] == 20
    assert result["joins"][0]["written_pos"] == 0
    assert result["executed_join_plan"][0]["type"] == \
        AiJoin.type_name


def distributed_payload(docs, joins=None):
    return {
        "model": "qwen3-4b-fp8",
        "chunk_tokens": 8192,
        "true_ids": [1],
        "false_ids": [2],
        "pre_ids": [],
        "filter_limit": None,
        "order_rule": "by_cost",
        "docs": docs,
        "joins": joins or [],
        "filters": {},
        "filter_arena_writes": {},
        "shards": {
            alias: tuple(
                tuple(range(worker, len(rows), 2))
                for worker in range(2)
            )
            for alias, rows in docs.items()
        },
    }


def attach_plan(payload, graph):
    from quail.physical import plan_envelope

    registry = built_in_registry()
    payload["physical_plan"] = plan_envelope(
        backend="quail",
        model="qwen3-4b-fp8",
        device="h100-sxm",
        workers=2,
        graph=graph,
        codecs=registry.codecs,
    )
    return payload


def test_filter_execution_and_retention_inputs(monkeypatch):
    scan = Scan(
        node_id="input:d", alias="d", input_id="d"
    )
    filtered = AiFilter(
        node_id="filter:d",
        inputs=input_ports((PortRef("input:d", "ids:d"),)),
        alias="d",
        stages=(FilterStage(0, 1, 1, None, 4),),
        question_token_ids=((9,),),
    )
    graph = PhysicalGraph(
        (scan, filtered), PortRef("filter:d", "ids:d")
    )

    def round_fn(kind, subs):
        assert kind == "filters"
        outputs = []
        for sub in subs:
            indices = sub["doc_index"]["d"]
            answers = {index: [index % 2 == 0] for index in indices}
            outputs.append({
                "filters": {"d": answers},
                "survivors": {
                    "d": [index for index in indices if index % 2 == 0]
                },
                "retained": {},
                "fresh_tokens": len(indices),
                "boot_s": 0.0,
                "boot_kind": "warm",
                "boot": {},
                "peak_gib": 1.0,
            })
        return outputs

    registry = built_in_registry()
    result = execute_distributed_graph(
        attach_plan(
            distributed_payload({"d": [[1], [2], [3], [4]]}),
            graph,
        ),
        graph,
        2,
        round_fn,
        MODELS["qwen3-4b-fp8"],
        DEVICES["h100-sxm"],
        registry.runtimes,
        registry,
    )

    assert result["filters"]["d"] == {
        0: [True], 1: [False], 2: [True], 3: [False]
    }
    assert result["fresh_tokens"] == 4

    with monkeypatch.context() as patch:
        received = {}

        def fake_run_filter(*args, retain_survivors, **kwargs):
            received["retain_survivors"] = retain_survivors
            return {0: [True]}, [], 3

        patch.setattr("quail.executor.loop.run_filter", fake_run_filter)
        execution = QuailModelExecution(SimpleNamespace())
        execution.bind_loaded_model(
            model=object(), arena=FakeArena(), pipeline=SimpleNamespace()
        )
        execution.bind_query(
            torch=fake_torch(), async_answers=object(), chunk_tokens=8192
        )
        node = AiFilter(
            node_id="filter:d",
            alias="d",
            stages=(FilterStage(0, 1, 1, None, 4),),
            question_token_ids=((9,),),
        )

        result = execution.execute(
            node,
            {
                "documents": [[1]],
                "document_ids": [10],
                "retain_survivors": False,
            },
        )

        assert received["retain_survivors"] == ()
        assert result.outputs["ids:d"] == [10]


# ------------------------------------------------ streamed edges on a page arena


def test_streamed_edge_runs_through_the_quail_graph(monkeypatch):
    """A filter and a join with a streamed edge, through the real graph runtime."""
    result, model, filter_truth, join_truth = run_graph_on_arena(
        monkeypatch, two_alias_graph(True, stages=2), n_partners=3, seed=3,
        pages=48, stages=2, partner_tokens=35)
    assert result["filters"]["r"] == expected_filter_rows(filter_truth)
    survivors = [d for d, truth in enumerate(filter_truth) if all(truth)]
    rows = result["joins"][0]
    assert sorted(rows["anchor_index"]) == survivors
    for local, document in enumerate(rows["anchor_index"]):
        assert rows["rows"][local] == join_truth[("r", document)]
    metrics = result["node_metrics"]
    assert metrics["filter:r"]["evaluated_documents"] == 14
    assert metrics["filter:r"]["fresh_tokens"] > 0
    assert metrics["group:0"]["kv_hits"] == len(survivors)
    assert metrics["group:0"]["kv_misses"] == 0
    assert result["fresh_tokens"] == (
        metrics["filter:r"]["fresh_tokens"] + metrics["group:0"]["fresh_tokens"])
    assert result["kv_manager"]["join_anchor_hits"] == len(survivors)
    kinds = [kind for kind, _ in model.launched]
    assert kinds.index("join") < len(kinds) - 1 - kinds[::-1].index("filter")


def _key_columns(r_keys, p_keys):
    return {
        "r": pa.table({"r": pa.array(range(len(r_keys)), pa.int32()),
                       "key": pa.array(r_keys)}),
        "p": pa.table({"p": pa.array(range(len(p_keys)), pa.int32()),
                       "key": pa.array(p_keys)}),
    }


def test_pair_join_runs_through_the_quail_graph(monkeypatch):
    # document d has key d % 4; partners 0 and 3 share key 0, so a
    # document with key 0 pairs with both and one with key 3 with none
    columns = _key_columns([d % 4 for d in range(14)], [0, 1, 2, 0])
    allowed = {d: [i for i in range(4) if d % 4 == i % 3] for d in range(14)}
    for pin_survivors in (True, False):
        result, _, filter_truth, join_truth = run_graph_on_arena(
            monkeypatch, two_alias_graph(pin_survivors, hash_join=True),
            columns=columns)
        cross, _, _, _ = run_graph_on_arena(
            monkeypatch, two_alias_graph(pin_survivors))
        survivors = [d for d, truth in enumerate(filter_truth) if all(truth)]
        stage = result["joins"][0]
        assert sorted(stage["anchor_index"]) == survivors
        members = stage["anchor_partners"]
        for local, document in enumerate(stage["anchor_index"]):
            mine = sorted(allowed[document])
            assert members[local] == mine
            # an anchor with no pair settles without a row
            assert stage["rows"].get(local, []) == [
                join_truth[("r", document)][i] for i in mine]
        # the exported answer table holds exactly the evaluated pairs
        table = result["_outputs"][PortRef("group:0", "join_answers:0")]
        assert sorted(zip(table.column("r").to_pylist(),
                          table.column("p").to_pylist())) == sorted(
            (d, i) for d in survivors for i in allowed[d])
        assert table.column("answer").to_pylist() == [
            bool(join_truth[("r", d)][i])
            for d, i in zip(table.column("r").to_pylist(),
                            table.column("p").to_pylist())]
        # fewer pairs, fewer fresh tokens than the cross join
        metrics = result["node_metrics"]["group:0"]
        assert metrics["evaluated_document_pairs"] == sum(
            len(allowed[d]) for d in survivors)
        assert metrics["fresh_tokens"] < (
            cross["node_metrics"]["group:0"]["fresh_tokens"])
        assert result["node_metrics"]["hash_join:r-p"]["output_rows"] == sum(
            len(mine) for mine in allowed.values())
        # anchors whose pairs all answered FALSE are gone; the rest
        # survive with the same rule as a cross join
        kept = [d for d in survivors
                if any(join_truth[("r", d)][i] for i in allowed[d])]
        root = result["_outputs"][PortRef("group:0", "ids:r")]
        assert sorted(root.column("r").to_pylist()) == kept


def _foreign_run(monkeypatch, graph, functions):
    # document d has key d % 4; partner i has key i
    result, _, filter_truth, join_truth = run_graph_on_arena(
        monkeypatch, graph, seed=9, functions=functions,
        columns=_key_columns([d % 4 for d in range(14)], list(range(4))))
    survivors = [d for d, truth in enumerate(filter_truth) if all(truth)]
    return result, survivors, join_truth


def test_foreign_runs_per_batch_on_the_stream_and_once_as_a_barrier(monkeypatch):
    functions = {"keep_even": keep_even, "same_key": same_key}
    # a per-batch drop on the pinned chain: only even survivors reach
    # the join, the filter still reports every survivor, and the
    # function ran once per chunk of survivors
    result, survivors, join_truth = _foreign_run(
        monkeypatch, two_alias_graph(True, foreign=("per_batch", "drop")),
        functions)
    assert sorted(result["filters"]["r"]) == list(range(14))
    stage = result["joins"][0]
    kept = [d for d in survivors if d % 2 == 0]
    assert sorted(stage["anchor_index"]) == kept
    foreign = result["node_metrics"]["apply:keep_even"]
    assert foreign["input_rows"] == len(survivors)
    assert foreign["output_rows"] == len(kept)
    assert result["_outputs"][PortRef("apply:keep_even", "ids:r")].column(
        "r").to_pylist() == kept
    assert result["node_metrics"]["group:0"]["kv_hits"] == len(kept)

    # the same function as a barrier over a materialized chain
    result, survivors, _ = _foreign_run(
        monkeypatch, two_alias_graph(False, foreign=("barrier", "drop")),
        functions)
    assert sorted(result["joins"][0]["anchor_index"]) == kept

    # pairs from a per-batch function equal pairs from a barrier one,
    # and both equal the key equality: document d pairs with partner
    # d % 4 only
    per_batch, survivors, join_truth = _foreign_run(
        monkeypatch, two_alias_graph(True, foreign=("per_batch", "pairs")),
        functions)
    barrier, _, _ = _foreign_run(
        monkeypatch, two_alias_graph(False, foreign=("barrier", "pairs")),
        functions)
    for result in (per_batch, barrier):
        table = result["_outputs"][PortRef("group:0", "join_answers:0")]
        assert sorted(zip(table.column("r").to_pylist(),
                          table.column("p").to_pylist())) == [
            (d, d % 4) for d in survivors]
        assert {(r, p): a for r, p, a in zip(
            table.column("r").to_pylist(), table.column("p").to_pylist(),
            table.column("answer").to_pylist())} == {
            (d, d % 4): bool(join_truth[("r", d)][d % 4]) for d in survivors}
        pairs = result["_outputs"][PortRef("apply:same_key", "pairs:0")]
        assert sorted(zip(pairs.column("r").to_pylist(),
                          pairs.column("p").to_pylist())) == [
            (d, d % 4) for d in survivors]
        assert result["node_metrics"]["apply:same_key"]["output_rows"] == len(
            survivors)

    # a function never invents an id, and preserve means every id
    def invent(tables):
        return [99]

    def lose_one(tables):
        (alias,) = tables
        return tables[alias].column(alias).to_pylist()[1:]

    def outer(tables):
        (alias,) = tables
        return [None] + tables[alias].column(alias).to_pylist()[1:]

    with pytest.raises(ValueError, match="never invents an id"):
        _foreign_run(monkeypatch,
                     two_alias_graph(True, foreign=("per_batch", "drop")),
                     {"keep_even": invent})
    with pytest.raises(ValueError, match="preserves ids but dropped"):
        _foreign_run(monkeypatch,
                     two_alias_graph(False, foreign=("barrier", "preserve")),
                     {"keep_even": lose_one})
    with pytest.raises(ValueError, match="returned a null id"):
        _foreign_run(monkeypatch,
                     two_alias_graph(False, foreign=("barrier", "drop")),
                     {"keep_even": outer})


def test_gpu_timing_sums_the_chunk_events_only_when_asked(monkeypatch):
    """gpu_s is the sum of every chunk's CUDA event pair, opt in."""
    off, _, _, _ = run_graph_on_arena(
        monkeypatch, two_alias_graph(True), pages=48)
    assert "gpu_s" not in off
    assert all(metrics["gpu_s"] == 0.0
               for metrics in off["node_metrics"].values())
    on, _, _, _ = run_graph_on_arena(
        monkeypatch, two_alias_graph(True), pages=48, gpu_timing=True)
    # the fake torch reports 2 ms per event pair, one pair per chunk
    per_node = [metrics["gpu_s"] for metrics in on["node_metrics"].values()]
    assert on["gpu_s"] == pytest.approx(sum(per_node), abs=1e-3)
    assert on["gpu_s"] > 0
    assert all(chunks == round(chunks)
               for chunks in (seconds / 0.002 for seconds in per_node))
