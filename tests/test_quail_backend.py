"""Quail backend execution tests without a GPU."""

from contextlib import nullcontext
from types import SimpleNamespace

from quail.backends.quail import QuailModelExecution
from quail.builtins import built_in_registry
from quail.physical import (
    AdaptiveJoinPlan,
    AnchoredJoin,
    DocumentInput,
    FilterStage,
    PackedFilter,
    PhysicalGraph,
    PortRef,
)
from quail.physical.base import input_ports
from quail.backends.quail.graph import execute_single_graph
from quail.backends.quail.distributed import execute_distributed_graph
from quail.runtime.runner import NodeMetrics, NodeResult
from quail.specs import DEVICES, MODELS


class FakeAccounting:
    def __init__(self):
        self.owned = set()
        self.retained = set()
        self.n_pages = 100
        self.page_tokens = 16

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

    def retain(self, key, prefix_tokens):
        self.accounting.owned.add(key)
        self.accounting.retained.add(key)

    def reset_stats(self):
        pass


def fake_torch():
    return SimpleNamespace(
        inference_mode=nullcontext,
        cuda=SimpleNamespace(
            synchronize=lambda: None,
            max_memory_allocated=lambda: 0,
        ),
    )


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


def test_adaptive_join_executes_typed_child_graph(monkeypatch):
    monkeypatch.setattr(
        "quail.planner.joins.search_joins",
        lambda *args, **kwargs: {
            "seq": [(0, "r")], "states": 1, "generated": 1
        },
    )
    scan_r = DocumentInput(
        node_id="input:r", alias="r", input_id="r"
    )
    scan_p = DocumentInput(
        node_id="input:p", alias="p", input_id="p"
    )
    adaptive = AdaptiveJoinPlan(
        node_id="adaptive",
        inputs=input_ports((PortRef("input:r", "ids:r"),
                            PortRef("input:p", "ids:p"))),
        aliases=("r", "p"),
        join_positions=(0,),
        full_join_positions=(0,),
        join_specs=({
            "written_pos": 0,
            "aliases": ["r", "p"],
            "anchor": "r",
            "partners": ["p"],
            "anchor_free": True,
            "semantics": "full",
            "selectivity": 0.5,
            "frames": {"r": [7], "p": [8]},
            "labels": {"r": [9], "p": [10]},
            "tail": [11],
        },),
    )
    graph = PhysicalGraph(
        (scan_r, scan_p, adaptive), PortRef("adaptive", "ids:r")
    )

    class FixedJoinExecution:
        def execute(self, node, inputs):
            assert isinstance(node, AnchoredJoin)
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
    assert result["join_optimizer"]["executed_plan"][0]["type"] == \
        AnchoredJoin.type_name


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


def test_distributed_filter_executes_typed_node():
    scan = DocumentInput(
        node_id="input:d", alias="d", input_id="d"
    )
    filtered = PackedFilter(
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


def test_filter_without_retention_passes_an_empty_selection(monkeypatch):
    received = {}

    def fake_run_filter(*args, retain_survivors, **kwargs):
        received["retain_survivors"] = retain_survivors
        return {0: [True]}, [], 3

    monkeypatch.setattr("quail.executor.loop.run_filter", fake_run_filter)
    execution = QuailModelExecution(SimpleNamespace())
    execution.bind_loaded_model(
        model=object(), arena=FakeArena(), pipeline=SimpleNamespace()
    )
    execution.bind_query(
        torch=fake_torch(), async_answers=object(), chunk_tokens=8192
    )
    node = PackedFilter(
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
