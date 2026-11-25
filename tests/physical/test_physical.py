"""Physical planning and runtime extension test."""

import copy
import json
import sys
import types

import pyarrow as pa
import pytest

import quail
from quail.api.execute import _execute_physical
from quail.builtins import built_in_registry
from quail.common.expressions import ColumnRef
from quail.execution.metrics import NodeResult
from quail.execution.runtimes import BarrierRuntime
from quail.execution.types import PhysicalRequest, PhysicalResponse, document_input
from quail.physical import (
    Barrier,
    ExtensionNode,
    InputPort,
    OutputPort,
    PhysicalGraph,
    PortRef,
    Scan,
    ValueType,
    decode_plan_value,
)
from quail.physical.plan import PhysicalPlan, graph_from_envelope


@pytest.mark.parametrize("shape", ["cycle"])
def test_barrier_survivors_are_the_ids_of_a_complete_join(shape):
    """A survivor needs a partner in every join at once, whatever the relation order."""
    import itertools

    from quail.physical import input_ports

    if shape == "disconnected":
        edges = [("a", "b", [(1, 0)]), ("c", "d", [(0, 1)])]
        expected = {"a": [1], "b": [0], "c": [0], "d": [1]}
    else:
        edges = [("a", "b", [(0, 0), (1, 1)]),
                 ("b", "c", [(1, 0)] if shape == "chain" else [(0, 0), (1, 1)])]
        if shape == "cycle":
            # every id has a partner in each join, but no triangle closes
            edges.append(("a", "c", [(0, 1), (1, 0)]))
        expected = ({"a": [1], "b": [1], "c": [0], "d": [0, 1]} if shape == "chain"
                    else {"a": [], "b": [], "c": [], "d": [0, 1]})
    for ordered in itertools.permutations(edges):
        ports = [PortRef(f"scan:{alias}", f"rows:{alias}") for alias in expected]
        values = [pa.array([0, 1], pa.int32()) for _ in ports]
        for index, (left, right, pairs) in enumerate(ordered):
            ports.append(PortRef(f"join:{index}", f"join_answers:{index}"))
            values.append(pa.table({
                left: pa.array([pair[0] for pair in pairs], pa.int32()),
                right: pa.array([pair[1] for pair in pairs], pa.int32()),
                "answer": [True] * len(pairs)}))
        node = Barrier(node_id="barrier", aliases=tuple(expected),
                       inputs=input_ports(tuple(ports)))
        outputs = BarrierRuntime().execute(
            node, {f"input:{i}": value for i, value in enumerate(values)}, None).outputs
        assert {alias: outputs[f"rows:{alias}"].ids[alias].to_pylist()
                for alias in expected} == expected, ordered


@pytest.mark.parametrize("aliases", [("left", "right")])
def test_hash_join_keeps_ids_apart_from_a_key_column_named_like_its_alias(aliases):
    from types import SimpleNamespace

    from quail.common.pairs import columns_key
    from quail.execution.runtimes.dataflow import HashJoinRuntime
    from quail.physical import HashJoin, input_ports

    left, right = aliases
    node = HashJoin(
        node_id="join", left=left, right=right, written_position=0,
        inputs=input_ports((PortRef("l", f"rows:{left}"),
                            PortRef("r", f"rows:{right}"))),
        on=((ColumnRef(left, left, left), ColumnRef(right, right, right)),))
    context = SimpleNamespace(sources={
        columns_key(left): pa.table({left: [30, 10, 20]}),
        columns_key(right): pa.table({right: [20, 40, 10]})})
    result = HashJoinRuntime().execute(
        node, {"input:0": pa.array([1, 2]), "input:1": pa.array([0, 2])}, context)
    assert result.outputs["pairs:0"].to_pydict() == {left: [1, 2], right: [2, 0]}


FIRST_DOCUMENTS = "test.first_documents"


class FirstDocumentsRuntime:
    def execute(self, node, inputs, context):
        documents = next(iter(inputs.values()))
        return NodeResult({f"rows:{node.alias}": documents[:node.count]})


def local_plan(*, num_documents, count):
    def ids_from(node_id):
        return (InputPort("input:0", ValueType.ROWS,
                          PortRef(node_id, "rows:d"), schema=("d",)),)

    scan = Scan(node_id="input:d", alias="d", input_id="d",
                num_documents=num_documents)
    filtered = ExtensionNode(
        FIRST_DOCUMENTS, "filter:d", inputs=ids_from("input:d"),
        outputs=(OutputPort("rows:d", ValueType.ROWS, schema=("d",)),),
        alias="d", count=count)
    return PhysicalPlan(model="qwen3-4b-fp8", device="h100-sxm", gpus=1,
                        nodes=(scan, filtered))


def test_an_extension_node_encodes_validates_and_runs(monkeypatch):
    registry = quail.ExtensionRegistry.with_built_ins()
    registry.register_runtime(FirstDocumentsRuntime(), key=FIRST_DOCUMENTS)
    plan = local_plan(num_documents=2, count=1)
    envelope = plan.to_envelope()
    assert graph_from_envelope(envelope) == plan.graph
    plan.graph.validate(runtime_keys=set(registry.runtimes))
    result = FirstDocumentsRuntime().execute(
        plan.nodes[1], {"input:0": ["a", "b"]}, None)
    assert result.outputs == {"rows:d": ["a"]}
    assert "test.first_documents" in plan.graph.to_json()

    scan = Scan(node_id="input:d", alias="d", input_id="d", num_documents=1,
                total_tokens=1)
    encoded = json.loads(PhysicalGraph((scan,), PortRef("input:d", "rows:d")).to_json())
    missing = copy.deepcopy(encoded)
    del missing["nodes"][0]["attributes"]["total_tokens"]
    with pytest.raises(ValueError, match="missing field"):
        PhysicalGraph.from_json(json.dumps(missing))

    extra = copy.deepcopy(encoded)
    extra["nodes"][0]["attributes"]["unused"] = True
    with pytest.raises(ValueError, match="unknown field"):
        PhysicalGraph.from_json(json.dumps(extra))

    class ExampleBackend:
        name = "test.example"

        def execute_request(self, context):
            nodes = [node.node_id for node in context.graph.nodes]
            return PhysicalResponse({}, {"backend": self.name, "nodes": nodes})

    def register(registry):
        registry.register_backend(ExampleBackend())

    module = types.ModuleType("test_quail_extension")
    module.register_quail_extension = register
    monkeypatch.setitem(sys.modules, module.__name__, module)
    registry = built_in_registry()
    scan = Scan(node_id="input:d", alias="d", input_id="d", num_documents=2)
    plan = PhysicalPlan(model="qwen3-4b-fp8", device="h100-sxm", gpus=1,
                        backend=ExampleBackend.name, nodes=(scan,))
    request = PhysicalRequest(
        plan.to_envelope(), {"d": document_input(pa.array([[1], [2]]))})
    registry.load_extension(module)
    response = _execute_physical(request, registry)
    assert response.metrics == {"backend": "test.example", "nodes": ["input:d"]}


def test_a_plan_saved_before_the_expressions_moved_still_decodes():
    column = ColumnRef("r", "reviews", "stars")
    saved = json.loads(column.to_json())
    saved["type"] = saved["type"].replace(
        "quail.common.expressions", "quail.logical.expressions")
    assert decode_plan_value(json.dumps(saved)) == column
