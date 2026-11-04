"""Physical planning and runtime extension test."""

import copy
import sys
import types
from dataclasses import dataclass
from typing import ClassVar

import pyarrow as pa
import pytest

import quail
from quail.api.config import EngineConfig
from quail.api.execute import _execute_physical
from quail.builtins import built_in_registry
from quail.catalog import DocumentProvider
from quail.execution.metrics import NodeResult
from quail.execution.runtimes import BarrierRuntime
from quail.execution.types import PhysicalRequest, PhysicalResponse, document_input
from quail.logical import ColumnRef
from quail.physical import (
    Barrier,
    ExecutionLocation,
    InputPort,
    OutputPort,
    PhysicalGraph,
    PhysicalNode,
    PortRef,
    Project,
    Scan,
    ValueType,
    decode_graph,
    encode_graph,
    plan_envelope,
)
from quail.physical.plan import PhysicalPlan
from quail.planner.candidates import PhysicalCandidate
from quail.specs import SupportResult


@pytest.mark.parametrize("shape", ["cycle"])
def test_barrier_survivors_are_the_ids_of_a_complete_join(shape):
    """A survivor needs a partner in every join at once, whatever the relation order."""
    import itertools

    from quail.physical.base import input_ports

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
    from quail.physical import HashJoin
    from quail.physical.base import input_ports

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


@dataclass(frozen=True)
class FirstDocuments(PhysicalNode):
    alias: str = ""
    count: int = 1

    type_name: ClassVar[str] = "test.first_documents"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self):
        return (OutputPort(f"rows:{self.alias}", ValueType.ROWS,
                           schema=(self.alias,)),)

class FirstDocumentsRuntime:
    def execute(self, node, inputs, context):
        documents = next(iter(inputs.values()))
        return NodeResult({f"rows:{node.alias}": documents[:node.count]})


def local_plan(context, *, count, estimate, source):
    def ids_from(node_id):
        return (InputPort("input:0", ValueType.ROWS,
                          PortRef(node_id, "rows:d"), schema=("d",)),)

    scan = Scan(node_id="input:d", alias="d", input_id="d",
                num_documents=len(context.document_tokens["d"]))
    filtered = FirstDocuments(
        node_id="filter:d", inputs=ids_from("input:d"), alias="d", count=count)
    project = Project(
        node_id="project", inputs=ids_from("filter:d"), columns=("d.id",))
    plan = PhysicalPlan(
        model=context.model.name, device=context.device.name,
        gpus=context.gpus, backend="local_filter",
        estimated_seconds=estimate, nodes=(scan, filtered, project),
        settings={"planner_source": source})
    return PhysicalCandidate(plan.graph, plan, estimate)


class LocalFilterBackend:
    name = "local_filter"

    def __init__(self):
        self.called = False

    def supports(self, model, device, gpus):
        return SupportResult.accept()

    def plan(self, region, context):
        self.called = True
        return (local_plan(context, count=1, estimate=0.0, source="backend"),)


def test_physical_extensions_plan_validate_and_execute(monkeypatch):
    registry = quail.ExtensionRegistry.with_built_ins()
    backend = LocalFilterBackend()
    registry.register_backend(backend)
    registry.register_node(FirstDocuments, runtime=FirstDocumentsRuntime())
    session = quail.Session(
        EngineConfig(gpus=1, model="qwen3-4b-fp8", backend="local_filter",
                     device="h100-sxm"),
        tokenizer=str.split,
        registry=registry,
    )
    session.register("documents", DocumentProvider.from_table(
        pa.table({"id": ["a", "b"], "body": ["one", "two"]}),
        id_col="id", identity="local-filter-docs"))
    query = session.sql(
        "SELECT d.id FROM documents d WHERE AI_FILTER(PROMPT('ok {0}', d.body))")
    plan = query.plan()
    envelope = plan.to_envelope(registry.codecs)

    assert backend.called
    assert plan.settings["planner_source"] == "backend"
    firsts = [node for node in plan.nodes if isinstance(node, FirstDocuments)]
    assert firsts and all(node.count == 1 for node in firsts)
    assert decode_graph(envelope["graph"], registry.codecs) == plan.graph

    registry = built_in_registry()
    scan = Scan(node_id="input:d", alias="d", input_id="d", num_documents=1,
                shard_ranges=((0, 1),), shard_token_loads=(1,))
    plan = PhysicalGraph((scan,), PortRef("input:d", "rows:d"))
    encoded = encode_graph(plan, registry.codecs)
    missing = copy.deepcopy(encoded)
    del missing["nodes"][0]["attributes"]["shard_ranges"]
    with pytest.raises(ValueError, match="missing fields"):
        decode_graph(missing, registry.codecs)

    extra = copy.deepcopy(encoded)
    extra["nodes"][0]["unused"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        decode_graph(extra, registry.codecs)

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
    graph = PhysicalGraph((scan,), PortRef("input:d", "rows:d"))
    request = PhysicalRequest(
        plan_envelope(backend=ExampleBackend.name, model="qwen3-4b-fp8",
                      device="h100-sxm", gpus=1, graph=graph,
                      codecs=registry.codecs),
        {"d": document_input(pa.array([[1], [2]]))},
    )
    registry.load_extension(module)
    response = _execute_physical(request, registry)
    assert response.metrics == {"backend": "test.example", "nodes": ["input:d"]}


def test_a_plan_saved_before_the_expressions_moved_still_decodes():
    from quail.common.expressions import ColumnRef, RegularPredicate
    from quail.physical.codec import decode_value, encode_value

    predicate = RegularPredicate(ColumnRef("r", "reviews", "stars"), ">", 3)
    saved = encode_value(predicate)
    for value in (saved, saved["expression"]):
        value["type"] = value["type"].replace(
            "quail.common.expressions", "quail.logical.expressions")
    assert decode_value(RegularPredicate, saved) == predicate
