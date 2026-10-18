"""Physical planning and runtime extension test."""

import copy
import re
import sys
import types
from dataclasses import dataclass, replace
from typing import ClassVar

import pyarrow as pa
import pytest

import quail
from quail.api.execute import _execute_physical
from quail.builtins import built_in_registry
from quail.catalog import DocumentProvider
from quail.execution.metrics import NodeResult
from quail.execution.runtimes import BarrierRuntime
from quail.execution.types import PhysicalRequest, PhysicalResponse, document_input
from quail.logical import LogicalPlan
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
from quail.planner.optimizer.candidates import PhysicalCandidate, SupportResult
from quail.planner.optimizer.rules import PhysicalRule
from quail.planner.plan import EngineConfig, PhysicalPlan


def test_barrier_preserves_survivor_order_and_duplicates():
    from quail.physical.base import input_ports

    node = Barrier(node_id="barrier", aliases=("a", "b"), inputs=input_ports((
        PortRef("scan:a", "rows:a"), PortRef("scan:b", "rows:b"),
        PortRef("join", "join_answers:0"))))
    for arrow in (False, True):
        for keep in (False, True):
            ids = ([3, 1, 3, 2], [8, 7])
            inputs = {f"input:{i}": pa.table({alias: values}) if arrow else values
                      for i, (alias, values) in enumerate(zip(node.aliases, ids))}
            inputs["input:2"] = pa.table({"a": [1, 3, 2, 9], "b": [7, 8, 7, 8],
                                          "answer": [keep, keep, False, keep]})
            outputs = BarrierRuntime().execute(node, inputs, None).outputs
            values = [outputs[f"rows:{alias}"].ids[alias].to_pylist()
                      for alias in node.aliases]
            assert values == ([[3, 1, 3], [8, 7]] if keep else [[], []])


@pytest.mark.parametrize("shape", ["chain", "cycle", "disconnected"])
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

    def attributes(self):
        return {"alias": self.alias, "count": self.count}

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(node_id=node_id, inputs=inputs, alias=attributes["alias"],
                   count=int(attributes["count"]))


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


class PreferredPlanner:
    name = "preferred"

    def __init__(self):
        self.called = False

    def plan(self, region, context):
        self.called = True
        return (local_plan(
            context, count=2, estimate=-1.0, source="preferred planner"),)


class KeepFirstDocument(PhysicalRule):
    name = "keep_first_document"

    def rewrite(self, graph, context):
        return PhysicalGraph(
            tuple(replace(node, count=1) if isinstance(node, FirstDocuments)
                  else node for node in graph.nodes),
            graph.root)


class TwoCandidatesBackend:
    """Offers the plan as written and an alternative, at chosen estimates."""

    name = "local_filter"

    def __init__(self, written, alternative):
        self.estimates = (written, alternative)
        self.alternative = None

    def supports(self, model, device, gpus):
        return SupportResult.accept()

    def plan(self, region, context):
        written, alternative = self.estimates
        self.alternative = LogicalPlan(region.logical_plan.root)
        return (
            local_plan(context, count=1, estimate=written, source="written"),
            replace(local_plan(context, count=1, estimate=alternative,
                               source="alternative"),
                    logical_plan=self.alternative),
        )


class RecordLogicalPlan(PhysicalRule):
    name = "record_logical_plan"

    def __init__(self):
        self.seen = []

    def rewrite(self, graph, context):
        self.seen.append(context.logical_plan)
        return None


@pytest.mark.parametrize("estimates, chosen", [
    ((1.0, 1.0), "written"), ((2.0, 1.0), "alternative")])
def test_plan_query_keeps_the_cheapest_candidate_and_its_logical_plan(
        estimates, chosen):
    registry = quail.ExtensionRegistry.with_built_ins()
    backend = TwoCandidatesBackend(*estimates)
    registry.register_backend(backend)
    rule = RecordLogicalPlan()
    registry.register_physical_rule(rule)
    session = quail.Session(
        EngineConfig(gpus=1, model="qwen3-4b-fp8", backend="local_filter",
                     device="h100-sxm"),
        tokenizer=str.split, registry=registry)
    session.register("documents", DocumentProvider.from_table(
        pa.table({"id": ["a", "b"], "body": ["one", "two"]}),
        id_col="id", identity="two-candidate-docs"))
    query = session.sql(
        "SELECT d.id FROM documents d WHERE AI_FILTER(PROMPT('ok {0}', d.body))")
    plan = query.plan()
    assert plan.settings["planner_source"] == chosen
    written, alternative = rule.seen
    assert query.logical is (written if chosen == "written" else alternative)
    assert plan.logical_plan is query.logical
    assert alternative is backend.alternative
    assert [candidate["selected"] for candidate in plan.settings["candidates"]] == (
        [True, False] if chosen == "written" else [False, True])


def test_physical_extensions_plan_validate_and_execute(monkeypatch):
    registry = quail.ExtensionRegistry.with_built_ins()
    backend = LocalFilterBackend()
    planner = PreferredPlanner()
    registry.register_backend(backend)
    registry.register_physical_planner(planner)
    registry.register_physical_rule(KeepFirstDocument())
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
    assert planner.called
    assert plan.settings["planner_source"] == "preferred planner"
    # the registered extension rule rewrote the planner's node
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


def test_physical_explain_shows_shared_inputs_and_scalar_metrics():
    from quail.execution.metrics import NodeMetrics, scalar_node_metrics
    from quail.explain import physical_tree
    from quail.physical.base import input_ports

    source = Scan(node_id="input:d", alias="d", num_documents=100)
    first = FirstDocuments(
        node_id="first", alias="d", count=5,
        inputs=input_ports((PortRef(source.node_id, "rows:d"),)))
    second = FirstDocuments(
        node_id="second", alias="d", count=10,
        inputs=input_ports((PortRef(source.node_id, "rows:d"),)))
    project = Project(
        node_id="project", columns=("d.id",),
        inputs=input_ports((PortRef(first.node_id, "rows:d"),
                            PortRef(second.node_id, "rows:d"))))
    graph = PhysicalGraph((source, first, second, project),
                          PortRef(project.node_id, "rows"))
    text = physical_tree(graph)
    assert text.splitlines()[0].split() == ["est.", "rows"]
    assert text.splitlines()[1].startswith("Project: d.id")
    assert text.count("Scan: d [1]") == 1
    assert text.count("Reuse [1]") == 1
    assert "count=5" in text and "count=10" in text
    assert "test.first_documents" in text
    assert "input:d" not in text
    assert "input:d[rows:d]" in physical_tree(graph, verbose=True)

    node = Scan(node_id="input:d", alias="d", num_documents=100)
    graph = PhysicalGraph((node,), PortRef(node.node_id, "rows:d"))
    assert "metrics unavailable" in physical_tree(graph, metrics={})
    measured = {node.node_id: NodeMetrics(output_rows=42, wall_s=1.25,
                                         fresh_tokens=200)}
    text = physical_tree(graph, metrics=measured)
    assert re.search(r"Scan: d\s+100\s+42\s+1\.25 s\s+200$", text,
                     re.MULTILINE)
    assert "est. time" not in text
    assert "fresh_tokens=200" in physical_tree(
        graph, metrics=measured, verbose=True)

    nodes = {"ai_filter:r": NodeResult({}, NodeMetrics(
        fresh_tokens=10,
        extension={"borrowed_prefix_tokens": 64, "answers": [1, 0],
                   "flag": True}))}
    metrics = scalar_node_metrics(nodes)["ai_filter:r"]
    assert metrics["fresh_tokens"] == 10
    assert metrics["borrowed_prefix_tokens"] == 64
    assert "answers" not in metrics and "flag" not in metrics
