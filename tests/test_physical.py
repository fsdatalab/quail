"""Physical planning and runtime extension test."""

import copy
from dataclasses import dataclass, replace
from typing import ClassVar

import pyarrow as pa
import pytest

import quail
from quail.builtins import built_in_registry
from quail.catalog import DocumentProvider
from quail.physical import (
    DocumentInput,
    ExecutionLocation,
    InputPort,
    OutputPort,
    PhysicalGraph,
    PhysicalNode,
    PortRef,
    Project,
    ValueType,
    decode_graph,
)
from quail.planner.plan import EngineConfig, PhysicalPlan
from quail.planning import PhysicalCandidate, SupportResult
from quail.runtime.runner import NodeResult


@dataclass(frozen=True)
class FirstDocuments(PhysicalNode):
    """Return a fixed number of document ids."""

    alias: str = ""
    count: int = 1

    type_name: ClassVar[str] = "test.first_documents"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self):
        return (OutputPort(
            f"ids:{self.alias}",
            ValueType.DOCUMENT_IDS,
            schema=(self.alias,),
        ),)

    def attributes(self):
        return {"alias": self.alias, "count": self.count}

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            alias=attributes["alias"],
            count=int(attributes["count"]),
        )


class FirstDocumentsRuntime:
    def execute(self, node, inputs, context):
        documents = next(iter(inputs.values()))
        return NodeResult({f"ids:{node.alias}": documents[:node.count]})


def local_plan(context, *, count, estimate, source):
    scan = DocumentInput(
        node_id="input:d",
        alias="d",
        input_id="d",
        n_docs=len(context.document_tokens["d"]),
    )
    filtered = FirstDocuments(
        node_id="filter:d",
        inputs=(InputPort(
            "input:0",
            ValueType.DOCUMENT_IDS,
            PortRef("input:d", "ids:d"),
            schema=("d",),
        ),),
        alias="d",
        count=count,
    )
    project = Project(
        node_id="project",
        inputs=(InputPort(
            "input:0",
            ValueType.DOCUMENT_IDS,
            PortRef("filter:d", "ids:d"),
            schema=("d",),
        ),),
        columns=("d.id",),
    )
    plan = PhysicalPlan(
        model=context.model.name,
        device=context.device.name,
        workers=context.gpu_count,
        backend="local_filter",
        estimated_seconds=estimate,
        nodes=(scan, filtered, project),
        settings={"planner_source": source},
    )
    return PhysicalCandidate(plan.graph, plan, estimate)


class LocalFilterBackend:
    name = "local_filter"

    def __init__(self):
        self.called = False

    def supports(self, model, device, gpu_count):
        return SupportResult.accept()

    def plan(self, region, context):
        self.called = True
        return (local_plan(
            context,
            count=1,
            estimate=0.0,
            source="backend",
        ),)


class PreferredPlanner:
    name = "preferred"

    def __init__(self):
        self.called = False

    def plan(self, region, context):
        self.called = True
        return (local_plan(
            context,
            count=2,
            estimate=-1.0,
            source="preferred planner",
        ),)


class KeepFirstDocument:
    name = "keep_first_document"

    def rewrite(self, graph, context):
        return PhysicalGraph(
            tuple(
                replace(node, count=1)
                if isinstance(node, FirstDocuments) else node
                for node in graph.nodes
            ),
            graph.root,
        )


def test_physical_extensions_plan_validate_and_execute(monkeypatch):
    registry = quail.ExtensionRegistry.with_built_ins()
    backend = LocalFilterBackend()
    planner = PreferredPlanner()
    registry.register_backend(backend)
    registry.register_physical_planner(planner)
    registry.register_physical_rule(KeepFirstDocument())
    registry.register_node(FirstDocuments, runtime=FirstDocumentsRuntime())
    session = quail.Session(
        EngineConfig(backend="local_filter"),
        tokenizer=str.split,
        registry=registry,
    )
    session.register("docs", DocumentProvider.from_table(
        pa.table({"id": ["a", "b"], "body": ["one", "two"]}),
        id_col="id",
        identity="local-filter-docs",
    ))
    query = session.sql(
        "SELECT d.id FROM docs d WHERE "
        "AI_FILTER(PROMPT('ok {0}', d.body))"
    )

    plan = query.plan()
    envelope = plan.to_envelope(registry.codecs)

    assert backend.called
    assert planner.called
    assert plan.settings["planner_source"] == "preferred planner"
    assert "physical rule keep_first_document" in plan.remarks[0]
    assert decode_graph(envelope["graph"], registry.codecs) == plan.graph

    registry = built_in_registry()
    scan = DocumentInput(
        node_id="input:d",
        alias="d",
        input_id="d",
        n_docs=1,
        shard_ranges=((0, 1),),
        shard_token_loads=(1,),
    )
    plan = PhysicalGraph((scan,), PortRef("input:d", "ids:d"))
    from quail.physical import encode_graph

    encoded = encode_graph(plan, registry.codecs)
    missing = copy.deepcopy(encoded)
    del missing["nodes"][0]["attributes"]["shard_ranges"]
    with pytest.raises(ValueError, match="missing fields"):
        decode_graph(missing, registry.codecs)

    extra = copy.deepcopy(encoded)
    extra["nodes"][0]["unused"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        decode_graph(extra, registry.codecs)

    with monkeypatch.context() as patch:
        import sys
        import types

        from quail.physical import plan_envelope
        from quail.runtime.execute import _execute_physical

        class ExampleBackend:
            name = "test.example"

            def execute_request(self, context):
                from quail.execution import PhysicalResponse

                return PhysicalResponse({}, {
                    "backend": self.name,
                    "nodes": [node.node_id for node in context.graph.nodes],
                })

        module_name = "test_quail_extension"
        module = types.ModuleType(module_name)

        def register(registry):
            registry.register_backend(ExampleBackend())

        module.register_quail_extension = register
        patch.setitem(sys.modules, module_name, module)
        registry = built_in_registry()
        scan = DocumentInput(
            node_id="input:d",
            alias="d",
            input_id="d",
            n_docs=2,
        )
        graph = PhysicalGraph((scan,), PortRef("input:d", "ids:d"))
        from quail.execution import (
            PhysicalRequest,
            document_input,
        )

        request = PhysicalRequest(
            plan_envelope(
                backend=ExampleBackend.name,
                model="qwen3-4b-fp8",
                device="h100-sxm",
                workers=1,
                graph=graph,
                codecs=registry.codecs,
            ),
            {"d": document_input(pa.array([[1], [2]]))},
        )
        registry.load_extension(module)
        response = _execute_physical(request, registry)

        assert response.metrics == {
            "backend": "test.example",
            "nodes": ["input:d"],
        }


def test_physical_explain_shows_shared_inputs_and_metrics():
    from quail.physical.base import input_ports

    source = DocumentInput(node_id="input:d", alias="d", n_docs=100)
    first = FirstDocuments(
        node_id="first", alias="d", count=5,
        inputs=input_ports((PortRef(source.node_id, "ids:d"),)))
    second = FirstDocuments(
        node_id="second", alias="d", count=10,
        inputs=input_ports((PortRef(source.node_id, "ids:d"),)))
    project = Project(
        node_id="project", columns=("d.id",),
        inputs=input_ports((PortRef(first.node_id, "ids:d"),
                            PortRef(second.node_id, "ids:d"))))
    graph = PhysicalGraph((source, first, second, project),
                          PortRef(project.node_id, "rows"))
    text = graph.explain()
    assert text.startswith("Project: d.id")
    assert text.count("DocumentInput: d [1]") == 1
    assert text.count("Reuse [1]") == 1
    assert "count=5" in text and "count=10" in text
    assert "test.first_documents" in text
    assert "input:d" not in text
    assert "input:d[ids:d]" in graph.explain(verbose=True)

    from quail.explain import physical_tree
    from quail.runtime.runner import NodeMetrics

    node = DocumentInput(node_id="input:d", alias="d", n_docs=100)
    graph = PhysicalGraph((node,), PortRef(node.node_id, "ids:d"))
    assert "metrics unavailable" in physical_tree(graph, metrics={})
    measured = {node.node_id: NodeMetrics(output_rows=42, wall_s=1.25,
                                         fresh_tokens=200)}
    text = physical_tree(graph, metrics=measured)
    assert "actual_rows=42, wall_s=1.250" in text
    assert "estimated_rows" not in text
    assert "fresh_tokens=200" in physical_tree(
        graph, metrics=measured, verbose=True)
