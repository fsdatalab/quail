"""Physical planning and runtime extension test."""

from dataclasses import dataclass, replace
from typing import ClassVar

import pyarrow as pa

import quail
from quail.catalog import DocumentProvider
from quail.extensions import built_in_registry
from quail.physical import (
    DocumentScan,
    ExecutionLocation,
    InputPort,
    NodeCodec,
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

    type_name: ClassVar[str] = "test.first_documents.v1"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.CLIENT

    @property
    def outputs(self):
        return (OutputPort(
            f"ids:{self.alias}",
            ValueType.DOCUMENT_IDS,
            schema=(self.alias,),
        ),)

    def attributes(self, *, include_runtime_data=True):
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
    scan = DocumentScan(
        node_id="scan:d",
        alias="d",
        provider="docs",
        column="body",
        n_docs=len(context.document_tokens["d"]),
    )
    filtered = FirstDocuments(
        node_id="filter:d",
        inputs=(InputPort(
            "input:0",
            ValueType.DOCUMENT_IDS,
            PortRef("scan:d", "ids:d"),
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
        tensor_parallel=1,
        kv_dtype="bf16",
        chunk_tokens=1,
        admission_tokens=1,
        order_rule="as_written",
        order_source=source,
        backend="local_filter",
        estimated_seconds=estimate,
        nodes=(scan, filtered, project),
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


def test_session_runs_registered_physical_extensions_locally():
    registry = built_in_registry()
    backend = LocalFilterBackend()
    planner = PreferredPlanner()
    registry.register_backend(backend)
    registry.register_physical_planner(planner.name, planner)
    registry.register_physical_rule(
        KeepFirstDocument.name, KeepFirstDocument()
    )
    registry.register_codec(NodeCodec(FirstDocuments))
    registry.register_runtime(
        FirstDocuments.runtime_key, FirstDocumentsRuntime()
    )
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
    result = query.run()
    envelope = plan.to_envelope(registry.codecs)

    assert backend.called
    assert planner.called
    assert plan.order_source == "preferred planner"
    assert "physical rule keep_first_document" in plan.remarks[0]
    assert decode_graph(envelope["graph"], registry.codecs) == plan.graph
    assert result.to_rows() == [("a",)]
    assert list(result.report["nodes"]) == [
        "scan:d", "filter:d", "project"
    ]
