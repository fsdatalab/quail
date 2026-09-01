"""Count documents inside the Modal execution graph."""

from dataclasses import dataclass, replace
from typing import ClassVar

from quail.physical import (
    ExecutionLocation,
    NodeCodec,
    OutputPort,
    PhysicalGraph,
    PhysicalNode,
    PortRef,
    ValueType,
)
from quail.runtime.runner import NodeMetrics, NodeResult


@dataclass(frozen=True)
class CountDocuments(PhysicalNode):
    """Pass document ids through and report their count."""

    alias: str = ""

    type_name: ClassVar[str] = "example.count_documents.v1"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self):
        return (OutputPort(
            f"ids:{self.alias}",
            ValueType.DOCUMENT_IDS,
            schema=(self.alias,),
        ),)

    def attributes(self, *, include_runtime_data=True):
        return {"alias": self.alias}

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(node_id=node_id, inputs=inputs, alias=attributes["alias"])


class CountDocumentsRuntime:
    """Execute CountDocuments inside the Modal coordinator."""

    def execute(self, node, inputs, context):
        documents = list(next(iter(inputs.values())))
        return NodeResult(
            {f"ids:{node.alias}": documents},
            NodeMetrics(
                input_rows=len(documents),
                output_rows=len(documents),
                evaluated_documents=len(documents),
            ),
        )


class AddDocumentCounts:
    """Place CountDocuments before each model filter."""

    name = "example.add_document_counts"

    def rewrite(self, graph, context):
        from quail.physical import PackedFilter

        nodes = []
        changed = False
        for node in graph.nodes:
            if not isinstance(node, PackedFilter):
                nodes.append(node)
                continue
            count_id = f"count:{node.node_id}"
            source = node.inputs[0]
            counter = CountDocuments(
                node_id=count_id,
                inputs=(source,),
                alias=node.alias,
            )
            counted_input = replace(
                source,
                source=PortRef(count_id, f"ids:{node.alias}"),
            )
            nodes.extend((counter, replace(node, inputs=(counted_input,))))
            changed = True
        if not changed:
            return None
        return PhysicalGraph(tuple(nodes), graph.root)


def register_quail_extension(registry):
    """Register the example node, runtime, and physical rule."""
    registry.register_codec(NodeCodec(CountDocuments))
    registry.register_runtime(
        CountDocuments.runtime_key, CountDocumentsRuntime()
    )
    registry.register_physical_rule(
        AddDocumentCounts.name, AddDocumentCounts()
    )
