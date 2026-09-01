"""Versioned JSON codecs for physical plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .base import InputPort, PhysicalGraph, PhysicalNode, PortRef, ValueType
from .nodes import (
    AdaptiveJoinPlan,
    AnchoredJoin,
    DocumentScan,
    Exchange,
    HashJoin,
    Limit,
    PackedFilter,
    Project,
)


PLAN_FORMAT_VERSION = 1


@dataclass(frozen=True)
class NodeCodec:
    """Encode and decode one registered physical node type."""

    node_type: type[PhysicalNode]

    @property
    def type_name(self) -> str:
        return self.node_type.type_name

    def encode(
        self,
        node: PhysicalNode,
        *,
        include_runtime_data: bool = True,
    ) -> dict:
        if not isinstance(node, self.node_type):
            raise TypeError(
                f"codec {self.type_name!r} cannot encode "
                f"{type(node).__name__}")
        return {
            "type": self.type_name,
            "id": node.node_id,
            "inputs": [
                {
                    "name": input_port.name,
                    "value_type": input_port.value_type.value,
                    "schema": list(input_port.schema),
                    "source": {
                        "node_id": input_port.source.node_id,
                        "port": input_port.source.port,
                    },
                }
                for input_port in node.inputs
            ],
            "attributes": node.attributes(
                include_runtime_data=include_runtime_data
            ),
        }

    def decode(self, value: Mapping[str, Any]) -> PhysicalNode:
        inputs = tuple(
            InputPort(
                name=input_port["name"],
                value_type=ValueType(input_port["value_type"]),
                source=PortRef(
                    input_port["source"]["node_id"],
                    input_port["source"]["port"],
                ),
                schema=tuple(input_port.get("schema", ())),
            )
            for input_port in value.get("inputs", ())
        )
        return self.node_type.from_attributes(
            value["id"], inputs, value.get("attributes", {})
        )


BUILT_IN_NODE_TYPES = (
    DocumentScan,
    PackedFilter,
    Exchange,
    AnchoredJoin,
    AdaptiveJoinPlan,
    HashJoin,
    Project,
    Limit,
)


def built_in_codecs() -> tuple[NodeCodec, ...]:
    """Return codecs for every built in physical node."""
    return tuple(NodeCodec(node_type) for node_type in BUILT_IN_NODE_TYPES)


def encode_graph(
    graph: PhysicalGraph,
    codecs: Mapping[str, NodeCodec],
    *,
    include_runtime_data: bool = True,
) -> dict:
    """Encode one physical graph."""
    return {
        "nodes": [
            codecs[node.type_name].encode(
                node, include_runtime_data=include_runtime_data
            )
            for node in graph.nodes
        ],
        "root": {
            "node_id": graph.root.node_id,
            "port": graph.root.port,
        },
    }


def decode_graph(
    value: Mapping[str, Any], codecs: Mapping[str, NodeCodec]
) -> PhysicalGraph:
    """Decode and validate one physical graph."""
    nodes = []
    for encoded in value["nodes"]:
        type_name = encoded["type"]
        if type_name not in codecs:
            raise ValueError(f"unknown physical node type {type_name!r}")
        nodes.append(codecs[type_name].decode(encoded))
    root = value["root"]
    graph = PhysicalGraph(
        nodes=tuple(nodes),
        root=PortRef(root["node_id"], root["port"]),
    )
    graph.validate()
    return graph


def plan_envelope(
    *,
    backend: str,
    model: str,
    device: str,
    workers: int,
    graph: PhysicalGraph,
    codecs: Mapping[str, NodeCodec],
    extension_modules: tuple[str, ...] = (),
    include_runtime_data: bool = True,
) -> dict:
    """Encode the physical plan fields sent across a process boundary."""
    graph.validate_backend(backend)
    return {
        "version": PLAN_FORMAT_VERSION,
        "backend": backend,
        "model": model,
        "device": device,
        "workers": workers,
        "extension_modules": list(extension_modules),
        "node_types": sorted({node.type_name for node in graph.nodes}),
        "graph": encode_graph(
            graph,
            codecs,
            include_runtime_data=include_runtime_data,
        ),
    }


def check_plan_envelope(value: Mapping[str, Any]) -> None:
    """Reject an unsupported physical plan envelope."""
    version = value.get("version")
    if version != PLAN_FORMAT_VERSION:
        raise ValueError(
            f"unsupported physical plan version {version!r}; "
            f"expected {PLAN_FORMAT_VERSION}")
    modules = value.get("extension_modules", [])
    if not isinstance(modules, list) \
            or not all(isinstance(module, str) and module for module in modules):
        raise ValueError("physical plan extension_modules must be strings")
    if len(modules) != len(set(modules)):
        raise ValueError("physical plan has duplicate extension modules")
