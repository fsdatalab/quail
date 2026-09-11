"""Shared types for typed physical plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, ClassVar


class ExecutionLocation(str, Enum):
    """Location where a physical node runs."""

    COORDINATOR = "coordinator"
    GPU_EXECUTOR = "gpu_executor"


class ValueType(str, Enum):
    """Value passed between physical node ports."""

    DOCUMENT_IDS = "document_ids"
    FILTER_ANSWERS = "filter_answers"
    JOIN_ANSWERS = "join_answers"
    ROWS = "rows"
    ANY = "any"


@dataclass(frozen=True)
class PortRef:
    """Reference to one output port on another node."""

    node_id: str
    port: str


@dataclass(frozen=True)
class InputPort:
    """Named input and the output port that supplies it."""

    name: str
    value_type: ValueType
    source: PortRef
    schema: tuple[str, ...] = ()


@dataclass(frozen=True)
class OutputPort:
    """Named output produced by a physical node."""

    name: str
    value_type: ValueType
    schema: tuple[str, ...] = ()


def value_type_for_port(name: str) -> ValueType:
    """Return the built in value type for a port name."""
    if name.startswith("ids:"):
        return ValueType.DOCUMENT_IDS
    if name.startswith("filter_answers:"):
        return ValueType.FILTER_ANSWERS
    if name.startswith("join_answers:"):
        return ValueType.JOIN_ANSWERS
    if name in {"tuples", "rows"}:
        return ValueType.ROWS
    return ValueType.ANY


def input_ports(values: tuple[PortRef, ...]) -> tuple[InputPort, ...]:
    """Create typed input ports for output references."""
    return tuple(
        InputPort(
            name=f"input:{index}",
            value_type=value_type_for_port(value.port),
            source=value,
        )
        for index, value in enumerate(values)
    )


@dataclass(frozen=True)
class PhysicalNode:
    """Base class for a node in a physical graph."""

    node_id: str
    inputs: tuple[InputPort, ...] = ()

    type_name: ClassVar[str] = "quail.physical_node"
    runtime_key: ClassVar[str] = "quail.physical_node"
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR
    backend: ClassVar[str | None] = None

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        raise NotImplementedError

    def attributes(self) -> dict:
        """Return node fields encoded by the node codec."""
        return {}

    @classmethod
    def from_attributes(
        cls,
        node_id: str,
        inputs: tuple[InputPort, ...],
        attributes: Mapping[str, Any],
    ) -> "PhysicalNode":
        """Decode a node from codec fields."""
        return cls(node_id=node_id, inputs=inputs)

    def with_inputs(self, inputs: tuple[InputPort, ...]) -> "PhysicalNode":
        """Return the same node with replacement inputs."""
        return replace(self, inputs=inputs)

    def embedded_nodes(self) -> tuple["PhysicalNode", ...]:
        """Return physical nodes stored inside this planning node."""
        return ()

    def streamed_inputs(self) -> tuple[str, ...]:
        """Return input port names read as a stream from their producer.

        The runner does not execute a streamed port's producer on its
        own. The consumer's runtime runs it and returns its results.
        """
        return ()

    def explain_fields(self) -> Mapping[str, Any]:
        """Return fields shown on one explain line."""
        return self.attributes()


class GraphValidationError(ValueError):
    """Raised when a physical graph is invalid."""


@dataclass(frozen=True)
class PhysicalGraph:
    """Immutable graph of typed physical nodes."""

    nodes: tuple[PhysicalNode, ...]
    root: PortRef

    def node(self, node_id: str) -> PhysicalNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def validate(self, runtime_keys: set[str] | None = None) -> None:
        """Validate ids, ports, types, cycles, root, and runtimes."""
        by_id: dict[str, PhysicalNode] = {}
        for node in self.nodes:
            if node.node_id in by_id:
                raise GraphValidationError(
                    f"duplicate physical node id {node.node_id!r}")
            by_id[node.node_id] = node

        output_types = {
            (node.node_id, output.name): output.value_type
            for node in self.nodes
            for output in node.outputs
        }
        for node in self.nodes:
            for input_port in node.inputs:
                key = (input_port.source.node_id, input_port.source.port)
                if key not in output_types:
                    raise GraphValidationError(
                        f"{node.node_id!r} reads missing output "
                        f"{key[0]!r}.{key[1]!r}")
                actual = output_types[key]
                expected = input_port.value_type
                if expected is not ValueType.ANY \
                        and actual is not ValueType.ANY \
                        and expected is not actual:
                    raise GraphValidationError(
                        f"{node.node_id!r} input {input_port.name!r} "
                        f"expects {expected.value}, got {actual.value}")
                source_node = by_id[input_port.source.node_id]
                source_output = next(
                    output for output in source_node.outputs
                    if output.name == input_port.source.port
                )
                if input_port.schema and source_output.schema \
                        and input_port.schema != source_output.schema:
                    raise GraphValidationError(
                        f"{node.node_id!r} input {input_port.name!r} "
                        "has a different schema from its source")

        if (self.root.node_id, self.root.port) not in output_types:
            raise GraphValidationError(
                f"root output {self.root.node_id!r}.{self.root.port!r} "
                "does not exist")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise GraphValidationError("physical graph contains a cycle")
            if node_id in visited:
                return
            visiting.add(node_id)
            for input_port in by_id[node_id].inputs:
                visit(input_port.source.node_id)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in by_id:
            visit(node_id)

        if runtime_keys is not None:
            missing = sorted({
                node.runtime_key for node in self.nodes
                if node.runtime_key not in runtime_keys
            })
            if missing:
                raise GraphValidationError(
                    f"physical graph needs unregistered runtimes {missing}")

    def validate_backend(self, backend: str) -> None:
        """Check that every model node uses the selected backend."""
        wrong = sorted(
            f"{node.node_id}:{node.backend}"
            for node in self.nodes
            if node.backend is not None and node.backend != backend
        )
        if wrong:
            raise GraphValidationError(
                f"physical graph selects backend {backend!r} but has "
                f"model nodes {wrong}")

    def topological_nodes(self) -> tuple[PhysicalNode, ...]:
        """Return nodes ordered after all their inputs."""
        by_id = {node.node_id: node for node in self.nodes}
        ordered: list[PhysicalNode] = []
        visited: set[str] = set()

        def visit(node: PhysicalNode) -> None:
            if node.node_id in visited:
                return
            for input_port in node.inputs:
                visit(by_id[input_port.source.node_id])
            visited.add(node.node_id)
            ordered.append(node)

        for node in self.nodes:
            visit(node)
        return tuple(ordered)

    def nodes_by_type(self, type_name: str) -> tuple[PhysicalNode, ...]:
        """Return graph and embedded nodes with one registered type name."""
        found = []

        def visit(node: PhysicalNode) -> None:
            if node.type_name == type_name:
                found.append(node)
            for child in node.embedded_nodes():
                visit(child)

        for node in self.nodes:
            visit(node)
        return tuple(found)

    def explain(self, *, verbose: bool = False) -> str:
        """Return an operator tree, optionally including internal fields."""
        from quail.explain import physical_tree

        return physical_tree(self, verbose=verbose)
