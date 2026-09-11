"""Internal physical execution data types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from quail.physical import (
    AiFilter,
    OutputPort,
    PhysicalGraph,
    PortRef,
    ValueType,
)
from quail.runtime.tokens import decode_token_documents


@dataclass(frozen=True)
class TokenizedInput:
    """A random access sequence of tokenized documents."""

    documents: Sequence

    def __post_init__(self) -> None:
        if not hasattr(self.documents, "__len__") \
                or not hasattr(self.documents, "__getitem__"):
            raise TypeError(
                "a tokenized input needs a random access document sequence"
            )

    def __len__(self) -> int:
        return len(self.documents)


def document_input(tokens) -> TokenizedInput:
    """Build one physical input from a token document sequence."""
    return TokenizedInput(decode_token_documents(tokens))


@dataclass(frozen=True)
class PhysicalRequest:
    """A physical plan and its token input bindings."""

    plan: Mapping[str, Any]
    inputs: Mapping[str, TokenizedInput]

    def __post_init__(self) -> None:
        for input_id, value in self.inputs.items():
            if not input_id:
                raise ValueError("execution input ids cannot be empty")
            if not isinstance(value, TokenizedInput):
                raise TypeError(
                    "physical execution inputs must be TokenizedInput values"
                )

    @property
    def gpu_count(self) -> int:
        """Return the GPU count selected by the physical plan."""
        return int(self.plan["workers"])


@dataclass(frozen=True)
class PhysicalResponse:
    """Arrow output relations and metrics from one physical plan."""

    outputs: Mapping[PortRef, pa.Table]
    metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        for ref, value in self.outputs.items():
            if not isinstance(ref, PortRef):
                raise TypeError("execution output keys must be PortRef values")
            if not isinstance(value, pa.Table):
                raise TypeError("physical execution outputs must be Arrow tables")
        if not isinstance(self.metrics, Mapping):
            raise TypeError("execution metrics must be a mapping")

    def output(self, node_id: str, port: str) -> pa.Table:
        """Return one physical output relation."""
        try:
            return self.outputs[PortRef(node_id, port)]
        except KeyError as error:
            raise KeyError(f"missing physical output {node_id}.{port}") \
                from error


def _document_ids_table(port: OutputPort, value: Any) -> pa.Table:
    alias = port.schema[0]
    schema = pa.schema(
        [pa.field(alias, pa.int32(), nullable=False)],
        metadata={
            b"quail.kind": b"document_ids",
        },
    )
    return pa.Table.from_arrays(
        [pa.array(value, type=pa.int32())], schema=schema
    )


def _filter_answers_table(node: AiFilter, value: Mapping) -> pa.Table:
    documents = []
    positions = []
    answers = []
    for document, row in value.items():
        for stage_index, answer in enumerate(row):
            documents.append(int(document))
            positions.append(node.stages[stage_index].written_pos)
            answers.append(bool(answer))
    schema = pa.schema(
        [
            pa.field(node.alias, pa.int32(), nullable=False),
            pa.field("predicate", pa.int32(), nullable=False),
            pa.field("answer", pa.bool_(), nullable=False),
        ],
        metadata={
            b"quail.kind": b"filter_answers",
            b"quail.alias": node.alias.encode("utf-8"),
        },
    )
    return pa.Table.from_arrays(
        [
            pa.array(documents, type=pa.int32()),
            pa.array(positions, type=pa.int32()),
            pa.array(answers, type=pa.bool_()),
        ],
        schema=schema,
    )


def _join_answers_table(value: Mapping[str, Any]) -> pa.Table:
    anchor = str(value["anchor"])
    partners = tuple(value["partners"])
    aliases = (anchor, *partners)
    columns = {alias: [] for alias in aliases}
    answers = []
    anchor_map = value["anchor_index"]
    partner_map = value["partner_index"]
    for raw_local, row in value["rows"].items():
        local = int(raw_local)
        for member_index, answer in enumerate(row):
            columns[anchor].append(int(anchor_map[local]))
            for alias, document in zip(partners, partner_map[member_index]):
                columns[alias].append(int(document))
            answers.append(bool(answer))
    fields = []
    fields.extend(
        pa.field(alias, pa.int32(), nullable=False) for alias in aliases
    )
    fields.append(pa.field("answer", pa.bool_(), nullable=False))
    metadata = {
        b"quail.kind": b"join_answers",
        b"quail.anchor": anchor.encode("utf-8"),
        b"quail.partners": ",".join(partners).encode("utf-8"),
        b"quail.semantics": str(value["semantics"]).encode("utf-8"),
        b"quail.written_pos": str(value["written_pos"]).encode("ascii"),
    }
    if value.get("selectivity") is not None:
        metadata[b"quail.selectivity"] = str(
            value["selectivity"]
        ).encode("ascii")
    arrays = []
    arrays.extend(
        pa.array(columns[alias], type=pa.int32()) for alias in aliases
    )
    arrays.append(pa.array(answers, type=pa.bool_()))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata))


def _output_table(node, port: OutputPort, value: Any) -> pa.Table:
    if isinstance(value, pa.Table):
        return value
    if port.value_type is ValueType.DOCUMENT_IDS:
        return _document_ids_table(port, value)
    if port.value_type is ValueType.FILTER_ANSWERS:
        if not isinstance(node, AiFilter):
            raise TypeError("filter answer output needs a AiFilter node")
        return _filter_answers_table(node, value)
    if port.value_type is ValueType.JOIN_ANSWERS:
        return _join_answers_table(value)
    raise TypeError(
        f"physical output {node.node_id}.{port.name} must be an Arrow table"
    )


def export_physical_outputs(graph: PhysicalGraph, run_result) -> dict:
    """Convert physical node outputs from one run to Arrow relations."""
    outputs = {}
    for node in graph.nodes:
        if node.node_id not in run_result.nodes:
            continue
        result = run_result.nodes[node.node_id]
        ports = {port.name: port for port in node.outputs}
        for name, value in result.outputs.items():
            outputs[PortRef(node.node_id, name)] = _output_table(
                node, ports[name], value
            )
    return outputs
