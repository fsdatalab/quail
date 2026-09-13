"""Internal physical execution data types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyarrow as pa

from quail.physical import (
    AiFilter,
    OutputPort,
    PhysicalGraph,
    PortRef,
    ValueType,
)
from quail.runtime.pairs import COLUMNS_PREFIX
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
    """A physical plan, its token input bindings, and its relations.

    relations holds one value table per alias a HashJoin or an apply()
    function reads, keyed ``columns:<alias>``; see quail.runtime.pairs.
    """

    plan: Mapping[str, Any]
    inputs: Mapping[str, TokenizedInput]
    relations: Mapping[str, pa.Table] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for input_id, value in self.inputs.items():
            if not input_id:
                raise ValueError("execution input ids cannot be empty")
            if not isinstance(value, TokenizedInput):
                raise TypeError(
                    "physical execution inputs must be TokenizedInput values"
                )
        for key, value in self.relations.items():
            if not key.startswith(COLUMNS_PREFIX):
                raise ValueError(
                    f"execution relations are keyed {COLUMNS_PREFIX}<alias>, "
                    f"got {key!r}")
            if not isinstance(value, pa.Table):
                raise TypeError("execution relations must be Arrow tables")

    def column_tables(self) -> dict[str, pa.Table]:
        """Return the value tables apply() functions read, by alias."""
        return {key[len(COLUMNS_PREFIX):]: table
                for key, table in self.relations.items()
                if key.startswith(COLUMNS_PREFIX)}

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


def join_answer_cells(value: Mapping[str, Any]):
    """Yield (anchor local index, partner member index, answer) triples.

    rows[local] runs over the partner members the anchor streamed:
    every member of partner_index, or the member indices listed in
    anchor_partners[local] when the join ran over pairs.
    """
    members = value.get("anchor_partners") or {}
    for raw_local, row in value["rows"].items():
        local = int(raw_local)
        streamed = members.get(local)
        for position, answer in enumerate(row):
            yield (local,
                   position if streamed is None else int(streamed[position]),
                   answer)


def _join_answers_table(value: Mapping[str, Any]) -> pa.Table:
    """Build the join answer table, one numpy slice per anchor.

    A join can hold tens of millions of pairs, so the work per pair
    stays in numpy; only the anchors are visited from Python.
    """
    anchor = str(value["anchor"])
    partners = tuple(value["partners"])
    aliases = (anchor, *partners)
    anchor_map = value["anchor_index"]
    # members x partners, so a member index selects every partner at once
    partner_map = np.asarray(value["partner_index"], dtype=np.int32).reshape(
        -1, len(partners))
    streamed = value.get("anchor_partners") or {}
    anchor_parts, member_parts, answer_parts = [], [], []
    for raw_local, row in value["rows"].items():
        local = int(raw_local)
        members = streamed.get(local)
        members = (np.arange(len(row), dtype=np.int32) if members is None
                   else np.asarray(members, dtype=np.int32))
        anchor_parts.append(np.full(len(row), int(anchor_map[local]), np.int32))
        member_parts.append(members)
        answer_parts.append(np.asarray(row, dtype=bool))
    empty = np.zeros(0, np.int32)
    members = np.concatenate(member_parts) if member_parts else empty
    columns = {anchor: np.concatenate(anchor_parts) if anchor_parts else empty}
    for index, alias in enumerate(partners):
        columns[alias] = partner_map[members, index] if len(members) else empty
    answers = (np.concatenate(answer_parts) if answer_parts
               else np.zeros(0, bool))
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
