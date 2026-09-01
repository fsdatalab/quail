"""Built in physical node definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from .base import (
    ExecutionLocation,
    InputPort,
    OutputPort,
    Partitioning,
    PartitioningKind,
    PhysicalNode,
    PortRef,
    ResourceRequirements,
    ValueType,
)

@dataclass(frozen=True)
class FilterStage:
    """One predicate inside a packed filter."""

    written_pos: int
    question_tokens: int
    preamble_tokens: int
    selectivity: float | None
    expected_docs: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FilterStage":
        return cls(
            written_pos=int(value["written_pos"]),
            question_tokens=int(value["question_tokens"]),
            preamble_tokens=int(value["preamble_tokens"]),
            selectivity=value.get("selectivity"),
            expected_docs=float(value["expected_docs"]),
        )

    def to_dict(self) -> dict:
        return {
            "written_pos": self.written_pos,
            "question_tokens": self.question_tokens,
            "preamble_tokens": self.preamble_tokens,
            "selectivity": self.selectivity,
            "expected_docs": self.expected_docs,
        }


@dataclass(frozen=True)
class JoinStage:
    """One predicate inside an anchored join."""

    written_pos: int
    exec_idx: int
    anchor: str
    partners: tuple[str, ...]
    semantics: str
    selectivity: float | None
    expected_tuples: float
    anchor_frame_tokens: int
    pair_tail_tokens: int
    anchor_resident: str
    tuple_tokens: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "JoinStage":
        return cls(
            written_pos=int(value["written_pos"]),
            exec_idx=int(value["exec_idx"]),
            anchor=str(value["anchor"]),
            partners=tuple(value["partners"]),
            semantics=str(value["semantics"]),
            selectivity=value.get("selectivity"),
            expected_tuples=float(value["expected_tuples"]),
            anchor_frame_tokens=int(value["anchor_frame_tokens"]),
            pair_tail_tokens=int(value["pair_tail_tokens"]),
            anchor_resident=str(value["anchor_resident"]),
            tuple_tokens=float(value["tuple_tokens"]),
        )

    def to_dict(self) -> dict:
        return {
            "written_pos": self.written_pos,
            "exec_idx": self.exec_idx,
            "anchor": self.anchor,
            "partners": list(self.partners),
            "semantics": self.semantics,
            "selectivity": self.selectivity,
            "expected_tuples": self.expected_tuples,
            "anchor_frame_tokens": self.anchor_frame_tokens,
            "pair_tail_tokens": self.pair_tail_tokens,
            "anchor_resident": self.anchor_resident,
            "tuple_tokens": self.tuple_tokens,
        }


@dataclass(frozen=True)
class DocumentScan(PhysicalNode):
    """Read document ids and token rows for one alias."""

    alias: str = ""
    provider: str = ""
    column: str = ""
    n_docs: int = 0
    total_tokens: int = 0
    shards: tuple[tuple[int, ...], ...] = ()
    shard_token_loads: tuple[int, ...] = ()

    type_name: ClassVar[str] = "quail.document_scan.v1"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort(
            f"ids:{self.alias}",
            ValueType.DOCUMENT_IDS,
            schema=(self.alias,),
            partitioning=Partitioning(
                PartitioningKind.HASH,
                (self.alias,),
                len(self.shards) or None,
            ),
        ),)

    def attributes(self, *, include_runtime_data: bool = True) -> dict:
        value = {
            "alias": self.alias,
            "provider": self.provider,
            "column": self.column,
            "n_docs": self.n_docs,
            "total_tokens": self.total_tokens,
            "shard_token_loads": list(self.shard_token_loads),
        }
        if include_runtime_data:
            value["shards"] = [list(shard) for shard in self.shards]
        else:
            value["shard_docs"] = [len(shard) for shard in self.shards]
        return value

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            alias=attributes["alias"],
            provider=attributes["provider"],
            column=attributes["column"],
            n_docs=int(attributes["n_docs"]),
            total_tokens=int(attributes["total_tokens"]),
            shards=tuple(tuple(shard) for shard in attributes.get("shards", ())),
            shard_token_loads=tuple(attributes.get("shard_token_loads", ())),
        )

@dataclass(frozen=True)
class PackedFilter(PhysicalNode):
    """Evaluate ordered predicates on one document input."""

    alias: str = ""
    arena_writes: bool = False
    keep_kv: bool = False
    keep_min_doc_tokens: int = 0
    keep_resident_fraction: float = 0.0
    stages: tuple[FilterStage, ...] = ()
    question_token_ids: tuple[tuple[Any, ...], ...] = ()

    type_name: ClassVar[str] = "quail.packed_filter.v1"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.GPU_EXECUTOR
    backend: ClassVar[str] = "quail"

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (
            OutputPort(
                f"ids:{self.alias}",
                ValueType.DOCUMENT_IDS,
                schema=(self.alias,),
                partitioning=Partitioning(
                    PartitioningKind.HASH, (self.alias,)
                ),
            ),
            OutputPort(
                f"filter_answers:{self.alias}",
                ValueType.FILTER_ANSWERS,
                schema=(self.alias, "answer"),
                partitioning=Partitioning(
                    PartitioningKind.HASH, (self.alias,)
                ),
            ),
        )

    @property
    def resources(self) -> ResourceRequirements:
        return ResourceRequirements(gpus=1)

    def attributes(self, *, include_runtime_data: bool = True) -> dict:
        attributes = {
            "alias": self.alias,
            "arena_writes": self.arena_writes,
            "keep_kv": self.keep_kv,
            "keep_min_doc_tokens": self.keep_min_doc_tokens,
            "keep_resident_fraction": self.keep_resident_fraction,
            "stages": [stage.to_dict() for stage in self.stages],
        }
        if include_runtime_data:
            attributes["question_token_ids"] = [
                list(question) for question in self.question_token_ids
            ]
        return attributes

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            alias=attributes["alias"],
            arena_writes=bool(attributes["arena_writes"]),
            keep_kv=bool(attributes["keep_kv"]),
            keep_min_doc_tokens=int(attributes["keep_min_doc_tokens"]),
            keep_resident_fraction=float(attributes["keep_resident_fraction"]),
            stages=tuple(
                FilterStage.from_mapping(stage)
                for stage in attributes.get("stages", ())
            ),
            question_token_ids=tuple(
                tuple(question)
                for question in attributes.get("question_token_ids", ())
            ),
        )

@dataclass(frozen=True)
class Exchange(PhysicalNode):
    """Move or repartition survivor ids between join steps."""

    next_anchor: str = ""
    aliases: tuple[str, ...] = ()

    type_name: ClassVar[str] = "quail.exchange.v1"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return tuple(
            OutputPort(
                f"ids:{alias}",
                ValueType.DOCUMENT_IDS,
                schema=(alias,),
                partitioning=Partitioning(
                    PartitioningKind.HASH, (self.next_anchor,)
                ),
            )
            for alias in self.aliases
        )

    def attributes(self, *, include_runtime_data: bool = True) -> dict:
        return {"next_anchor": self.next_anchor, "aliases": list(self.aliases)}

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            next_anchor=attributes["next_anchor"],
            aliases=tuple(attributes["aliases"]),
        )

@dataclass(frozen=True)
class AnchoredJoin(PhysicalNode):
    """Evaluate join predicates that use one anchor alias."""

    anchor: str = ""
    anchor_resident: str = "none"
    keep_anchor_kv: bool = False
    stage_idxs: tuple[int, ...] = ()
    stages: tuple[JoinStage, ...] = ()

    type_name: ClassVar[str] = "quail.anchored_join.v1"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.GPU_EXECUTOR
    backend: ClassVar[str] = "quail"

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        outputs = [OutputPort(
            f"ids:{self.anchor}",
            ValueType.DOCUMENT_IDS,
            schema=(self.anchor,),
            partitioning=Partitioning(
                PartitioningKind.HASH, (self.anchor,)
            ),
        )]
        outputs.extend(
            OutputPort(
                f"pairs:{stage.written_pos}",
                ValueType.JOIN_ANSWERS,
                schema=(stage.anchor, *stage.partners, "answer"),
                partitioning=Partitioning(
                    PartitioningKind.HASH, (stage.anchor,)
                ),
            )
            for stage in self.stages
            if stage.semantics == "full"
        )
        return tuple(outputs)

    @property
    def resources(self) -> ResourceRequirements:
        return ResourceRequirements(gpus=1)

    def attributes(self, *, include_runtime_data: bool = True) -> dict:
        return {
            "anchor": self.anchor,
            "anchor_resident": self.anchor_resident,
            "keep_anchor_kv": self.keep_anchor_kv,
            "stage_idxs": list(self.stage_idxs),
            "stages": [stage.to_dict() for stage in self.stages],
        }

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            anchor=attributes["anchor"],
            anchor_resident=attributes["anchor_resident"],
            keep_anchor_kv=bool(attributes["keep_anchor_kv"]),
            stage_idxs=tuple(attributes["stage_idxs"]),
            stages=tuple(
                JoinStage.from_mapping(stage)
                for stage in attributes.get("stages", ())
            ),
        )

@dataclass(frozen=True)
class AdaptiveJoinPlan(PhysicalNode):
    """Choose and run anchored join steps after filters finish."""

    expected_nodes: tuple[AnchoredJoin | Exchange, ...] = ()
    aliases: tuple[str, ...] = ()
    full_join_positions: tuple[int, ...] = ()
    join_specs: tuple[Mapping[str, Any], ...] = ()

    type_name: ClassVar[str] = "quail.adaptive_join_plan.v1"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR
    backend: ClassVar[str] = "quail"

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        outputs = [
            OutputPort(
                f"ids:{alias}",
                ValueType.DOCUMENT_IDS,
                schema=(alias,),
                partitioning=Partitioning(
                    PartitioningKind.HASH, (alias,)
                ),
            )
            for alias in self.aliases
        ]
        outputs.extend(
            OutputPort(f"pairs:{position}", ValueType.JOIN_ANSWERS)
            for position in self.full_join_positions
        )
        return tuple(outputs)

    def attributes(self, *, include_runtime_data: bool = True) -> dict:
        return {
            "aliases": list(self.aliases),
            "full_join_positions": list(self.full_join_positions),
            "expected_nodes": [
                {
                    "type": node.type_name,
                    "id": node.node_id,
                    "inputs": [
                        {
                            "name": port.name,
                            "value_type": port.value_type.value,
                            "schema": list(port.schema),
                            "source": {
                                "node_id": port.source.node_id,
                                "port": port.source.port,
                            },
                        }
                        for port in node.inputs
                    ],
                    "attributes": node.attributes(
                        include_runtime_data=include_runtime_data
                    ),
                }
                for node in self.expected_nodes
            ],
            **({"join_specs": list(self.join_specs)}
               if include_runtime_data else {}),
        }

    def explain_fields(self) -> Mapping[str, Any]:
        return {
            "aliases": list(self.aliases),
            "full_join_positions": list(self.full_join_positions),
            "expected_steps": [
                node.type_name for node in self.expected_nodes
            ],
        }

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        expected = []
        node_types = {
            AnchoredJoin.type_name: AnchoredJoin,
            Exchange.type_name: Exchange,
        }
        for encoded in attributes.get("expected_nodes", ()):
            node_type = node_types.get(encoded["type"])
            if node_type is None:
                raise ValueError(
                    "AdaptiveJoinPlan expected plan contains unknown node "
                    f"type {encoded['type']!r}")
            child_inputs = tuple(
                InputPort(
                    name=input_port["name"],
                    value_type=ValueType(input_port["value_type"]),
                    source=PortRef(
                        input_port["source"]["node_id"],
                        input_port["source"]["port"],
                    ),
                    schema=tuple(input_port.get("schema", ())),
                )
                for input_port in encoded.get("inputs", ())
            )
            expected.append(node_type.from_attributes(
                encoded["id"], child_inputs, encoded["attributes"]
            ))
        return cls(
            node_id=node_id,
            inputs=inputs,
            expected_nodes=tuple(expected),
            aliases=tuple(attributes.get("aliases", ())),
            full_join_positions=tuple(
                attributes.get("full_join_positions", ())
            ),
            join_specs=tuple(attributes.get("join_specs", ())),
        )

@dataclass(frozen=True)
class HashJoin(PhysicalNode):
    """Combine answer relations by exact document ids."""

    alias_order: tuple[str, ...] = ()

    type_name: ClassVar[str] = "quail.hash_join.v1"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort(
            "tuples",
            ValueType.ROWS,
            schema=self.alias_order,
            partitioning=Partitioning(PartitioningKind.SINGLE),
        ),)

    def attributes(self, *, include_runtime_data: bool = True) -> dict:
        return {"alias_order": list(self.alias_order)}

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            alias_order=tuple(attributes["alias_order"]),
        )

@dataclass(frozen=True)
class Project(PhysicalNode):
    """Select the requested result columns."""

    columns: tuple[str, ...] = ()

    type_name: ClassVar[str] = "quail.project.v1"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.CLIENT

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort(
            "rows",
            ValueType.ROWS,
            schema=self.columns,
            partitioning=Partitioning(PartitioningKind.SINGLE),
        ),)

    def attributes(self, *, include_runtime_data: bool = True) -> dict:
        return {"columns": list(self.columns)}

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            columns=tuple(attributes["columns"]),
        )

@dataclass(frozen=True)
class Limit(PhysicalNode):
    """Stop result output after a fixed number of rows."""

    count: int = 0

    type_name: ClassVar[str] = "quail.limit.v1"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.CLIENT

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort("rows", ValueType.ROWS),)

    def attributes(self, *, include_runtime_data: bool = True) -> dict:
        return {"count": self.count}

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(node_id=node_id, inputs=inputs, count=int(attributes["count"]))
