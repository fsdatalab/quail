"""Built in physical node definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from .base import (
    ExecutionLocation,
    InputPort,
    OutputPort,
    PhysicalNode,
    PortRef,
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
            selectivity=value["selectivity"],
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
            selectivity=value["selectivity"],
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
class RequestFilterSpec:
    """One filter chain submitted as model requests."""

    alias: str
    written_positions: tuple[int, ...]
    question_token_ids: tuple[tuple[Any, ...], ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RequestFilterSpec":
        return cls(
            alias=str(value["alias"]),
            written_positions=tuple(
                int(position) for position in value["written_positions"]
            ),
            question_token_ids=tuple(
                tuple(question) for question in value["question_token_ids"]
            ),
        )

    def to_dict(self) -> dict:
        return {
            "alias": self.alias,
            "written_positions": list(self.written_positions),
            "question_token_ids": [
                list(question) for question in self.question_token_ids
            ],
        }


@dataclass(frozen=True)
class RequestJoinSpec:
    """One join predicate submitted over its complete cross product."""

    written_pos: int
    aliases: tuple[str, ...]
    outer_aliases: tuple[str, ...]
    anchor: str | None
    semantics: str
    selectivity: float | None
    label_token_ids: tuple[tuple[str, tuple[Any, ...]], ...]
    frame_token_ids: tuple[tuple[str, tuple[Any, ...]], ...]
    tail_token_ids: tuple[Any, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RequestJoinSpec":
        return cls(
            written_pos=int(value["written_pos"]),
            aliases=tuple(value["aliases"]),
            outer_aliases=tuple(value["outer_aliases"]),
            anchor=value["anchor"],
            semantics=str(value["semantics"]),
            selectivity=value["selectivity"],
            label_token_ids=tuple(
                (str(alias), tuple(tokens))
                for alias, tokens in value["label_token_ids"]
            ),
            frame_token_ids=tuple(
                (str(alias), tuple(tokens))
                for alias, tokens in value["frame_token_ids"]
            ),
            tail_token_ids=tuple(value["tail_token_ids"]),
        )

    def to_dict(self) -> dict:
        return {
            "written_pos": self.written_pos,
            "aliases": list(self.aliases),
            "outer_aliases": list(self.outer_aliases),
            "anchor": self.anchor,
            "semantics": self.semantics,
            "selectivity": self.selectivity,
            "label_token_ids": [
                [alias, list(tokens)] for alias, tokens in self.label_token_ids
            ],
            "frame_token_ids": [
                [alias, list(tokens)] for alias, tokens in self.frame_token_ids
            ],
            "tail_token_ids": list(self.tail_token_ids),
        }


@dataclass(frozen=True)
class DocumentInput(PhysicalNode):
    """Read one tokenized document input supplied by the coordinator."""

    alias: str = ""
    input_id: str = ""
    n_docs: int = 0
    total_tokens: int = 0
    shard_ranges: tuple[tuple[int, int], ...] = ()
    shard_token_loads: tuple[int, ...] = ()

    type_name: ClassVar[str] = "quail.document_input"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    def __post_init__(self) -> None:
        if len(self.shard_ranges) != len(self.shard_token_loads):
            raise ValueError(
                "document shard ranges and token loads must have equal size"
            )
        expected = 0
        for start, stop in self.shard_ranges:
            if start != expected or stop < start:
                raise ValueError(
                    "document shard ranges must be ordered and contiguous"
                )
            expected = stop
        if self.shard_ranges and expected != self.n_docs:
            raise ValueError(
                "document shard ranges must cover every document"
            )

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort(
            f"ids:{self.alias}",
            ValueType.DOCUMENT_IDS,
            schema=(self.alias,),
        ),)

    def attributes(self) -> dict:
        return {
            "alias": self.alias,
            "input_id": self.input_id,
            "n_docs": self.n_docs,
            "total_tokens": self.total_tokens,
            "shard_token_loads": list(self.shard_token_loads),
            "shard_ranges": [list(bounds) for bounds in self.shard_ranges],
        }

    @property
    def shards(self) -> tuple[range, ...]:
        """Return each worker's contiguous document range."""
        return tuple(range(start, stop) for start, stop in self.shard_ranges)

    def explain_fields(self) -> Mapping[str, Any]:
        return {
            "alias": self.alias,
            "input_id": self.input_id,
            "n_docs": self.n_docs,
            "total_tokens": self.total_tokens,
            "shard_token_loads": list(self.shard_token_loads),
            "shard_docs": [stop - start
                           for start, stop in self.shard_ranges],
        }

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            alias=attributes["alias"],
            input_id=attributes["input_id"],
            n_docs=int(attributes["n_docs"]),
            total_tokens=int(attributes["total_tokens"]),
            shard_ranges=tuple(
                (int(start), int(stop))
                for start, stop in attributes["shard_ranges"]
            ),
            shard_token_loads=tuple(attributes["shard_token_loads"]),
        )


@dataclass(frozen=True)
class RequestExecution(PhysicalNode):
    """Run filters and joins through an independent request engine."""

    backend_name: str = ""
    aliases: tuple[str, ...] = ()
    preamble_token_ids: tuple[Any, ...] = ()
    filters: tuple[RequestFilterSpec, ...] = ()
    joins: tuple[RequestJoinSpec, ...] = ()

    type_name: ClassVar[str] = "quail.request_execution"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.GPU_EXECUTOR

    @property
    def backend(self) -> str:
        """Return the backend selected for this model node."""
        return self.backend_name

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        outputs = [
            OutputPort(
                f"ids:{alias}",
                ValueType.DOCUMENT_IDS,
                schema=(alias,),
            )
            for alias in self.aliases
        ]
        outputs.extend(
            OutputPort(
                f"filter_answers:{spec.alias}",
                ValueType.FILTER_ANSWERS,
                schema=(spec.alias, "predicate", "answer"),
            )
            for spec in self.filters
        )
        outputs.extend(
            OutputPort(
                f"join_answers:{spec.written_pos}",
                ValueType.JOIN_ANSWERS,
                schema=(*spec.aliases, "answer"),
            )
            for spec in self.joins
        )
        return tuple(outputs)

    def attributes(self) -> dict:
        return {
            "backend_name": self.backend_name,
            "aliases": list(self.aliases),
            "preamble_token_ids": list(self.preamble_token_ids),
            "filters": [spec.to_dict() for spec in self.filters],
            "joins": [spec.to_dict() for spec in self.joins],
        }

    def explain_fields(self) -> Mapping[str, Any]:
        return {
            "backend": self.backend_name,
            "aliases": list(self.aliases),
            "filter_chains": len(self.filters),
            "joins": len(self.joins),
        }

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            backend_name=str(attributes["backend_name"]),
            aliases=tuple(attributes["aliases"]),
            preamble_token_ids=tuple(attributes["preamble_token_ids"]),
            filters=tuple(
                RequestFilterSpec.from_mapping(spec)
                for spec in attributes["filters"]
            ),
            joins=tuple(
                RequestJoinSpec.from_mapping(spec)
                for spec in attributes["joins"]
            ),
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

    type_name: ClassVar[str] = "quail.packed_filter"
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
            ),
            OutputPort(
                f"filter_answers:{self.alias}",
                ValueType.FILTER_ANSWERS,
                schema=(self.alias, "predicate", "answer"),
            ),
        )

    def attributes(self) -> dict:
        return {
            "alias": self.alias,
            "arena_writes": self.arena_writes,
            "keep_kv": self.keep_kv,
            "keep_min_doc_tokens": self.keep_min_doc_tokens,
            "keep_resident_fraction": self.keep_resident_fraction,
            "stages": [stage.to_dict() for stage in self.stages],
            "question_token_ids": [
                list(question) for question in self.question_token_ids
            ],
        }

    def explain_fields(self) -> Mapping[str, Any]:
        value = self.attributes()
        value.pop("question_token_ids")
        return value

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
                for stage in attributes["stages"]
            ),
            question_token_ids=tuple(
                tuple(question)
                for question in attributes["question_token_ids"]
            ),
        )


@dataclass(frozen=True)
class Exchange(PhysicalNode):
    """Move or repartition survivor ids between join steps."""

    next_anchor: str = ""
    aliases: tuple[str, ...] = ()

    type_name: ClassVar[str] = "quail.exchange"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return tuple(
            OutputPort(
                f"ids:{alias}",
                ValueType.DOCUMENT_IDS,
                schema=(alias,),
            )
            for alias in self.aliases
        )

    def attributes(self) -> dict:
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

    type_name: ClassVar[str] = "quail.anchored_join"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.GPU_EXECUTOR
    backend: ClassVar[str] = "quail"

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        outputs = [OutputPort(
            f"ids:{self.anchor}",
            ValueType.DOCUMENT_IDS,
            schema=(self.anchor,),
        )]
        outputs.extend(OutputPort(
            f"join_answers:{stage.written_pos}",
            ValueType.JOIN_ANSWERS,
            schema=(stage.anchor, *stage.partners, "answer"),
        ) for stage in self.stages)
        return tuple(outputs)

    def attributes(self) -> dict:
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
                for stage in attributes["stages"]
            ),
        )


@dataclass(frozen=True)
class AdaptiveJoinPlan(PhysicalNode):
    """Choose and run anchored join steps after filters finish."""

    expected_nodes: tuple[AnchoredJoin | Exchange, ...] = ()
    aliases: tuple[str, ...] = ()
    join_positions: tuple[int, ...] = ()
    full_join_positions: tuple[int, ...] = ()
    join_specs: tuple[Mapping[str, Any], ...] = ()

    type_name: ClassVar[str] = "quail.adaptive_join_plan"
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
            )
            for alias in self.aliases
        ]
        outputs.extend(OutputPort(
            f"join_answers:{position}", ValueType.JOIN_ANSWERS
        ) for position in self.join_positions)
        return tuple(outputs)

    def attributes(self) -> dict:
        return {
            "aliases": list(self.aliases),
            "join_positions": list(self.join_positions),
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
                    "attributes": node.attributes(),
                }
                for node in self.expected_nodes
            ],
            "join_specs": list(self.join_specs),
        }

    def explain_fields(self) -> Mapping[str, Any]:
        return {
            "aliases": list(self.aliases),
            "join_positions": list(self.join_positions),
            "full_join_positions": list(self.full_join_positions),
            "expected_steps": [
                node.type_name for node in self.expected_nodes
            ],
        }

    def embedded_nodes(self) -> tuple[PhysicalNode, ...]:
        return self.expected_nodes

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        expected = []
        node_types = {
            AnchoredJoin.type_name: AnchoredJoin,
            Exchange.type_name: Exchange,
        }
        for encoded in attributes["expected_nodes"]:
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
                    schema=tuple(input_port["schema"]),
                )
                for input_port in encoded["inputs"]
            )
            expected.append(node_type.from_attributes(
                encoded["id"], child_inputs, encoded["attributes"]
            ))
        return cls(
            node_id=node_id,
            inputs=inputs,
            expected_nodes=tuple(expected),
            aliases=tuple(attributes["aliases"]),
            join_positions=tuple(attributes["join_positions"]),
            full_join_positions=tuple(
                attributes["full_join_positions"]
            ),
            join_specs=tuple(attributes["join_specs"]),
        )


@dataclass(frozen=True)
class HashJoin(PhysicalNode):
    """Combine answer relations by exact document ids."""

    alias_order: tuple[str, ...] = ()

    type_name: ClassVar[str] = "quail.hash_join"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort(
            "tuples",
            ValueType.ROWS,
            schema=self.alias_order,
        ),)

    def attributes(self) -> dict:
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

    type_name: ClassVar[str] = "quail.project"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.CLIENT

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort(
            "rows",
            ValueType.ROWS,
            schema=self.columns,
        ),)

    def attributes(self) -> dict:
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

    type_name: ClassVar[str] = "quail.limit"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.CLIENT

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (OutputPort("rows", ValueType.ROWS),)

    def attributes(self) -> dict:
        return {"count": self.count}

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(node_id=node_id, inputs=inputs, count=int(attributes["count"]))
