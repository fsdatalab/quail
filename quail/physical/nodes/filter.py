"""Filter predicates and the Filter and AiFilter nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from ..base import (
    ExecutionLocation,
    OutputPort,
    PhysicalNode,
    ValueType,
)


@dataclass(frozen=True)
class InList:
    """The predicate ``column IN (values)`` over a label column."""

    column: str
    values: tuple[str, ...]

    def accepts(self, value) -> bool:
        """Return whether the value is one of the accepted labels."""
        return value in self.values

    def to_dict(self) -> dict:
        return {"kind": "in_list", "column": self.column,
                "values": list(self.values)}

    def describe(self) -> str:
        return f"{self.column} IN {list(self.values)}"


@dataclass(frozen=True)
class Comparison:
    """The predicate ``column <op> value`` over a score column."""

    column: str
    op: str
    value: float

    def accepts(self, value) -> bool:
        """Return whether the value satisfies the score comparison."""
        import operator

        compare = {"<": operator.lt, "<=": operator.le,
                   ">": operator.gt, ">=": operator.ge}[self.op]
        return bool(compare(value, self.value))

    def to_dict(self) -> dict:
        return {"kind": "comparison", "column": self.column, "op": self.op,
                "value": self.value}

    def describe(self) -> str:
        return f"{self.column} {self.op} {self.value}"


def predicate_from_mapping(value: Mapping[str, Any]) -> "InList | Comparison":
    """Decode a filter predicate from its serialized mapping."""
    if value["kind"] == "in_list":
        return InList(str(value["column"]),
                      tuple(str(item) for item in value["values"]))
    if value["kind"] == "comparison":
        return Comparison(str(value["column"]), str(value["op"]),
                          float(value["value"]))
    raise ValueError(f"unknown filter predicate {value['kind']!r}")


@dataclass(frozen=True)
class Filter(PhysicalNode):
    """Physical operator that keeps rows matching a score or label condition.

    InList tests a label produced by AI.CLASSIFY. Comparison tests a score
    produced by AI.SCORE. Single-document filters also return the kept
    document IDs; filters over joined pairs return join answers.
    """

    predicate: InList | Comparison | None = None
    aliases: tuple[str, ...] = ()
    selectivity: float | None = None
    written_pos: int = 0

    type_name: ClassVar[str] = "quail.filter"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.GPU_EXECUTOR

    @property
    def document_aliases(self) -> tuple[str, ...]:
        return self.aliases if len(self.aliases) == 1 else ()

    def document_outputs(self, alias: str) -> tuple[str, ...]:
        return ("rows", f"ids:{alias}")

    @property
    def column(self) -> str:
        """Return the column tested by the predicate."""
        return self.predicate.column

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        if len(self.aliases) == 1:
            (alias,) = self.aliases
            return (
                OutputPort("rows", ValueType.ROWS),
                OutputPort(f"filter_answers:{alias}", ValueType.FILTER_ANSWERS),
                OutputPort(f"ids:{alias}", ValueType.DOCUMENT_IDS,
                           schema=(alias,)),
            )
        return (
            OutputPort("rows", ValueType.ROWS),
            OutputPort(f"join_answers:{self.written_pos}",
                       ValueType.JOIN_ANSWERS),
        )

    def attributes(self) -> dict:
        return {
            "predicate": self.predicate.to_dict(),
            "aliases": list(self.aliases),
            "selectivity": self.selectivity,
            "written_pos": self.written_pos,
        }

    def explain_fields(self) -> Mapping[str, Any]:
        return {
            "predicate": self.predicate.describe(),
            "selectivity": self.selectivity,
            "aliases": list(self.aliases),
        }

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            predicate=predicate_from_mapping(attributes["predicate"]),
            aliases=tuple(attributes["aliases"]),
            selectivity=attributes["selectivity"],
            written_pos=int(attributes["written_pos"]),
        )


@dataclass(frozen=True)
class AiFilter(PhysicalNode):
    """Evaluate one predicate on a document input."""

    alias: str = ""
    arena_writes: bool = False
    keep_kv: bool = False
    written_pos: int = 0
    question_tokens: int = 0
    preamble_tokens: int = 0
    selectivity: float | None = None
    expected_docs: float = 0.0
    question_token_ids: tuple[Any, ...] = ()
    # documents borrow the KV pages of a document that shares their
    # token prefix (the prefix_sharing rule)
    share_prefixes: bool = False
    # "unified" or "tree": the attention path the tree_attention rule
    # chose; empty leaves the executor's default
    attention: str = ""
    # value columns of the alias; the executor reads one document of
    # each key value at a time and skips the rest once one survives
    stop_key: tuple[str, ...] = ()

    type_name: ClassVar[str] = "quail.ai_filter"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.GPU_EXECUTOR
    backend: ClassVar[str] = "quail"

    worker_pipelining: ClassVar[bool] = True

    @property
    def document_aliases(self) -> tuple[str, ...]:
        return (self.alias,)

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
            "written_pos": self.written_pos,
            "question_tokens": self.question_tokens,
            "preamble_tokens": self.preamble_tokens,
            "selectivity": self.selectivity,
            "expected_docs": self.expected_docs,
            "question_token_ids": list(self.question_token_ids),
            "share_prefixes": self.share_prefixes,
            "attention": self.attention,
            "stop_key": list(self.stop_key),
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
            written_pos=int(attributes["written_pos"]),
            question_tokens=int(attributes["question_tokens"]),
            preamble_tokens=int(attributes["preamble_tokens"]),
            selectivity=attributes["selectivity"],
            expected_docs=float(attributes["expected_docs"]),
            question_token_ids=tuple(attributes["question_token_ids"]),
            share_prefixes=bool(attributes.get("share_prefixes", False)),
            attention=str(attributes.get("attention", "")),
            stop_key=tuple(str(column)
                           for column in attributes.get("stop_key", ())),
        )
