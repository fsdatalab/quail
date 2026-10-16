"""The Filter condition and the Filter and AiFilter nodes."""

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
class Condition:
    """The predicate ``column <comparison> value`` over a result column.

    The fields are those of quail.logical.RegularPredicate with the
    column named as the rows table names it; the executor evaluates
    both with quail.catalog.selection.evaluate_predicate.
    """

    column: str
    comparison: str
    value: Any = None
    path: tuple[str, ...] = ()
    function: str | None = None

    def to_dict(self) -> dict:
        return {"column": self.column, "comparison": self.comparison,
                "value": list(self.value) if isinstance(self.value, tuple)
                else self.value, "path": list(self.path),
                "function": self.function}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Condition":
        """Decode a condition from its serialized mapping."""
        literal = value["value"]
        return cls(str(value["column"]), str(value["comparison"]),
                   tuple(literal) if isinstance(literal, list) else literal,
                   tuple(str(field) for field in value.get("path", ())),
                   value.get("function"))

    @classmethod
    def from_predicate(cls, column: str, predicate) -> "Condition":
        """Build the condition a logical RegularPredicate applies."""
        return cls(column, predicate.comparison, predicate.value,
                   tuple(predicate.path), predicate.function)

    def describe(self) -> str:
        name = ".".join((self.column, *self.path))
        name = f"{self.function}({name})" if self.function else name
        if self.comparison == "in":
            return f"{name} IN {list(self.value)}"
        if self.comparison.startswith("is"):
            return f"{name} {self.comparison.upper()}"
        return f"{name} {self.comparison} {self.value!r}"


@dataclass(frozen=True)
class Filter(PhysicalNode):
    """Physical operator that keeps the rows whose condition holds.

    The condition tests a label AI.CLASSIFY produced, a score AI.SCORE
    produced, or an answer or span AI.EXTRACT produced. Single-document
    filters also return the kept document IDs; filters over joined
    pairs return join answers.
    """

    predicate: Condition | None = None
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

    def can_follow(self, previous: PhysicalNode, first: PhysicalNode) -> bool:
        # a condition reads no KV: it gates the documents of any pipeline
        return True

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
            predicate=Condition.from_dict(attributes["predicate"]),
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
    # token prefix (the should_share_prefixes rule)
    share_prefixes: bool = False
    # "unified" or "tree": the attention path the choose_attention_algorithm rule
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
