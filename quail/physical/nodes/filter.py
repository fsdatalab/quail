"""The Filter and AiFilter nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from quail.logical import RegularPredicate

from ..base import (
    ExecutionLocation,
    OutputPort,
    PhysicalNode,
    ValueType,
)


@dataclass(frozen=True)
class Filter(PhysicalNode):
    """Physical operator that keeps the rows whose predicate holds.

    The predicate is the logical plan's RegularPredicate over a label
    AI.CLASSIFY produced, a score AI.SCORE produced, or an answer or
    span AI.EXTRACT produced; its column names the rows table's
    column. Single-document filters also return the kept document IDs;
    filters over joined pairs return join answers.
    """

    predicate: RegularPredicate | None = None
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
        return self.predicate.column.column

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
            "predicate": str(self.predicate),
            "selectivity": self.selectivity,
            "aliases": list(self.aliases),
        }

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(
            node_id=node_id,
            inputs=inputs,
            predicate=RegularPredicate.from_dict(attributes["predicate"]),
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
    # the token ids the documents are shown under, before the document
    document_head: tuple[int, ...] = ()
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
            "document_head": list(self.document_head),
            "share_prefixes": self.share_prefixes,
            "attention": self.attention,
            "stop_key": list(self.stop_key),
        }

    def explain_fields(self) -> Mapping[str, Any]:
        value = self.attributes()
        value.pop("question_token_ids")
        value.pop("document_head")
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
            document_head=tuple(attributes.get("document_head", ())),
            share_prefixes=bool(attributes.get("share_prefixes", False)),
            attention=str(attributes.get("attention", "")),
            stop_key=tuple(str(column)
                           for column in attributes.get("stop_key", ())),
        )
