"""The Filter and AiFilter nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from quail.logical import RegularPredicate

from ..base import (
    ExecutionLocation,
    OutputPort,
    PhysicalNode,
    Token,
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
    written_position: int = 0

    type_name: ClassVar[str] = "quail.filter"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.GPU_EXECUTOR

    @property
    def document_aliases(self) -> tuple[str, ...]:
        return self.aliases if len(self.aliases) == 1 else ()

    @property
    def rows_port(self) -> str:
        """The output carrying the rows that pass: per alias for one table."""
        return f"rows:{self.aliases[0]}" if len(self.aliases) == 1 else "rows"

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
                OutputPort(f"rows:{alias}", ValueType.ROWS),
                OutputPort(f"filter_answers:{alias}", ValueType.FILTER_ANSWERS),
            )
        return (
            OutputPort("rows", ValueType.ROWS),
            OutputPort(f"join_answers:{self.written_position}",
                       ValueType.JOIN_ANSWERS),
        )

    def explain_fields(self) -> Mapping[str, Any]:
        return {
            "predicate": str(self.predicate),
            "selectivity": self.selectivity,
            "aliases": list(self.aliases),
        }

@dataclass(frozen=True)
class AiFilter(PhysicalNode):
    """Evaluate one predicate on a document input."""

    alias: str = ""
    arena_writes: bool = False
    keep_kv: bool = False
    written_position: int = 0
    question_tokens: int = 0
    preamble_tokens: int = 0
    selectivity: float | None = None
    expected_rows: float = 0.0
    question_token_ids: tuple[Token, ...] = ()
    # the token ids the documents are shown under, before the document
    document_head: tuple[Token, ...] = ()
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


    @property
    def document_aliases(self) -> tuple[str, ...]:
        return (self.alias,)

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        return (
            OutputPort(
                f"rows:{self.alias}",
                ValueType.ROWS,
                schema=(self.alias,),
            ),
            OutputPort(
                f"filter_answers:{self.alias}",
                ValueType.FILTER_ANSWERS,
                schema=(self.alias, "predicate", "answer"),
            ),
        )

    def explain_fields(self) -> Mapping[str, Any]:
        return {name: value for name, value in super().explain_fields().items()
                if name not in ("question_token_ids", "document_head")}
