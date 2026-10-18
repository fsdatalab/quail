"""Inference specs and the AiInference and AiScore nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from ..base import (
    AnswerOutput,
    ExecutionLocation,
    OutputPort,
    PhysicalNode,
    Token,
    ValueType,
)


@dataclass(frozen=True)
class InferenceSpec:
    """Bound inputs, output columns, and cost of one model call."""

    name: str
    aliases: tuple[str, ...]
    query_template: str
    arguments: tuple[tuple[str, str], ...]
    expected_rows: float
    estimated_seconds: float
    prompt_token_parts: tuple[tuple[Token, ...], ...] = ()
    share_prefixes: bool = False

    @property
    def output_columns(self) -> tuple[str, ...]:
        """Return the document IDs and computed columns."""
        return (*self.aliases, self.name)


@dataclass(frozen=True)
class ScoreSpec(InferenceSpec):
    """Settings for scoring documents or pairs, including diffusion draws."""

    pair_fraction: float = 1.0
    draws: int = 1

@dataclass(frozen=True)
class AiInference(PhysicalNode):
    """Append model outputs to candidate rows."""

    backend_name: str = ""
    model: str = ""
    spec: InferenceSpec | None = None
    keep_kv: bool = False

    answer_kind: ClassVar[str] = "scores"
    spec_type: ClassVar[type[InferenceSpec]] = InferenceSpec
    location: ClassVar[ExecutionLocation] = ExecutionLocation.GPU_EXECUTOR

    @property
    def document_head(self) -> tuple[int, ...]:
        return (tuple(self.spec.prompt_token_parts[0])
                if self.spec and self.spec.prompt_token_parts else ())

    @property
    def backend(self) -> str:
        """Return the backend selected for this model node."""
        return self.backend_name

    @property
    def outputs(self) -> tuple[OutputPort, ...]:
        if self.spec is None:
            return ()
        columns = (self.spec.output_columns if self.answer_kind == "extracts"
                   else (*self.spec.aliases, self.spec.name))
        return (OutputPort(self.rows_port, ValueType.ROWS,
                           schema=self.spec.output_columns,
                           answer=AnswerOutput(self.answer_kind, self.spec.name,
                                               columns)),)

    @property
    def rows_port(self) -> str:
        """The output carrying the rows: per alias for one table, rows for pairs."""
        aliases = () if self.spec is None else self.spec.aliases
        return f"rows:{aliases[0]}" if len(aliases) == 1 else "rows"

    def explain_fields(self) -> Mapping[str, Any]:
        spec = self.spec
        return {
            "backend": self.backend_name,
            "keep_kv": self.keep_kv,
            "model": self.model,
            "batching": "token_based_admission",
            "output": None if spec is None else spec.name,
            "aliases": [] if spec is None else list(spec.aliases),
            "expected_rows": 0 if spec is None else spec.expected_rows,
            "estimated_seconds": (
                0 if spec is None else spec.estimated_seconds
            ),
            "share_prefixes": False if spec is None else spec.share_prefixes,
        }

@dataclass(frozen=True)
class AiScore(AiInference):
    """Append one FLOAT64 reranker score to candidate rows."""

    spec: ScoreSpec | None = None
    spec_type: ClassVar[type[InferenceSpec]] = ScoreSpec
    type_name: ClassVar[str] = "quail.ai_score"
    runtime_key: ClassVar[str] = type_name
