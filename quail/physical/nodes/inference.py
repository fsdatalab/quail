"""Inference specs and the AiInference and AiScore nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from ..base import (
    AnswerOutput,
    ExecutionLocation,
    OutputPort,
    PhysicalNode,
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
    prompt_token_parts: tuple[tuple[int, ...], ...] = ()
    share_prefixes: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "InferenceSpec":
        return cls(
            name=str(value["name"]),
            aliases=tuple(value["aliases"]),
            query_template=str(value["query_template"]),
            arguments=tuple(
                (str(alias), str(column))
                for alias, column in value["arguments"]
            ),
            expected_rows=float(value["expected_rows"]),
            estimated_seconds=float(value["estimated_seconds"]),
            prompt_token_parts=tuple(
                tuple(part) for part in value.get("prompt_token_parts", ())
            ),
            share_prefixes=bool(value.get("share_prefixes", False)),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "query_template": self.query_template,
            "arguments": [list(argument) for argument in self.arguments],
            "expected_rows": self.expected_rows,
            "estimated_seconds": self.estimated_seconds,
            "prompt_token_parts": [list(part) for part in self.prompt_token_parts],
            "share_prefixes": self.share_prefixes,
        }

    @property
    def output_columns(self) -> tuple[str, ...]:
        """Return the document IDs and computed columns."""
        return (*self.aliases, self.name)


@dataclass(frozen=True)
class ScoreSpec(InferenceSpec):
    """Settings for scoring documents or pairs, including diffusion draws."""

    pair_fraction: float = 1.0
    draws: int = 1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ScoreSpec":
        return cls(
            **vars(InferenceSpec.from_mapping(value)),
            pair_fraction=float(value.get("pair_fraction", 1.0)),
            draws=int(value.get("draws", 1)),
        )

    def to_dict(self) -> dict:
        return {**super().to_dict(), "pair_fraction": self.pair_fraction,
                "draws": self.draws}


@dataclass(frozen=True)
class AiInference(PhysicalNode):
    """Append model outputs to candidate rows."""

    backend_name: str = ""
    model: str = ""
    spec: InferenceSpec | None = None
    keep_kv: bool = False

    answer_kind: ClassVar[str] = "scores"
    spec_type: ClassVar[type[InferenceSpec]] = InferenceSpec
    document_ids: ClassVar[bool] = False
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
        ports = (OutputPort("rows", ValueType.ROWS,
                            schema=self.spec.output_columns,
                            answer=AnswerOutput(self.answer_kind, self.spec.name,
                                                columns)),)
        if self.document_ids and len(self.spec.aliases) == 1:
            (alias,) = self.spec.aliases
            ports += (OutputPort(f"ids:{alias}", ValueType.DOCUMENT_IDS,
                                 schema=(alias,)),)
        return ports

    def attributes(self) -> dict:
        return {
            "backend_name": self.backend_name,
            "keep_kv": self.keep_kv,
            "model": self.model,
            "spec": None if self.spec is None else self.spec.to_dict(),
        }

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

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        value = attributes["spec"]
        return cls(
            node_id=node_id,
            inputs=inputs,
            backend_name=str(attributes["backend_name"]),
            keep_kv=bool(attributes.get("keep_kv", False)),
            model=str(attributes["model"]),
            spec=None if value is None else cls.spec_type.from_mapping(value),
        )


@dataclass(frozen=True)
class AiScore(AiInference):
    """Append one FLOAT64 reranker score to candidate rows."""

    spec: ScoreSpec | None = None
    spec_type: ClassVar[type[InferenceSpec]] = ScoreSpec
    type_name: ClassVar[str] = "quail.ai_score"
    runtime_key: ClassVar[str] = type_name
