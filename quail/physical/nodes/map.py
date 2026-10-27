"""The MapSpec and AiMap node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from ..base import PhysicalNode, validate_integer_setting
from .inference import AiInference, InferenceSpec


@dataclass(frozen=True)
class MapSpec(InferenceSpec):
    """Specification for writing each document's answer to an instruction.

    The prompt token parts are the head before the document and the
    tail after it, ending in the answer cue. The answer is the model's
    greedy decoding, up to its turn end or max_answer_tokens.

    Attributes:
        max_answer_tokens: The most tokens an answer may have.
        expected_answer_tokens: The planner's guess at the mean answer
            length, used only for the cost estimate.
    """

    max_answer_tokens: int = 256
    expected_answer_tokens: int = 64

    def __post_init__(self):
        for name in ("max_answer_tokens", "expected_answer_tokens"):
            validate_integer_setting(name, getattr(self, name), minimum=1)


@dataclass(frozen=True)
class AiMap(AiInference):
    """Physical operator that appends each row's written answer.

    The rows port carries one row per input document: its id and the
    answer the model wrote.
    """

    spec: MapSpec | None = None

    answer_kind: ClassVar[str] = "maps"
    spec_type: ClassVar[type[InferenceSpec]] = MapSpec
    type_name: ClassVar[str] = "quail.ai_map"
    runtime_key: ClassVar[str] = type_name

    @property
    def document_aliases(self) -> tuple[str, ...]:
        return () if self.spec is None else (self.spec.aliases[0],)

    def can_follow(self, previous: PhysicalNode, first: PhysicalNode) -> bool:
        # a map continues any pipeline that shows its documents under
        # the same head, and reads the KV the operators before it wrote
        return self.document_head == first.document_head

    def can_precede(self, consumer: PhysicalNode) -> bool:
        # the next operator rewinds the KV past the answer
        return True

    def explain_fields(self) -> Mapping[str, Any]:
        return {
            **super().explain_fields(),
            "max_answer_tokens": (None if self.spec is None
                                  else self.spec.max_answer_tokens),
        }
