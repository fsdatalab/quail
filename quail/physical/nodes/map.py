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
    tail after it, ending in the answer cue.

    Attributes:
        instruction: The instruction the answer is written for.
        max_answer_tokens: Maximum tokens in a written answer.
        draft_tokens: Tokens to verify after each chosen token; the
            most per round under the adaptive policy. 0 feeds the
            chosen token alone.
        draft_policy: "adaptive" sizes each round's draft from its
            acceptance odds, the round's KV read, and the pass's
            slack; "fixed" copies up to draft_tokens from the document.
        expected_answer_tokens: The planner's guess at the mean answer
            length.
        round_cost_per_token: One document token's KV read per round,
            in fed tokens of compute; the planner sets it from the
            model and device.
        ridge_tokens: Fed tokens per forward pass at which the weight
            work turns compute bound; the planner sets it from the
            model and device. Below it a marginal draft token is close
            to free.
        top_tokens: Vocabulary candidates returned by each GPU readout.
            Greedy decoding needs the likeliest token alone, so each
            row carries three values: the stop, the token, and its log
            probability.
    """

    instruction: str = ""
    max_answer_tokens: int = 256
    draft_tokens: int = 15
    draft_policy: str = "adaptive"
    expected_answer_tokens: int = 64
    round_cost_per_token: float = 0.0
    ridge_tokens: float = 0.0
    top_tokens: int = 1

    def __post_init__(self):
        for name in ("max_answer_tokens", "expected_answer_tokens", "top_tokens"):
            validate_integer_setting(name, getattr(self, name), minimum=1)
        validate_integer_setting("draft_tokens", self.draft_tokens, minimum=0)
        if self.draft_policy not in ("fixed", "adaptive"):
            raise ValueError("draft_policy is 'fixed' or 'adaptive'")
        for name in ("round_cost_per_token", "ridge_tokens"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} is a nonnegative number")

    def same_decoding(self, other: "MapSpec") -> bool:
        """Return whether two maps can use the same decoder."""
        return all(getattr(self, name) == getattr(other, name) for name in (
            "max_answer_tokens", "draft_tokens", "draft_policy",
            "expected_answer_tokens", "round_cost_per_token", "ridge_tokens",
            "top_tokens"))

    @property
    def alias(self) -> str:
        """Return the table alias the answer column belongs to."""
        return self.aliases[0]


@dataclass(frozen=True)
class AiMap(AiInference):
    """Physical operator that appends each row's written answer.

    The rows port carries one row per input document: its id and the
    answer text the model wrote for the instruction.
    """

    spec: MapSpec | None = None

    answer_kind: ClassVar[str] = "maps"
    spec_type: ClassVar[type[InferenceSpec]] = MapSpec
    type_name: ClassVar[str] = "quail.ai_map"
    runtime_key: ClassVar[str] = type_name

    @property
    def document_aliases(self) -> tuple[str, ...]:
        return () if self.spec is None else (self.spec.alias,)

    def can_follow(self, previous: PhysicalNode, first: PhysicalNode) -> bool:
        # a map continues any pipeline that shows the documents under
        # its head; consecutive maps share one decoder
        if isinstance(previous, AiMap) and not previous.spec.same_decoding(
                self.spec):
            return False
        return first.document_head == self.document_head

    def can_precede(self, consumer: PhysicalNode) -> bool:
        # the next operator rewinds the KV past the answer
        return True

    def explain_fields(self) -> Mapping[str, Any]:
        return {
            **super().explain_fields(),
            "instruction": None if self.spec is None else self.spec.instruction,
        }
