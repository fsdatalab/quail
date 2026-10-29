"""The ExtractSpec and AiExtract node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from ..base import PhysicalNode, validate_integer_setting
from .inference import AiInference, InferenceSpec


@dataclass(frozen=True)
class ExtractSpec(InferenceSpec):
    """Specification for copying each document's answer to a question.

    The prompt token parts are the head before the document and the
    tail after it, ending in the phrase cue.

    Attributes:
        question: The question the answer is copied for.
        max_answer_tokens: Maximum tokens in an extracted answer.
        draft_tokens: Source tokens to verify after each chosen token;
            the most per round under the adaptive policy.
        draft_policy: "adaptive" picks each round's draft length from
            the round's KV read, the answer-length curve, and the odds
            that the position followed is the answer's source; "fixed"
            feeds draft_tokens every round.
        expected_answer_tokens: The planner's guess at the mean answer
            length, the seed of the adaptive policy's answer-length curve.
        round_cost_per_token: One document token's KV read per round,
            in fed tokens of compute; the planner sets it from the
            model and device.
        top_tokens: Vocabulary candidates returned by each GPU readout.
        first_window_chars: Source characters tokenized in the first window.
        window_chars: Source characters tokenized in later windows.
    """

    question: str = ""
    max_answer_tokens: int = 64
    draft_tokens: int = 63
    draft_policy: str = "adaptive"
    expected_answer_tokens: int = 16
    round_cost_per_token: float = 0.0
    top_tokens: int = 1024
    first_window_chars: int = 64
    window_chars: int = 512

    def __post_init__(self):
        for name in ("max_answer_tokens", "draft_tokens", "expected_answer_tokens",
                     "top_tokens", "first_window_chars", "window_chars"):
            validate_integer_setting(name, getattr(self, name),
                                      minimum=0 if name == "draft_tokens" else 1)
        if self.draft_policy not in ("fixed", "adaptive"):
            raise ValueError("draft_policy is 'fixed' or 'adaptive'")
        if not isinstance(self.round_cost_per_token, (int, float)) or (
                self.round_cost_per_token < 0):
            raise ValueError("round_cost_per_token is a nonnegative number")

    def same_decoding(self, other: "ExtractSpec") -> bool:
        """Return whether two extractions can use the same decoder."""
        return all(getattr(self, name) == getattr(other, name) for name in (
            "max_answer_tokens", "draft_tokens", "draft_policy",
            "expected_answer_tokens", "round_cost_per_token", "top_tokens",
            "first_window_chars", "window_chars"))

    @property
    def alias(self) -> str:
        """Return the table alias the answer columns belong to."""
        return self.aliases[0]

    @property
    def span_name(self) -> str:
        """Return the name of the span column."""
        return self.name + "_span"

    @property
    def output_columns(self) -> tuple[str, ...]:
        """Return the document ID, answer, and source span columns."""
        return (*super().output_columns, self.span_name)

@dataclass(frozen=True)
class AiExtract(AiInference):
    """Physical operator that appends each row's answer and its span.

    The rows port carries one row per input document: its id, the
    answer text, and the answer's (start, end) character span in the
    document; both are null when the document does not answer the
    question. The ids port carries the same documents.
    """

    spec: ExtractSpec | None = None
    # "unified" or "tree": the attention path the choose_attention_algorithm
    # rule chose for the document's answer branches (its fields and maps)
    attention: str = ""

    answer_kind: ClassVar[str] = "extracts"
    spec_type: ClassVar[type[InferenceSpec]] = ExtractSpec
    type_name: ClassVar[str] = "quail.ai_extract"
    runtime_key: ClassVar[str] = type_name


    @property
    def document_aliases(self) -> tuple[str, ...]:
        return () if self.spec is None else (self.spec.alias,)

    def can_follow(self, previous: PhysicalNode, first: PhysicalNode) -> bool:
        # an extraction continues any pipeline that shows the documents
        # under its head; consecutive extractions share one decoder
        if isinstance(previous, AiExtract) and not previous.spec.same_decoding(
                self.spec):
            return False
        return first.document_head == self.document_head

    def can_precede(self, consumer: PhysicalNode) -> bool:
        # the next operator rewinds the KV past the answer; a Filter on
        # the answers gates the documents it reads
        return True

    def explain_fields(self) -> Mapping[str, Any]:
        return {
            **super().explain_fields(),
            "question": None if self.spec is None else self.spec.question,
        }
