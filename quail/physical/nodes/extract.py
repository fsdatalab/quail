"""The ExtractSpec and AiExtract node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from ..base import PhysicalNode, validate_integer_setting
from .classify import AiClassify
from .filter import AiFilter
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
        draft_policy: "fixed" feeds draft_tokens every round; "adaptive"
            picks each round's draft length from the round's KV read
            and the answer-length curve, and feeds none while several
            document positions fit the answer.
        expected_answer_tokens: The planner's guess at the mean answer
            length, the seed of the adaptive policy's answer-length curve.
        round_cost_per_token: One document token's KV read per round,
            in fed tokens of compute; the planner sets it from the
            model and device.
        top_tokens: Vocabulary candidates returned by each GPU readout.
        first_window_chars: Source characters tokenized in the first window.
        window_chars: Source characters tokenized in later windows.
        shares_head: Whether the prompt's head is the query's shared
            preamble, so the extraction reads the document KV a filter
            wrote and continues its pipeline.
    """

    question: str = ""
    shares_head: bool = False
    max_answer_tokens: int = 64
    draft_tokens: int = 16
    draft_policy: str = "fixed"
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

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExtractSpec":
        return cls(
            **vars(InferenceSpec.from_mapping(value)),
            question=str(value["question"]),
            shares_head=bool(value.get("shares_head", False)),
            **{name: value.get(name, getattr(cls, name))
               for name in ("max_answer_tokens", "draft_tokens", "draft_policy",
                            "expected_answer_tokens", "round_cost_per_token",
                            "top_tokens", "first_window_chars", "window_chars")},
        )

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "question": self.question,
            "shares_head": self.shares_head,
            "max_answer_tokens": self.max_answer_tokens,
            "draft_tokens": self.draft_tokens,
            "draft_policy": self.draft_policy,
            "expected_answer_tokens": self.expected_answer_tokens,
            "round_cost_per_token": self.round_cost_per_token,
            "top_tokens": self.top_tokens,
            "first_window_chars": self.first_window_chars,
            "window_chars": self.window_chars,
        }


@dataclass(frozen=True)
class AiExtract(AiInference):
    """Physical operator that appends each row's answer and its span.

    The rows port carries one row per input document: its id, the
    answer text, and the answer's (start, end) character span in the
    document; both are null when the document does not answer the
    question. The ids port carries the same documents.
    """

    spec: ExtractSpec | None = None

    answer_kind: ClassVar[str] = "extracts"
    spec_type: ClassVar[type[InferenceSpec]] = ExtractSpec
    document_ids: ClassVar[bool] = True
    type_name: ClassVar[str] = "quail.ai_extract"
    runtime_key: ClassVar[str] = type_name

    worker_pipelining: ClassVar[bool] = True

    @property
    def document_head(self) -> tuple[int, ...] | None:
        return (None if self.spec and self.spec.shares_head
                else super().document_head)

    @property
    def document_aliases(self) -> tuple[str, ...]:
        return () if self.spec is None else (self.spec.alias,)

    def document_outputs(self, alias: str) -> tuple[str, ...]:
        return ("rows", f"ids:{alias}")

    def can_follow(self, previous: PhysicalNode, first: PhysicalNode) -> bool:
        if isinstance(previous, AiExtract):
            return (previous.spec.same_decoding(self.spec)
                    and previous.document_head == self.document_head)
        if not self.spec.shares_head:
            return False
        if isinstance(first, AiClassify):
            return (first.spec.partner is None
                    and first.spec.prompt_token_parts[0]
                    == self.spec.prompt_token_parts[0])
        return isinstance(first, AiFilter)

    def can_precede(self, consumer: PhysicalNode) -> bool:
        return isinstance(consumer, AiExtract)

    def explain_fields(self) -> Mapping[str, Any]:
        return {
            **super().explain_fields(),
            "question": None if self.spec is None else self.spec.question,
        }
