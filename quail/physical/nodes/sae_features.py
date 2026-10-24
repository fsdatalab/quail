"""The SaeFeaturesSpec and AiSaeFeatures node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from ..base import PhysicalNode
from .inference import AiInference, InferenceSpec

SAE_READINGS = ("last", "max")


@dataclass(frozen=True)
class SaeFeaturesSpec(InferenceSpec):
    """Specification for reading each document prompt's SAE features.

    The prompt token parts are the head before the document and the
    question tail after it. The call generates no text: the engine runs
    prefill, encodes the reading's tokens with the model's sparse
    autoencoder, and returns the nonzero features as a
    ``list<struct<feature: int32, strength: float32>>`` column.

    Attributes:
        reading: Which tokens become a row's list. ``last`` encodes the
            final prompt token. ``max`` takes each feature's largest
            strength over every prompt token, with the document tokens'
            maxima kept across the prompts that share the document head.
    """

    reading: str = "last"

    def __post_init__(self):
        if self.reading not in SAE_READINGS:
            raise ValueError(
                f"reading is one of {SAE_READINGS}, got {self.reading!r}")


@dataclass(frozen=True)
class AiSaeFeatures(AiInference):
    """Physical operator that appends a feature list to each input row."""

    spec: SaeFeaturesSpec | None = None

    answer_kind: ClassVar[str] = "features"
    spec_type: ClassVar[type[InferenceSpec]] = SaeFeaturesSpec
    type_name: ClassVar[str] = "quail.ai_sae_features"
    runtime_key: ClassVar[str] = type_name

    @property
    def document_aliases(self) -> tuple[str, ...]:
        return () if self.spec is None else (self.spec.aliases[0],)

    def can_follow(self, previous: PhysicalNode, first: PhysicalNode) -> bool:
        # a feature call continues any pipeline that shows its
        # documents under the same head
        return self.document_head == first.document_head

    def explain_fields(self) -> Mapping[str, Any]:
        return {
            **super().explain_fields(),
            "reading": None if self.spec is None else self.spec.reading,
        }
