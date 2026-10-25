"""The SaeFeaturesSpec and AiSaeFeatures node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from ..base import PhysicalNode
from .inference import AiInference, InferenceSpec

SAE_POOLINGS = ("last", "max")


@dataclass(frozen=True)
class SaeFeaturesSpec(InferenceSpec):
    """Specification for reading each document prompt's SAE features.

    The prompt token parts are the head before the document and the
    question tail after it. The call generates no text: the engine runs
    prefill, encodes the pooling's tokens with the model's sparse
    autoencoder, and returns the nonzero features as a
    ``list<struct<feature: int32, strength: float32>>`` column.

    Attributes:
        pooling: Which tokens become a row's list. ``last`` encodes the
            final prompt token. ``max`` takes each feature's largest
            strength over every prompt token, with the document tokens'
            maxima kept across the prompts that share the document head.
    """

    pooling: str = "last"

    def __post_init__(self):
        if self.pooling not in SAE_POOLINGS:
            raise ValueError(
                f"pooling is one of {SAE_POOLINGS}, got {self.pooling!r}")


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
            "pooling": None if self.spec is None else self.spec.pooling,
        }
