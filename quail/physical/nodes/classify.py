"""The ClassifySpec and AiClassify node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from quail.logical import PROBABILITIES_SUFFIX

from ..base import PhysicalNode
from .inference import AiInference, InferenceSpec
from .join import AiJoin


@dataclass(frozen=True)
class ClassifySpec(InferenceSpec):
    """Specification for classifying individual documents or joined pairs.

    Attributes:
        labels: Labels in query order.
        label_token_ids: Token ids for each label: its letter for the
            letters rule, its text for the trie rules.
        label_word_starts: Per label, which of its tokens begin a new
            word, from ``quail.labels.word_starts``; empty marks none.
            The trie_tree rule reads a label through the end of the
            word that tells it apart.
        scoring: The label scoring rule: letters, trie_tree, or trie_decode.
            The planner leaves it empty until the choose_classify_implementation rule
            picks one.
        probabilities: Whether to add a name + "_probabilities" map column.
        join_layout: Anchor note and partner label token sequences for pair
            classification, or None for individual documents. The question
            follows the partner document.
        frame_tokens: For the decision_choice rule, the tokens of the
            prompt tail before the first option block; the option
            blocks and the closing line follow.
    """

    pair_fraction: float = 1.0
    draws: int = 1
    labels: tuple[str, ...] = ()
    label_token_ids: tuple[tuple[int, ...], ...] = ()
    label_word_starts: tuple[tuple[bool, ...], ...] = ()
    scoring: str = "letters"
    probabilities: bool = False
    join_layout: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    frame_tokens: int = 0

    @property
    def output_columns(self) -> tuple[str, ...]:
        """Return document IDs, the label, and requested label probabilities."""
        columns = super().output_columns
        return (columns + (self.name + PROBABILITIES_SUFFIX,)
                if self.probabilities else columns)

    @property
    def anchor(self) -> str:
        """Return the anchor table alias."""
        return self.aliases[0]

    @property
    def partner(self) -> str | None:
        """Return the partner table alias, or None for individual documents."""
        return self.aliases[1] if len(self.aliases) == 2 else None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ClassifySpec":
        return cls(
            **vars(InferenceSpec.from_mapping(value)),
            pair_fraction=float(value.get("pair_fraction", 1.0)),
            draws=int(value.get("draws", 1)),
            labels=tuple(str(label) for label in value["labels"]),
            label_token_ids=tuple(
                tuple(int(token) for token in ids)
                for ids in value["label_token_ids"]),
            label_word_starts=tuple(
                tuple(bool(flag) for flag in flags)
                for flags in value.get("label_word_starts", ())),
            scoring=str(value.get("scoring", "letters")),
            probabilities=bool(value.get("probabilities", False)),
            join_layout=(None if value.get("join_layout") is None else tuple(
                tuple(int(token) for token in part)
                for part in value["join_layout"])),
            frame_tokens=int(value.get("frame_tokens", 0)),
        )

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "pair_fraction": self.pair_fraction,
            "draws": self.draws,
            "labels": list(self.labels),
            "label_token_ids": [list(ids) for ids in self.label_token_ids],
            "label_word_starts": [list(flags) for flags in self.label_word_starts],
            "scoring": self.scoring,
            "probabilities": self.probabilities,
            "join_layout": (None if self.join_layout is None
                            else [list(part) for part in self.join_layout]),
            "frame_tokens": self.frame_tokens,
        }


@dataclass(frozen=True)
class AiClassify(AiInference):
    """Physical operator that appends a label to each input row.

    The spec names the label scoring rule. The backend runs any decode
    rounds within this node and returns the labels through its rows port.
    """

    spec: ClassifySpec | None = None

    answer_kind: ClassVar[str] = "classifies"
    spec_type: ClassVar[type[InferenceSpec]] = ClassifySpec
    document_ids: ClassVar[bool] = True
    type_name: ClassVar[str] = "quail.ai_classify"
    runtime_key: ClassVar[str] = type_name

    @property
    def document_aliases(self) -> tuple[str, ...]:
        return () if self.spec is None else (self.spec.anchor,)

    def document_outputs(self, alias: str) -> tuple[str, ...]:
        return ("rows", f"ids:{alias}")

    def can_follow(self, previous: PhysicalNode, first: PhysicalNode) -> bool:
        if isinstance(previous, AiJoin):
            return self.classifies_join(previous)
        # a classification of documents continues any pipeline that
        # shows them under its head
        return (self.spec.partner is None
                and self.document_head == first.document_head)

    def classifies_join(self, join: "AiJoin") -> bool:
        """Return whether this classification reads the join's surviving pairs."""
        return (self.spec is not None and self.spec.partner is not None
                and self.spec.anchor == join.anchor
                and sorted(self.spec.aliases) == sorted((join.anchor, *join.partners)))

    def explain_fields(self) -> Mapping[str, Any]:
        return {
            **super().explain_fields(),
            "labels": [] if self.spec is None else list(self.spec.labels),
            "scoring": None if self.spec is None else self.spec.scoring,
            "draws": 1 if self.spec is None else self.spec.draws,
            "probabilities": (False if self.spec is None
                              else self.spec.probabilities),
        }
