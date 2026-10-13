"""Execution settings carried by extraction plans."""

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping


@dataclass(frozen=True)
class ExtractionConfig:
    """Limits for answer decoding and source continuation tokenization.

    Attributes:
        max_answer_tokens: Maximum tokens in an extracted answer.
        draft_tokens: Source tokens to verify after each chosen token.
        top_tokens: Vocabulary candidates returned by each GPU readout.
        first_window_chars: Source characters tokenized in the first window.
        window_chars: Source characters tokenized in later windows.
    """

    max_answer_tokens: int = 64
    draft_tokens: int = 16
    top_tokens: int = 1024
    first_window_chars: int = 64
    window_chars: int = 512

    def __post_init__(self):
        for attribute in fields(self):
            value = getattr(self, attribute.name)
            minimum = 0 if attribute.name == "draft_tokens" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{attribute.name} must be an integer >= {minimum}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExtractionConfig":
        """Read extraction settings from a serialized plan."""
        return cls(**value)

    def to_dict(self) -> dict:
        """Return settings for plan serialization."""
        return asdict(self)
