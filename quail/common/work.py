"""Token, attention, and KV work at the model's attention window."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Work:
    """Tokens, full and sliding attention pairs, and KV and state movement.

    state_written and state_read count checkpoints: one write when a
    sequence's state is saved for later reuse, and one read when a
    sequence starts from a saved state. The bytes of one checkpoint
    come from the model, and are zero for a model without state layers.
    """

    tokens: float = 0.0
    pairs: float = 0.0
    kv_written: float = 0.0
    kv_read: float = 0.0
    sliding_pairs: float = 0.0
    sliding_kv_read: float = 0.0
    state_written: float = 0.0
    state_read: float = 0.0
    # tokens the model's SAE encodes, for AI.SAE_FEATURES
    sae_encoded: float = 0.0

    def __add__(self, other: "Work") -> "Work":
        return Work(
            self.tokens + other.tokens,
            self.pairs + other.pairs,
            self.kv_written + other.kv_written,
            self.kv_read + other.kv_read,
            self.sliding_pairs + other.sliding_pairs,
            self.sliding_kv_read + other.sliding_kv_read,
            self.state_written + other.state_written,
            self.state_read + other.state_read,
            self.sae_encoded + other.sae_encoded,
        )

    def __mul__(self, count: float) -> "Work":
        return Work(
            self.tokens * count,
            self.pairs * count,
            self.kv_written * count,
            self.kv_read * count,
            self.sliding_pairs * count,
            self.sliding_kv_read * count,
            self.state_written * count,
            self.state_read * count,
            self.sae_encoded * count,
        )

    def total(self) -> "Work":
        """Sum work over document arrays, leaving scalar counts unchanged."""
        return Work(*(float(np.sum(value)) for value in (
            self.tokens, self.pairs, self.kv_written, self.kv_read,
            self.sliding_pairs, self.sliding_kv_read, self.state_written,
            self.state_read, self.sae_encoded)))

def triangle(n: float, window: int = 0) -> float:
    """Count causal attention pairs, including each token's own key."""
    if window:
        visible = np.minimum(n, window)
        return visible * n - visible * (visible - 1) / 2
    return n * (n + 1) / 2
