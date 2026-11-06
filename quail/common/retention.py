"""Priority and price of document KV retained between operators."""

from dataclasses import dataclass, field

import _quail_native

from quail.common.budgets import budgets_of
from quail.common.work import triangle


@dataclass(frozen=True)
class RetentionPolicy:
    """Price the next planned reuse of each document prefix."""

    linear_seconds: float
    pair_seconds: float
    uses: dict = field(default_factory=dict)
    sliding_pair_seconds: float = 0.0
    sliding_window: int = 0
    checkpoint_pages: float = 0.0    # pages of the state a hybrid model's
    #                                  document keeps beside its KV

    def prefill_seconds(self, tokens: int) -> float:
        """Return ideal compute time to rebuild a prefix of this length."""
        return (self.linear_seconds * tokens
                + self.pair_seconds * triangle(tokens)
                + self.sliding_pair_seconds * triangle(tokens, self.sliding_window))

    def priority(self, key, tokens: int, pages: int) -> tuple:
        alias, document = key
        probability, next_use = self.uses.get(alias, (0.0, 0))
        seconds = self.prefill_seconds(tokens)
        held = pages + self.checkpoint_pages
        return (probability * seconds / held, -next_use, -document)


def retention_pages(arena_tokens: int, chunk_tokens: int, page_tokens: int) -> int:
    """Return the pages left for retained KV after two execution chunks."""
    return _quail_native.retention_pages(arena_tokens, chunk_tokens, page_tokens)


def coefficients(model, device, *, peak=False) -> dict:
    """Return the model's ideal prefix computation coefficients."""
    return budgets_of(model, device).coefficients(peak)
