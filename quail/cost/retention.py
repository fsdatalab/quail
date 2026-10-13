"""Priority and price of document KV retained between operators."""

from dataclasses import dataclass, field

from quail.cost.dense_decoder_cost import (
    attention_pair_flops,
    attention_rates,
    dense_params,
)
from quail.cost.work import triangle
from quail.specs.base import causal_rate


@dataclass(frozen=True)
class RetentionPolicy:
    """Price the next planned reuse of each document prefix.

    With causal_tokens set, a prefix's attention pairs price at the
    measured causal rate for its length, pair_flops each; otherwise at
    pair_seconds each.
    """

    linear_seconds: float
    pair_seconds: float
    uses: dict = field(default_factory=dict)
    sliding_pair_seconds: float = 0.0
    sliding_window: int = 0
    pair_flops: float = 0.0
    causal_tokens: tuple = ()
    causal_rates: tuple = ()

    def priority(self, key, tokens: int, pages: int) -> tuple:
        alias, document = key
        probability, next_use = self.uses.get(alias, (0.0, 0))
        pairs = triangle(tokens)
        pair_seconds = (
            self.pair_flops * pairs
            / causal_rate(self.causal_tokens, self.causal_rates, tokens)
            if self.causal_tokens and tokens > 0
            else self.pair_seconds * pairs)
        seconds = (self.linear_seconds * tokens
                   + pair_seconds
                   + self.sliding_pair_seconds * triangle(tokens, self.sliding_window))
        return (probability * seconds / pages, -next_use, -document)


def retention_pages(arena_tokens: int, chunk_tokens: int, page_tokens: int) -> int:
    """Reserve two execution chunks before allocating retained pages."""
    return max(0, arena_tokens // page_tokens
               - -(-2 * chunk_tokens // page_tokens))


def coefficients(model, device) -> dict:
    """Return the model's ideal prefix computation coefficients."""
    full, sliding = attention_pair_flops(model)
    bandwidth = device.arithmetic_bandwidth(model.attention_precision)
    rates = attention_rates(model, device)
    measured = {} if rates is None else {
        "pair_flops": full,
        "causal_tokens": list(rates.causal_tokens),
        "causal_rates": list(rates.causal),
    }
    return measured | {
        "linear_seconds": 2 * dense_params(model) / device.arithmetic_bandwidth(
            model.weight_precision),
        "pair_seconds": full / bandwidth,
        "sliding_pair_seconds": sliding / bandwidth,
        "sliding_window": model.sliding_window,
    }
