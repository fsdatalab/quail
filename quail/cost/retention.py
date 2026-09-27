"""Priority and price of document KV retained between operators."""

from dataclasses import dataclass, field

from quail.cost.dense_decoder_cost import attention_pair_flops, dense_params
from quail.cost.work import triangle


@dataclass(frozen=True)
class RetentionPolicy:
    """Price the next planned reuse of each document prefix."""

    linear_seconds: float
    pair_seconds: float
    uses: dict = field(default_factory=dict)
    sliding_pair_seconds: float = 0.0
    sliding_window: int = 0

    def priority(self, key, tokens: int, pages: int) -> tuple:
        alias, document = key
        probability, next_use = self.uses.get(alias, (0.0, 0))
        seconds = (self.linear_seconds * tokens
                   + self.pair_seconds * triangle(tokens)
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
    return {
        "linear_seconds": 2 * dense_params(model) / device.arithmetic_bandwidth(
            model.weight_precision),
        "pair_seconds": full / bandwidth,
        "sliding_pair_seconds": sliding / bandwidth,
        "sliding_window": model.sliding_window,
    }
