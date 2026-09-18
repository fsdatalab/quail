"""Priority and price of document KV retained between operators."""

from dataclasses import dataclass, field

from quail.cost.dense_decoder_cost import dense_params, flops_per_pair


@dataclass(frozen=True)
class RetentionPolicy:
    """Price the next planned reuse of each document prefix."""

    linear_seconds: float
    pair_seconds: float
    uses: dict = field(default_factory=dict)

    def priority(self, key, tokens: int, pages: int) -> tuple:
        alias, document = key
        probability, next_use = self.uses.get(alias, (0.0, 0))
        seconds = (self.linear_seconds * tokens
                   + self.pair_seconds * tokens * (tokens + 1) / 2)
        return (probability * seconds / pages, -next_use, -document)


def retention_pages(arena_tokens: int, chunk_tokens: int, page_tokens: int) -> int:
    """Reserve two execution chunks before allocating retained pages."""
    return max(0, arena_tokens // page_tokens
               - -(-2 * chunk_tokens // page_tokens))


def coefficients(model, device) -> dict:
    """Return the model's ideal prefix computation coefficients."""
    return {
        "linear_seconds": 2 * dense_params(model) / device.arithmetic_bandwidth(
            model.weight_precision),
        "pair_seconds": flops_per_pair(model) * model.layers
        / device.arithmetic_bandwidth(model.attention_precision),
    }
