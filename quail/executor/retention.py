"""Priority for document KV retained between operators."""

from dataclasses import dataclass, field


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
