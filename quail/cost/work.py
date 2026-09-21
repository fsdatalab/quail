"""Work counts for filters and joins at the model's attention window."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Work:
    """Tokens, attention pairs, KV movement, and image-tower counts.

    `image_patches` is the patches the vision tower embeds.
    `image_pairs` is the sum, over images, of patches squared: attention
    stays inside one image, so the images do not form one sequence.
    `image_soft_tokens` is the pooled tokens projected into the decoder.
    """

    tokens: float = 0.0
    pairs: float = 0.0
    kv_written: float = 0.0
    kv_read: float = 0.0
    sliding_pairs: float = 0.0
    sliding_kv_read: float = 0.0
    image_patches: float = 0.0
    image_pairs: float = 0.0
    image_soft_tokens: float = 0.0

    def __add__(self, other: "Work") -> "Work":
        return Work(
            self.tokens + other.tokens,
            self.pairs + other.pairs,
            self.kv_written + other.kv_written,
            self.kv_read + other.kv_read,
            self.sliding_pairs + other.sliding_pairs,
            self.sliding_kv_read + other.sliding_kv_read,
            self.image_patches + other.image_patches,
            self.image_pairs + other.image_pairs,
            self.image_soft_tokens + other.image_soft_tokens,
        )

    def __mul__(self, count: float) -> "Work":
        return Work(
            self.tokens * count,
            self.pairs * count,
            self.kv_written * count,
            self.kv_read * count,
            self.sliding_pairs * count,
            self.sliding_kv_read * count,
            self.image_patches * count,
            self.image_pairs * count,
            self.image_soft_tokens * count,
        )

    def dominates(self, other: "Work") -> bool:
        """Return whether every count is no larger than the other record."""
        return (
            self.tokens <= other.tokens
            and self.pairs <= other.pairs
            and self.kv_written <= other.kv_written
            and self.kv_read <= other.kv_read
            and self.sliding_pairs <= other.sliding_pairs
            and self.sliding_kv_read <= other.sliding_kv_read
            and self.image_patches <= other.image_patches
            and self.image_pairs <= other.image_pairs
            and self.image_soft_tokens <= other.image_soft_tokens
        )


def triangle(n: float, window: int = 0) -> float:
    """Count causal attention pairs, including each token's own key."""
    if window and n > window:
        return window * n - window * (window - 1) / 2
    return n * (n + 1) / 2


def scan(prefix: float, suffix: float, *, window: int = 0) -> Work:
    """Compute one document and its first suffix from nothing."""
    n = prefix + suffix
    return Work(tokens=n, pairs=triangle(n), kv_written=n,
                sliding_pairs=triangle(n, window) if window else 0.0)


def ask(prefix: float, suffix: float, *, window: int = 0) -> Work:
    """Attach one suffix to a prefix already available in KV."""
    return Work(
        tokens=suffix,
        pairs=suffix * prefix + triangle(suffix),
        kv_written=suffix,
        kv_read=prefix,
        sliding_pairs=(triangle(prefix + suffix, window) - triangle(prefix, window)
                       if window else 0.0),
        sliding_kv_read=min(prefix, window - 1) if window else 0.0,
    )


def stream(prefix: float, suffixes, *, window: int = 0) -> Work:
    """Attach several independent suffixes to one prefix in KV."""
    tokens = pairs = sliding_pairs = 0.0
    for suffix in suffixes:
        tokens += suffix
        pairs += suffix * prefix + triangle(suffix)
        if window:
            sliding_pairs += (triangle(prefix + suffix, window)
                              - triangle(prefix, window))
    return Work(
        tokens=tokens,
        pairs=pairs,
        kv_written=tokens,
        kv_read=prefix,
        sliding_pairs=sliding_pairs,
        sliding_kv_read=min(prefix, window - 1) if window else 0.0,
    )
