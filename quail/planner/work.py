"""Hardware independent work for filters and joins."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Work:
    """Tokens, attention pairs, and KV movement for a query."""

    tokens: float = 0.0
    pairs: float = 0.0
    kv_written: float = 0.0
    kv_read: float = 0.0

    def __add__(self, other: "Work") -> "Work":
        return Work(
            self.tokens + other.tokens,
            self.pairs + other.pairs,
            self.kv_written + other.kv_written,
            self.kv_read + other.kv_read,
        )

    def __mul__(self, count: float) -> "Work":
        return Work(
            self.tokens * count,
            self.pairs * count,
            self.kv_written * count,
            self.kv_read * count,
        )

    def dominates(self, other: "Work") -> bool:
        """Return whether every count is no larger than the other record."""

        return (
            self.tokens <= other.tokens
            and self.pairs <= other.pairs
            and self.kv_written <= other.kv_written
            and self.kv_read <= other.kv_read
        )


def triangle(n: float) -> float:
    """Return the causal attention pairs for a sequence of length n."""

    return n * (n + 1) / 2


def scan(prefix: float, suffix: float) -> Work:
    """Compute one document and its first suffix from nothing."""

    n = prefix + suffix
    return Work(tokens=n, pairs=triangle(n), kv_written=n)


def ask(prefix: float, suffix: float) -> Work:
    """Attach one suffix to a prefix already available in KV."""

    return Work(
        tokens=suffix,
        pairs=suffix * prefix + triangle(suffix),
        kv_written=suffix,
        kv_read=prefix,
    )


def stream(prefix: float, suffixes) -> Work:
    """Attach several independent suffixes to one prefix in KV."""

    tokens = 0.0
    pairs = 0.0
    for suffix in suffixes:
        tokens += suffix
        pairs += suffix * prefix + triangle(suffix)
    return Work(
        tokens=tokens,
        pairs=pairs,
        kv_written=tokens,
        kv_read=prefix,
    )
