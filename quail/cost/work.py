"""Token, attention, and KV work at the model's attention window."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Work:
    """Tokens, attention pairs, KV movement, and output-head work.

    readout_flops and readout_bytes count the selected output head's
    computation and weight reads, using the operator's packing assumption.
    """

    tokens: float = 0.0
    pairs: float = 0.0
    kv_written: float = 0.0
    kv_read: float = 0.0
    sliding_pairs: float = 0.0
    sliding_kv_read: float = 0.0
    readout_flops: float = 0.0
    readout_bytes: float = 0.0

    def __add__(self, other: "Work") -> "Work":
        return Work(
            self.tokens + other.tokens,
            self.pairs + other.pairs,
            self.kv_written + other.kv_written,
            self.kv_read + other.kv_read,
            self.sliding_pairs + other.sliding_pairs,
            self.sliding_kv_read + other.sliding_kv_read,
            self.readout_flops + other.readout_flops,
            self.readout_bytes + other.readout_bytes,
        )

    def __mul__(self, count: float) -> "Work":
        return Work(
            self.tokens * count,
            self.pairs * count,
            self.kv_written * count,
            self.kv_read * count,
            self.sliding_pairs * count,
            self.sliding_kv_read * count,
            self.readout_flops * count,
            self.readout_bytes * count,
        )

    def total(self) -> "Work":
        """Sum work over document arrays, leaving scalar counts unchanged."""
        return Work(*(float(np.sum(value)) for value in (
            self.tokens, self.pairs, self.kv_written, self.kv_read,
            self.sliding_pairs, self.sliding_kv_read,
            self.readout_flops, self.readout_bytes)))

    def dominates(self, other: "Work") -> bool:
        """Return whether every count is no larger than the other record."""
        return (
            self.tokens <= other.tokens
            and self.pairs <= other.pairs
            and self.kv_written <= other.kv_written
            and self.kv_read <= other.kv_read
            and self.sliding_pairs <= other.sliding_pairs
            and self.sliding_kv_read <= other.sliding_kv_read
            and self.readout_flops <= other.readout_flops
            and self.readout_bytes <= other.readout_bytes
        )


def triangle(n: float, window: int = 0) -> float:
    """Count causal attention pairs, including each token's own key."""
    if window:
        visible = np.minimum(n, window)
        return visible * n - visible * (visible - 1) / 2
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
        sliding_kv_read=np.minimum(prefix, window - 1) if window else 0.0,
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
        sliding_kv_read=np.minimum(prefix, window - 1) if window else 0.0,
    )


def decode(prefix, rounds, *, window: int = 0) -> Work:
    """Append accepted tokens over sequential rounds, rereading KV each round.

    Args:
        prefix: Context length, or an array of document context lengths.
        rounds: Number of tokens fed in each fully accepted round.
        window: Attention window, or 0 for full attention.

    Returns:
        Token, attention, and KV work over the rounds.
    """
    work = Work()
    for tokens in rounds:
        work += ask(prefix, tokens, window=window)
        prefix = prefix + tokens
    if np.ndim(prefix):
        # Scalar suffix lengths apply once to each document in the array.
        work = Work(work.tokens * np.ones_like(prefix), work.pairs,
                    work.kv_written * np.ones_like(prefix), work.kv_read,
                    work.sliding_pairs, work.sliding_kv_read)
    return work
