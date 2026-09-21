"""Stable sampling of named items, so a smaller sample nests in a larger one."""

from __future__ import annotations

import hashlib


def stable_rank(name: str, seed: int) -> int:
    """A stable random rank for one name; the lowest ranks are sampled."""
    value = f"{seed}\0{name}".encode()
    return int.from_bytes(
        hashlib.blake2b(value, digest_size=16).digest(), "big")


def stable_sample(names, n: int, seed: int) -> list[str]:
    """The n names of lowest rank, in name order.

    Ranks depend only on the name and the seed, so a smaller sample is
    a subset of a larger one.
    """
    ranked = sorted(names, key=lambda name: (stable_rank(name, seed), name))
    return sorted(ranked[:n])
