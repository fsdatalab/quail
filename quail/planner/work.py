"""Hardware work counts shared by planning and the SoL report."""

import math
from dataclasses import dataclass

from quail.specs import DeviceSpec, ModelSpec


@dataclass(frozen=True)
class Work:
    """Hardware independent work for one query plan."""

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
        """Return whether this record is no larger in every category."""

        return (
            self.tokens <= other.tokens
            and self.pairs <= other.pairs
            and self.kv_written <= other.kv_written
            and self.kv_read <= other.kv_read
        )


def triangle(tokens: float) -> float:
    """Causal attention pairs for one sequence."""

    return tokens * (tokens + 1) / 2


def dense_params(model: ModelSpec) -> int:
    """Parameters used by every fresh token in the Qwen3 block."""

    h, dh = model.hidden, model.d_head
    attention = (
        h * model.n_q * dh
        + 2 * h * model.n_kv * dh
        + model.n_q * dh * h
    )
    mlp = 3 * h * model.intermediate
    norms = 2 * h + 2 * dh
    return (attention + mlp + norms) * model.layers + h


def flops_per_pair(model: ModelSpec) -> int:
    """Attention FLOPs for one query and key pair in one layer."""

    return 4 * model.n_q * model.d_head


def prefix_recompute_seconds(
    prefix_tokens: int, model: ModelSpec, device: DeviceSpec
) -> float:
    """Ideal compute time avoided by retaining one document prefix.

    A miss writes the prefix KV and a hit reads it, so the prefix moves
    the same number of KV bytes either way. The avoided work is the dense
    work for the fresh prefix and its causal attention triangle.
    """

    if prefix_tokens < 0:
        raise ValueError("prefix_tokens must be nonnegative")
    dense = (
        2.0 * dense_params(model) * prefix_tokens / device.peak_flops
    )
    attention = (
        flops_per_pair(model)
        * triangle(prefix_tokens)
        * model.layers
        / device.attn_flops
    )
    return dense + attention


def scan(prefix: float, suffix: float) -> Work:
    """Compute a prefix and suffix as one causal sequence."""

    total = prefix + suffix
    return Work(tokens=total, pairs=triangle(total),
                kv_written=total)


def ask(prefix: float, suffix: float) -> Work:
    """Compute one suffix against resident prefix KV."""

    return Work(tokens=suffix,
                pairs=suffix * prefix + triangle(suffix),
                kv_written=suffix, kv_read=prefix)


def stream(prefix: float, suffix: float, count: float) -> Work:
    """Compute count independent suffixes against one resident prefix."""

    return Work(tokens=count * suffix,
                pairs=count * (suffix * prefix + triangle(suffix)),
                kv_written=count * suffix,
                kv_read=prefix if count else 0.0)


def ideal_seconds(work: Work, model: ModelSpec, device: DeviceSpec,
                  chunk_tokens: int) -> float:
    """One H100! lower bound from counted work and published limits."""

    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    dense = 2.0 * dense_params(model) * work.tokens / device.peak_flops
    attention = (
        flops_per_pair(model) * work.pairs * model.layers
        / device.attn_flops
    )
    passes = math.ceil(work.tokens / chunk_tokens) if work.tokens else 0
    moved = (model.W_mem * passes
             + model.kappa * (work.kv_written + work.kv_read))
    return max(dense + attention, moved / device.hbm_bw)
