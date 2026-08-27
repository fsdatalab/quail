"""The speed of light model: work records, the three KV operations,
and the time floor, all from counted constants.

Three things cost time and nothing else is counted:

  1. the dense projections - 2 FLOPs per parameter per token
  2. attention - 4 * n_q * d_head FLOPs per scored (query, key) pair,
     per layer
  3. moving bytes - the weights once per forward pass and KV once per
     token written, plus KV read back wherever a later stage reads it

No measured or fitted performance constant appears anywhere here.
Every number comes from the model architecture (counted) or the
device datasheet (published). That is what lets the same arithmetic
serve two callers: the QUAIL-B speed of light calculation, where it
is a floor no run can beat, and the planner, where it ranks candidate
plans without trusting any calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from quail.specs import DeviceSpec, ModelSpec


@dataclass(frozen=True)
class Work:
    """Hardware independent work counted by the SoL model."""

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


# ---------------------------------------------------------- the model
# Counted rather than looked up.


def dense_params(model: ModelSpec) -> int:
    """Parameters every token passes through.

    Counted rather than quoted: `ModelSpec.params` is rounded to
    3.6e9 against Qwen3-4B's real 3,633,511,936, and that 0.93%
    lands straight on the largest term of the bound.

    Assumes the Qwen3 block: q/k/v/o projections with no bias, a
    gated MLP, two RMS norms per layer, and q/k head norms.
    Embeddings and the lm_head are left out: a token touches one
    embedding row rather than doing 2 FLOPs per parameter, and a
    filter reads logits at one position per evaluation.
    """
    h, dh = model.hidden, model.d_head
    attn = h * model.n_q * dh + 2 * h * model.n_kv * dh + model.n_q * dh * h
    mlp = 3 * h * model.intermediate
    norms = 2 * h + 2 * dh
    return (attn + mlp + norms) * model.layers + h


def flops_per_pair(model: ModelSpec) -> int:
    """Attention FLOPs for one (query token, key token) pair in one
    layer. The QK dot product runs over d_head dimensions, so
    2 * d_head; multiplying the weight into V costs another
    2 * d_head. Times n_q heads. At 4B: 4 * 32 * 128 = 16,384."""
    return 4 * model.n_q * model.d_head


def kv_bytes_per_token(model: ModelSpec) -> float:
    """One token's KV: a key and a value, per layer, per KV head.
    At 4B: 2 * 36 * 8 * 128 * 2 bytes = 147,456."""
    return model.kappa


# ------------------------------------------------ the three operations
# What the GPU is asked to do.


def triangle(n: float) -> float:
    """A causal sequence attending to itself: token 1 sees 1 key,
    token 2 sees 2, and so on. 1 + 2 + ... + n."""
    return n * (n + 1) / 2


def scan(prefix: float, suffix: float) -> Work:
    """Compute one document from nothing: [prefix | suffix] as one
    causal sequence. Every token attends to itself and everything
    before it, so the pairs are one triangle over the whole length.

    This is the only operation that pays for the document text.
    """
    n = prefix + suffix
    return Work(tokens=n, pairs=triangle(n), kv_written=n, kv_read=0.0)


def ask(prefix: float, suffix: float) -> Work:
    """Attach one more suffix to a prefix already in the arena.

    Only the suffix is computed. Each of its tokens attends to the
    whole resident prefix - a rectangle, `suffix * prefix` - and to
    itself and the suffix tokens before it - a triangle. The prefix
    is read back out of the arena once.

    The document is not recomputed and does not appear in `tokens`.
    That is KV rewind.
    """
    return Work(tokens=suffix,
                pairs=suffix * prefix + triangle(suffix),
                kv_written=suffix,
                kv_read=prefix)


def stream(prefix: float, suffixes) -> Work:
    """One resident prefix, many suffixes: a join anchor and its
    tuples.

    Suffixes are atomic and never attend to each other
    (`executor/pack.py`), so each is its own rectangle over the
    prefix plus its own triangle - exactly `ask`, repeated. The
    difference is the arena: the prefix is read back once for the
    whole stream, not once per tuple, because the tuples run
    consecutively against it.
    """
    tokens = pairs = 0.0
    for u in suffixes:
        tokens += u
        pairs += u * prefix + triangle(u)
    return Work(tokens=tokens, pairs=pairs, kv_written=tokens,
                kv_read=prefix)


# ------------------------------------------------------ speed of light


@dataclass(frozen=True)
class SpeedOfLight:
    """The bound, with every term it was built from."""
    work: Work
    passes: int
    bytes_moved: float
    dense: float
    attention: float
    compute: float
    memory: float

    @property
    def seconds(self) -> float:
        return max(self.compute, self.memory)

    @property
    def bound_by(self) -> str:
        return "compute" if self.compute >= self.memory else "memory"

    def explain(self) -> str:
        w = self.work
        return "\n".join([
            f"tokens         {w.tokens:>18,.0f}",
            f"pairs          {w.pairs:>18,.0f}",
            f"kv written     {w.kv_written:>18,.0f}",
            f"kv read        {w.kv_read:>18,.0f}",
            f"forward passes {self.passes:>18,d}",
            f"bytes moved    {self.bytes_moved:>18,.0f}",
            f"T_dense        {self.dense:>18.4f} s",
            f"T_attention    {self.attention:>18.4f} s",
            f"T_compute      {self.compute:>18.4f} s",
            f"T_memory       {self.memory:>18.4f} s",
            f"SoL            {self.seconds:>18.4f} s ({self.bound_by} bound)",
        ])


def speed_of_light(work: Work, model: ModelSpec, device: DeviceSpec,
                   chunk_tokens: int) -> SpeedOfLight:
    """Turn the four counts into a floor on wall time.

    `chunk_tokens` is the batch size the forward pass runs at. It
    decides how many times the weights are re-read, and what the
    engine picks for it is a planner decision, so it is an input
    here with no default.

    Compute and memory are combined with max, not added: the
    arithmetic units and the memory system run at once and a floor
    may assume they overlap perfectly. Inside compute the two terms
    are added, because the dense and attention kernels are separate
    launches on the same SMs.
    """
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be at least 1")
    dense = 2.0 * dense_params(model) * work.tokens / device.peak_flops
    # attention runs in bf16 (FlashAttention-3 over bf16 KV), so it
    # prices against the bf16 peak, not the fp8 one
    attention = (flops_per_pair(model) * work.pairs * model.layers
                 / device.attn_flops)
    passes = math.ceil(work.tokens / chunk_tokens) if work.tokens else 0
    moved = (model.W_mem * passes
             + kv_bytes_per_token(model) * (work.kv_written + work.kv_read))
    return SpeedOfLight(work=work, passes=passes, bytes_moved=moved,
                        dense=dense, attention=attention,
                        compute=dense + attention,
                        memory=moved / device.hbm_bw)
