"""Speed-of-light (SOL): the minimum possible wall time for a query's
real workload, if every kernel ran at peak rate. A lower bound, not a
prediction - real runs are always slower. `sol_efficiency = sol_s /
wall_s` should never exceed 1.0; if it does, either the wall-time
measurement or this estimate is wrong.

Three equations, ported directly from a research note's roofline
derivation (not re-derived here):

    T_dense     = 2 * P * tokens / fp8_peak_flops
    T_attention = 4 * n_q * d_head * pairs * layers / bf16_peak_flops
    T_memory    = (W_mem * passes + kappa * (tokens + context_reads))
                  / hbm_bw
    SOL         = max(T_dense + T_attention, T_memory)

T_dense and T_attention use different peak rates because FlashAttention
runs bf16 even when the GEMMs run fp8 - on Hopper that's half the fp8
rate. `tokens`, `pairs`, and `context_reads` are workload-shape
numbers, not spec constants: see runtime/session.py's per-query walk
for how they're built from what a query actually ran.

Deliberately simpler than the causal/streaming split in
quail/planner/budgets.py's roofline section: this aggregates all
compute into one max() against all memory traffic, rather than pricing
each kernel with its own max() and summing. It does not net out KV
store restores or price a join stage's "frame" write separately - see
reports/shipped_features/ for the stated caveats that follow from
that.
"""

import math

from quail.planner.budgets import chunk_budget
from quail.specs import DeviceSpec, ModelSpec


def dense_seconds(model: ModelSpec, device: DeviceSpec,
                  tokens: int) -> float:
    """Ideal seconds for every token to pass through every dense
    projection (QKV, output, MLP), all layers, at the fp8 rate."""
    return 2.0 * model.params * tokens / device.peak_flops


def attention_seconds(model: ModelSpec, device: DeviceSpec,
                      pairs: int) -> float:
    """Ideal seconds for the attention pair FLOPs, all layers, at the
    bf16 rate. `pairs`: the count of (query, key) token pairs actually
    scored - quadratic for a document's causal self-attention, linear
    for tokens streaming against an already-cached prefix."""
    return (4.0 * model.n_q * model.d_head * pairs * model.layers
            / device.bf16_peak_flops)


def memory_seconds(model: ModelSpec, device: DeviceSpec, tokens: int,
                   context_reads: int) -> float:
    """Ideal seconds for the bytes moved: the weight set, once per
    forward pass the chunk budget splits `tokens` into, plus the KV
    write/read for every token processed (`tokens`) and every token
    read from an already-resident cached prefix (`context_reads`)."""
    passes = math.ceil(tokens / chunk_budget(model, device)) if tokens else 0
    moved = model.W_mem * passes + model.kappa * (tokens + context_reads)
    return moved / device.hbm_bw


def sol_seconds(model: ModelSpec, device: DeviceSpec, *, tokens: int,
                pairs: int, context_reads: int) -> float:
    """The floor: minimum seconds to process a query's real workload
    if every kernel ran at peak. Compute (dense + attention) and
    memory overlap, so the wall time is whichever is larger, not
    their sum."""
    compute = (dense_seconds(model, device, tokens)
              + attention_seconds(model, device, pairs))
    memory = memory_seconds(model, device, tokens, context_reads)
    return max(compute, memory)