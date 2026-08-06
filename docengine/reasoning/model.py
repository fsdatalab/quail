"""Cost primitives and Bellman value recurrences for reasoning filters.

Implements the reasoning-filter model (paper/PAPER.md sections 5 and
8, E4; the original design note is in git history) at the fluid
level: mean document
length, mean thinking length per stage, fractional survivor counts.
Makespan estimates are the maximum of three certified components
(total compute seconds, total bandwidth seconds, one block's
dependency path), the paper's resource-bound style extended with
stepwise generation. Simplifications, documented here on purpose: the
last cohort's decode tail is folded into the resource totals rather
than the path, and cross-block overlap is credited fully (blocks
pipeline), so the estimate is a certified-flavor lower envelope of
each policy family, not a schedule.
"""

import math
from dataclasses import dataclass

from ..configs import DeviceConfig, ModelConfig
from ..costmodel import kv_capacity_tokens
from ..plan.cost import (decode_step_seconds, dense_seconds,
                         fluid_block_seconds)


@dataclass(frozen=True)
class RInstance:
    model: ModelConfig
    device: DeviceConfig
    N: int                  # documents
    d: float                # mean document tokens (with flags line)
    s: tuple                # per-stage pass rates, length n
    p: tuple                # per-stage question tokens
    g: tuple                # per-stage generated tokens (thinking + answer)
    f: tuple = None         # task template extra tokens (prefix + cue)
    calib: float = 1.0      # phi: effective compute = calib * R_D
    cap_tokens: int = None  # override the analytic KV capacity

    @property
    def n(self):
        return len(self.s)

    def fj(self, j):
        return 32.0 if self.f is None else self.f[j]

    @property
    def cap(self):
        if self.cap_tokens is not None:
            return self.cap_tokens
        return kv_capacity_tokens(self.model, self.device)

    @property
    def R_C(self):
        return self.calib * self.device.R_D

    def survival(self, j):
        """Fraction of documents alive entering 0-indexed stage j."""
        out = 1.0
        for i in range(j):
            out *= self.s[i]
        return out


def t_pre(inst, tokens):
    """Seconds of dense compute to read `tokens` (no attention)."""
    return dense_seconds(inst.model, tokens, inst.R_C)


def t_pre_doc(inst, count, d):
    """Seconds to prefill `count` documents of length d, including the
    quadratic self-attention compute (4 L w_Q d^2/2 per document),
    negligible at hundreds of tokens and dominant past roughly 25,000
    (the crossover d* = P / (L w_Q)). Long-document validation measured
    2.15x the dense-only cost at 30k tokens against 2.23x predicted by
    this term."""
    m = inst.model
    attn = 2.0 * m.L * m.attn_width * d * d * count / inst.R_C
    return t_pre(inst, count * d) + attn


def step_time(inst, m, ctx):
    """Seconds for one decode step advancing m calls at mean context ctx."""
    return decode_step_seconds(inst.model, inst.device, m, ctx, inst.R_C)


def compositions(n):
    """All ordered partitions of stages 1..n into lookahead blocks."""
    if n == 0:
        return [()]
    out = []
    for k in range(1, n + 1):
        for rest in compositions(n - k):
            out.append((k,) + rest)
    return out


def value_taskfirst(inst):
    """Bellman chain over gated stage waves for the task-first policy."""
    comp = bw = 0.0
    stage_times = []
    for j in range(inst.n):
        Nj = inst.N * inst.survival(j)
        if Nj < 1e-9:
            break
        pre = Nj * (inst.fj(j) + inst.d)
        foot = inst.fj(j) + inst.d + inst.g[j]
        m = max(1.0, min(Nj, inst.cap / foot))
        rounds = math.ceil(Nj / m)
        ctx = inst.fj(j) + inst.d + inst.g[j] / 2.0
        dec_tokens = Nj * inst.g[j]
        comp_j = t_pre_doc(inst, Nj, inst.fj(j) + inst.d) \
            + t_pre(inst, dec_tokens)
        steps = inst.g[j] * rounds
        bw_j = (inst.model.W_run * steps
                + inst.model.kappa * (dec_tokens * ctx + pre + dec_tokens)
                ) / inst.device.BW
        stage_times.append(max(comp_j, bw_j))
        comp += comp_j
        bw += bw_j
    return dict(T=sum(stage_times), compute_s=comp, bandwidth_s=bw,
                stages=stage_times)


def value_blockwise(inst, comp_tuple):
    """Value of the document-first policy with a fixed stage composition
    (all ones: pipeline; (n,): full speculation). Documents move in
    memory-sized blocks; within a composition block all its branch
    questions are issued ungated, wasting the branches of documents
    that fail inside the block."""
    starts = []
    j = 0
    for k in comp_tuple:
        starts.append((j, k))
        j += k
    assert j == inst.n
    peak = max(sum(inst.p[jj] + inst.g[jj] for jj in range(j0, j0 + k))
               for j0, k in starts)
    B = max(1.0, min(float(inst.N), inst.cap / (inst.d + peak)))
    nb = inst.N / B

    comp_b = t_pre_doc(inst, B, inst.d)
    bw_b = inst.model.kappa * B * inst.d / inst.device.BW
    path = t_pre_doc(inst, B, inst.d)
    for j0, k in starts:
        bt = B * inst.survival(j0)
        if bt < 1e-9:
            continue
        p_sum = sum(inst.p[jj] for jj in range(j0, j0 + k))
        g_sum = sum(inst.g[jj] for jj in range(j0, j0 + k))
        g_max = max(inst.g[jj] for jj in range(j0, j0 + k))
        calls = bt * k
        ctx = inst.d + p_sum / k + g_sum / (2.0 * k)
        pre = bt * p_sum
        dec = bt * g_sum
        comp_t = t_pre(inst, pre + dec)
        bw_t = (inst.model.W_run * g_max
                + inst.model.kappa * (calls * inst.d + dec * ctx + pre + dec)
                ) / inst.device.BW
        comp_b += comp_t
        bw_b += bw_t
        path += t_pre(inst, pre) + g_max * step_time(inst, calls, ctx)
    return dict(T=max(nb * comp_b, nb * bw_b, path),
                compute_s=nb * comp_b, bandwidth_s=nb * bw_b,
                path_s=path, block_docs=B, blocks=nb)


def best_composition(inst):
    """Solve the composition Bellman recurrence by exact enumeration
    (n <= 4 in all our workloads, so at most eight compositions)."""
    best = None
    for K in compositions(inst.n):
        v = value_blockwise(inst, K)
        if best is None or v["T"] < best[1]["T"]:
            best = (K, v)
    return best


def block_cost_fluid(inst, j0, k, surv):
    """Additive fluid cost, in seconds, of one lookahead block covering
    0-indexed stages j0..j0+k-1 entered by fraction `surv` of the N
    documents. Inside a block every question launches at block entry
    (ungated), so each member stage is charged at the block's entering
    survival; every question and generated token passes the forward
    pass once and writes its KV once. Additive across blocks, which
    the prefix dynamic program group_stages requires; value_blockwise's
    certified envelope takes a max over resource totals, is not
    additive, and is therefore minimized by best_composition's
    exhaustive enumeration instead."""
    docs = inst.N * surv
    toks = docs * sum(inst.p[j] + inst.g[j] for j in range(j0, j0 + k))
    return fluid_block_seconds(inst.model, inst.device, toks, inst.R_C)


def group_stages(inst, block_cost=block_cost_fluid):
    """Algorithm 3 of the paper: cheapest contiguous partition of the
    fixed stage order by a prefix dynamic program. The survival
    entering a block is a product of earlier selectivities only
    (Lemma 1), so prefix optima compose. n(n+1)/2 block-cost
    evaluations. Returns (composition tuple, cost)."""
    n = inst.n
    best = [0.0] + [float("inf")] * n
    cut = [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j):
            c = best[i] + block_cost(inst, i, j - i, inst.survival(i))
            if c < best[j]:
                best[j], cut[j] = c, i
    blocks, j = [], n
    while j > 0:
        blocks.append(j - cut[j])
        j = cut[j]
    return tuple(reversed(blocks)), best[n]
