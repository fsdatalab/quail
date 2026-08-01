"""Certified resource lower bounds (revised paper sec. on resource lower
bounds, eq. resource-lb).

The bound is built from four certified lower ledgers (l_U, l_A, l_V, l_m),
each of which must lower-bound the corresponding total of EVERY feasible
schedule of the fixed policy instance, and none of which may be copied from
a candidate schedule. Ledgers here are the recomputation-free, maximal-
sharing totals for the realized outcome matrix X:

  l_U  every schedule performs at least the causally required dense tokens;
       recomputation only adds tokens.
  l_A  same argument for attention pairs.
  l_V  KV bytes under the primary write-through fused ledger: every doc or
       prompt-block position with a later causal consumer is stored exactly
       once by any schedule of the modeled action families, and branch
       interiors are stored (p_j - 1 per branch); recomputation only adds
       stores, and loads are lower-bounded by zero. Pass kv_ledger='zero'
       for the paper's universally safe default l_V = 0 (valid even for
       action families outside the write-through convention).
  l_m  batch count from the certified per-batch token bound: under the
       conservative lifetime rule all of a batch's new document tokens are
       live simultaneously, so document tokens per batch cannot exceed the
       free-KV capacity (the paper's kernel premise with gamma = kappa),
       plus any configured new-token cap. Falls back to 1.
"""

import math

import numpy as np

from .costmodel import a_pairs, kv_capacity_tokens
from .instance import Instance, survival


def ledgers(inst: Instance, policy: str, X, kv_ledger: str = "write_through") -> dict:
    d = np.asarray(inst.d, dtype=np.int64)
    Y = survival(np.asarray(X))
    n, N = inst.n, inst.N
    U = A = V = 0  # dense tokens, attention pairs, stored tokens
    if policy == "task":
        for j in range(1, n + 1):
            reach = Y[:, j - 1].astype(bool)
            if reach.any():
                U += inst.p[j - 1]
                A += a_pairs(0, inst.p[j - 1])
                V += inst.p[j - 1]
            dj = d[reach]
            U += int(dj.sum())
            A += int((dj * inst.p[j - 1]).sum() + (dj * (dj + 1) // 2).sum())
            V += int((dj - 1).sum())        # decision leaf is ephemeral
        doc_tokens = U
    else:
        U += int(d.sum())
        A += int((d * (d + 1) // 2).sum())
        V += int(d.sum())
        doc_tokens = int(d.sum())
        for j in range(1, n + 1):
            if policy == "fullspec":
                reach = np.ones(N, dtype=bool)   # every branch, every doc
            else:
                # pipe, and adaptive spec (which may do as little as pipe):
                # only logically required branches are guaranteed work
                reach = Y[:, j - 1].astype(bool)
            dj = d[reach]
            cnt = int(reach.sum())
            U += cnt * inst.p[j - 1]
            A += int((dj * inst.p[j - 1]).sum()) \
                + cnt * (inst.p[j - 1] * (inst.p[j - 1] + 1) // 2)
            V += cnt * (inst.p[j - 1] - 1)   # branch interiors are stored
    if kv_ledger == "zero":
        V = 0
    return dict(U_tot=U, A_tot=A, V_tot=V, doc_tokens=doc_tokens)


def resource_lb(inst: Instance, policy: str, X,
                kv_ledger: str = "write_through") -> dict:
    led = ledgers(inst, policy, X, kv_ledger)
    m, dev = inst.model, inst.device
    cap = kv_capacity_tokens(m, dev)
    b_min = max(1, math.ceil(led["doc_tokens"] / cap)) if cap > 0 else 1
    if inst.max_new_tokens:
        b_min = max(b_min, math.ceil(led["U_tot"] / inst.max_new_tokens))
    t_dense = max(2.0 * m.P * led["U_tot"] / dev.R_D,
                  b_min * m.W_run / dev.BW)
    t_attn = max(4.0 * m.L * m.attn_width * led["A_tot"] / dev.R_A,
                 m.kappa * led["V_tot"] / dev.BW)
    return dict(LB=t_dense + t_attn, t_dense=t_dense, t_attn=t_attn,
                B_min=b_min, **led)
