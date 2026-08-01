"""Resource lower bounds (paper eq. 55) from per-policy ideal ledgers.

All ledgers take the realized outcome matrix X (offline scenario). Expected
versions follow by averaging over scenarios. B_KV_tot follows convention C3
(every document/prompt-block token is written once; branch tokens never), so
the bound is a valid LB for the modeled cost; reads are lower-bounded by 0.
B_min follows C8: under C3 every new document token's KV exists in HBM during
its batch, so per-batch document tokens cannot exceed free-KV capacity."""

import math

import numpy as np

from .costmodel import a_pairs, kv_capacity_tokens
from .instance import Instance, survival


def ledgers(inst: Instance, policy: str, X) -> dict:
    d = np.asarray(inst.d, dtype=np.int64)
    Y = survival(np.asarray(X))
    n, N = inst.n, inst.N
    U = A = W = 0  # dense tokens, attention pairs, written tokens (C3)
    if policy == "task":
        for j in range(1, n + 1):
            reach = Y[:, j - 1].astype(bool)
            if reach.any():
                U += inst.p[j - 1]
                A += a_pairs(0, inst.p[j - 1])
                W += inst.p[j - 1]
            dj = d[reach]
            U += int(dj.sum())
            A += int((dj * inst.p[j - 1]).sum() + (dj * (dj + 1) // 2).sum())
            W += int(dj.sum())
        doc_tokens = W
    else:
        U += int(d.sum())
        A += int((d * (d + 1) // 2).sum())
        W += int(d.sum())
        doc_tokens = int(d.sum())
        for j in range(1, n + 1):
            if policy == "fullspec":
                reach = np.ones(N, dtype=bool)   # every branch, every doc
            else:
                # pipe, and adaptive spec (which may do as little as pipe):
                # only logically required branches are guaranteed work
                reach = Y[:, j - 1].astype(bool)
            dj = d[reach]
            U += int(reach.sum()) * inst.p[j - 1]
            A += int((dj * inst.p[j - 1]).sum()) \
                + int(reach.sum()) * (inst.p[j - 1] * (inst.p[j - 1] + 1) // 2)
            # branch tokens are never written (C3)
    return dict(U_tot=U, A_tot=A, KW_tot=W, doc_tokens=doc_tokens)


def resource_lb(inst: Instance, policy: str, X) -> dict:
    led = ledgers(inst, policy, X)
    m, dev = inst.model, inst.device
    cap = kv_capacity_tokens(m, dev)
    b_min = max(1, math.ceil(led["doc_tokens"] / cap)) if cap > 0 else 1
    if inst.max_new_tokens:
        b_min = max(b_min, math.ceil(led["U_tot"] / inst.max_new_tokens))
    t_dense = max(2.0 * m.P * led["U_tot"] / dev.R_D,
                  b_min * m.W_run / dev.BW)
    t_attn = max(4.0 * m.L * m.attn_width * led["A_tot"] / dev.R_A,
                 m.kappa * led["KW_tot"] / dev.BW)
    return dict(LB=t_dense + t_attn, t_dense=t_dense, t_attn=t_attn,
                B_min=b_min, **led)
