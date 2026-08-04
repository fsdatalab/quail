"""Pessimistic-traffic repricing: charge KV reads and writes as time added
to compute instead of overlapped under it.

The paper's tau_0 assigns each batch max(attention compute, KV traffic) for
the attention group, which grants perfect overlap: whenever attention
compute dominates, the KV bytes ride free. Real kernels overlap imperfectly
and can transfer a block more than once, so this module reprices a
validated manifest under the no-overlap rule

    tau_add(B) = D(B) + attention_compute(B) + kv_traffic(B),

with D(B) unchanged (its internal weight-read max kept). The schedules
themselves are unchanged and remain feasible; only the assigned times move.
This is a declared sensitivity variant, not the primary model; a calibrated
engine sits between the two rules.

The matching lower bound adds the minimum traffic any schedule of the
policy must move: the write-through stores (as in lb.py) plus the reads the
policy structurally cannot avoid. Task-first and full speculation consume
every stored block inside its producing batch, so their minimum reads are
zero; the strict pipeline must re-read a survivor's document KV for every
later-stage branch, executed with the realized reach of that stage."""

import numpy as np

from docengine.costmodel import kv_capacity_tokens
from docengine.instance import Instance, survival
from docengine.lb import ledgers


def reprice_records(inst: Instance, records) -> dict:
    m, dev = inst.model, inst.device
    wq = m.n_q * m.d_h
    D = sum(r["D"] for r in records)
    attn = sum(4.0 * m.L * wq * r["A"] / dev.R_A for r in records)
    traffic = sum(m.kappa * (r["K_L"] + r["K_W"]) / dev.BW for r in records)
    reads = sum(m.kappa * r["K_L"] / dev.BW for r in records)
    tau_max = sum(r["tau"] for r in records)
    return dict(tau_max=tau_max, tau_add=D + attn + traffic, D=D,
                attn=attn, traffic=traffic, reads=reads)


def resource_lb_additive(inst: Instance, policy: str, X, k: int = 1) -> float:
    """policy in {'task','pipe','fullspec'}; for a lookahead-k class pass
    policy='pipe' with k: its minimum reads occur only at block boundaries
    (stages 1+k, 1+2k, ...), because branches inside a block are fused with
    the read of the block's first stage or with the prefill."""
    m, dev = inst.model, inst.device
    led = ledgers(inst, policy, X)
    d = np.asarray(inst.d, dtype=np.int64)
    Y = survival(np.asarray(X))
    reads_min = 0
    if policy == "pipe":
        for j in range(1 + k, inst.n + 1, k):
            reads_min += int(d[Y[:, j - 1].astype(bool)].sum())
    cap = kv_capacity_tokens(m, dev)
    import math
    b_min = max(1, math.ceil(led["doc_tokens"] / cap)) if cap > 0 else 1
    t_dense = max(2.0 * m.P * led["U_tot"] / dev.R_D,
                  b_min * m.W_run / dev.BW)
    t_attn = 4.0 * m.L * m.attn_width * led["A_tot"] / dev.R_A
    t_traffic = m.kappa * (led["V_tot"] + reads_min) / dev.BW
    return t_dense + t_attn + t_traffic
