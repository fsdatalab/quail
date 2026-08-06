"""Steady-state throughput programs for reasoning filters.

One program per policy family, per the reasoning-filter model in
docengine/reasoning/model.py. Documents
flow at rate lambda; three resource constraints bound lambda:

  compute    lambda * 2P * (prefill + decode tokens per doc) <= phi R_D
  bandwidth  lambda * (weight reads per doc at cohort m + note traffic
             per doc) <= BW
  memory     lambda * (token-seconds of residency per doc) <= capacity

The decode cohort m amortizes the per-step weight read and is swept
over a small grid; residency spans follow from the step time at that
cohort. The reported lambda is the best feasible over the grid with
the binding constraint named. At g = 1 per stage this must agree with
the answer-only world, the anchor tested in tests/test_reasoning.py.
"""

from .model import step_time

_M_GRID = (32, 64, 128, 256, 512, 1024, 2048, 4096)


def _starts(comp_tuple):
    out, j = [], 0
    for k in comp_tuple:
        out.append((j, k))
        j += k
    return out


def lp_throughput(inst, comp_tuple, m_grid=_M_GRID):
    """Max document rate. comp_tuple None means task-first; otherwise a
    stage composition for the document-first family."""
    mdl, dev = inst.model, inst.device
    task = comp_tuple is None

    if task:
        pre = sum(inst.survival(j) * (inst.fj(j) + inst.d)
                  for j in range(inst.n))
        dec = sum(inst.survival(j) * inst.g[j] for j in range(inst.n))
        doc_reads = 0.0
        resident = 0.0
    else:
        starts = _starts(comp_tuple)
        pre = inst.d + sum(inst.survival(j0)
                           * sum(inst.p[jj] for jj in range(j0, j0 + k))
                           for j0, k in starts)
        dec = sum(inst.survival(j0)
                  * sum(inst.g[jj] for jj in range(j0, j0 + k))
                  for j0, k in starts)
        doc_reads = sum(inst.survival(j0) * k * inst.d
                        for j0, k in starts)
        resident = inst.d
    ctx = inst.d + (max(inst.p) + max(inst.g) / 2.0)

    best = None
    for m in m_grid:
        st = step_time(inst, m, ctx)
        lam_c = inst.R_C / (2.0 * mdl.P * (pre + dec))
        bw_doc = ((mdl.W_run / m) * dec / dev.BW
                  + mdl.kappa * (dec * ctx + doc_reads + pre + dec) / dev.BW)
        lam_b = 1.0 / bw_doc if bw_doc > 0 else float("inf")

        # residency spans at this cohort's step time
        if task:
            occ = sum(inst.survival(j)
                      * (inst.fj(j) + inst.d + inst.g[j])
                      * (inst.g[j] * st
                         + 2.0 * mdl.P * (inst.fj(j) + inst.d) / inst.R_C)
                      for j in range(inst.n))
        else:
            span = 2.0 * mdl.P * inst.d / inst.R_C
            occ = 0.0
            for j0, k in _starts(comp_tuple):
                alive = inst.survival(j0)
                g_max = max(inst.g[jj] for jj in range(j0, j0 + k))
                peak = sum(inst.p[jj] + inst.g[jj]
                           for jj in range(j0, j0 + k))
                span += alive * g_max * st
                occ += alive * peak * g_max * st
            occ += resident * span
        lam_m = inst.cap / occ if occ > 0 else float("inf")

        lam = min(lam_c, lam_b, lam_m)
        which = {lam_c: "compute", lam_b: "bandwidth",
                 lam_m: "memory"}[lam]
        if best is None or lam > best["lam"]:
            best = dict(lam=lam, binding=which, cohort=m,
                        pre_tokens=pre, dec_tokens=dec)
    return best
