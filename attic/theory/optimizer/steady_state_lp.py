"""The steady-state expected-flow LP (revised paper sec. 'The linear
program').

Variables: y_a >= 0, the fluid rate (uses per second) of each state-action
column, and lambda >= 0, the admitted-and-completed document rate.

  maximize    lambda
  subject to  lambda * b_r + sum_a y_a (vbar_{a,r} - u_{a,r}) = 0   (host flow)
              sum_{a from s} y_a - sum_a y_a P(s|a) = 0             (cache flow)
              sum_a y_a tau_a <= 1                                  (GPU time)
              lambda - sum_a y_a hbar_a = 0                         (completion)

The optimum lambda* is the throughput upper bound of the supplied restricted
expected-flow model; N / lambda* is the asymptotic fluid latency target,
not a finite-N bound (paper sec. 'What the optimum proves'). One redundant
cache-flow row is dropped (the rows sum to zero). Residuals of the returned
solution against every constraint are reported so the checker contract in
Appendix B can be met without re-deriving the model.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from .state_actions import MethodModel


@dataclass
class LPResult:
    lam: float                  # lambda*, documents per second
    y: Dict[str, float]         # action key -> rate (uses per second)
    residuals: dict
    status: str
    meta: dict


def solve_expected_flow(model: MethodModel) -> LPResult:
    acts = model.actions
    n_a = len(acts)
    host = list(model.host_types)
    states = list(model.states)
    h_idx = {r: i for i, r in enumerate(host)}
    s_idx = {s: i for i, s in enumerate(states)}
    n_h, n_s = len(host), len(states)

    # variable order: [y_0..y_{n_a-1}, lambda]
    n_var = n_a + 1
    # equalities: host flow rows + cache flow rows (drop last) + completion
    cache_rows = max(0, n_s - 1)
    n_eq = n_h + cache_rows + 1
    A_eq = lil_matrix((n_eq, n_var))
    b_eq = np.zeros(n_eq)

    for k, a in enumerate(acts):
        for r, c in a.consume.items():
            A_eq[h_idx[r], k] -= c
        for r, c in a.produce.items():
            A_eq[h_idx[r], k] += c
        si = s_idx[a.state]
        if si < cache_rows:
            A_eq[n_h + si, k] += 1.0
        for s2, pr in a.trans.items():
            i2 = s_idx[s2]
            if i2 < cache_rows:
                A_eq[n_h + i2, k] -= pr
        A_eq[n_eq - 1, k] = -a.complete
    for r, mass in model.b.items():
        A_eq[h_idx[r], n_a] = mass
    A_eq[n_eq - 1, n_a] = 1.0

    A_ub = lil_matrix((1, n_var))
    for k, a in enumerate(acts):
        A_ub[0, k] = a.tau
    b_ub = np.array([1.0])

    c = np.zeros(n_var)
    c[n_a] = -1.0
    res = linprog(c, A_ub=A_ub.tocsr(), b_ub=b_ub,
                  A_eq=A_eq.tocsr(), b_eq=b_eq,
                  bounds=[(0, None)] * n_var, method="highs")
    if not res.success:
        return LPResult(lam=0.0, y={}, residuals={}, status=res.message,
                        meta=dict(n_actions=n_a, n_states=n_s))
    yvec = res.x[:n_a]
    lam = res.x[n_a]

    # residual report (recomputed, not taken from the solver)
    eq_res = A_eq.tocsr() @ res.x - b_eq
    time_used = float(sum(a.tau * yvec[k] for k, a in enumerate(acts)))
    residuals = dict(
        host_flow=float(np.abs(eq_res[:n_h]).max()) if n_h else 0.0,
        cache_flow=float(np.abs(eq_res[n_h:n_h + cache_rows]).max())
        if cache_rows else 0.0,
        completion=float(abs(eq_res[-1])),
        gpu_time=time_used,
    )
    y = {acts[k].key: float(yvec[k]) for k in range(n_a) if yvec[k] > 1e-12}
    return LPResult(lam=float(lam), y=y, residuals=residuals, status="optimal",
                    meta=dict(n_actions=n_a, n_states=n_s,
                              method=model.method))
