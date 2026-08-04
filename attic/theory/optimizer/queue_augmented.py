"""The exact queue-augmented occupation LP (revised paper sec. 'The exact
queue-augmented occupation LP'), implemented for small task-first instances
as a validation tool.

The state is x = (q1, q2): bounded host-queue inventories of stage-1 and
stage-2 requests for a single length type with n = 2 filters. An action
(c1, c2, e) consumes c1 stage-1 and c2 stage-2 requests (both must be
present), admits e new documents into the stage-1 queue, and reveals
Z ~ Binomial(c1, s1) survivors, so q' = (q1 - c1 + e, q2 - c2 + Z).
Actions whose queues could overflow under any outcome are infeasible
(no silent drops). Costs come from the shared cost model.

The optimum lambda_hat is the maximum average completion rate of this
bounded-queue control model and must satisfy lambda_hat <= lambda* of the
expected-flow relaxation (paper eq. augmented-relaxation-order); the test
suite checks that ordering.
"""

from math import comb
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from docengine.costmodel import Op, batch_stats, tau
from docengine.instance import Instance


def _binom_pmf(k, n, p):
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def solve_queue_augmented_taskfirst(inst: Instance, d: int, Q: Tuple[int, int],
                                    c_max: int = 3, e_max: int = 3):
    assert inst.n == 2
    s1 = inst.s[0]
    p1, p2 = inst.p
    resident_map = {("prompt", 1): p1, ("prompt", 2): p2}
    pins = p1 + p2

    def action_cost(c1, c2):
        ops = []
        for _ in range(c1):
            ops.append(Op(kind="doc_chunk", doc=-1, stage=1, new=d,
                          cached_resident=p1, read_blocks=(("prompt", 1),),
                          ephemeral_tail=1))
        for _ in range(c2):
            ops.append(Op(kind="doc_chunk", doc=-1, stage=2, new=d,
                          cached_resident=p2, read_blocks=(("prompt", 2),),
                          ephemeral_tail=1))
        st = batch_stats(ops, resident_map)
        return tau(inst.model, inst.device, st)

    states = [(q1, q2) for q1 in range(Q[0] + 1) for q2 in range(Q[1] + 1)]
    s_idx = {x: i for i, x in enumerate(states)}
    cols = []          # (x, c1, c2, e, tau, complete, admit, trans{x': p})
    for x in states:
        q1, q2 = x
        for c1 in range(min(c_max, q1) + 1):
            for c2 in range(min(c_max, q2) + 1):
                if c1 == 0 and c2 == 0:
                    continue
                if q2 - c2 + c1 > Q[1]:
                    continue          # a full-survival outcome would overflow
                for e in range(min(e_max, Q[0] - (q1 - c1)) + 1):
                    trans: Dict = {}
                    for z in range(c1 + 1):
                        pr = _binom_pmf(z, c1, s1)
                        nxt = (q1 - c1 + e, q2 - c2 + z)
                        trans[nxt] = trans.get(nxt, 0.0) + pr
                    complete = c2 + (1 - s1) * c1
                    cols.append((x, c1, c2, e, action_cost(c1, c2),
                                 complete, e, trans))

    n_c = len(cols)
    n_s = len(states)
    # rows: state balance (drop last) + source mix; vars: z_cols
    n_eq = (n_s - 1) + 1
    A_eq = lil_matrix((n_eq, n_c))
    b_eq = np.zeros(n_eq)
    A_ub = lil_matrix((1, n_c))
    obj = np.zeros(n_c)
    for k, (x, c1, c2, e, t, h, adm, trans) in enumerate(cols):
        if s_idx[x] < n_s - 1:
            A_eq[s_idx[x], k] += 1.0
        for x2, pr in trans.items():
            if s_idx[x2] < n_s - 1:
                A_eq[s_idx[x2], k] -= pr
        # source mix: admissions = b * completions with b = 1 (one type)
        A_eq[n_eq - 1, k] = adm - h
        A_ub[0, k] = t
        obj[k] = -h
    res = linprog(obj, A_ub=A_ub.tocsr(), b_ub=np.array([1.0]),
                  A_eq=A_eq.tocsr(), b_eq=b_eq,
                  bounds=[(0, None)] * n_c, method="highs")
    if not res.success:
        return 0.0, res.message
    return float(-res.fun), "optimal"
