"""The N=10k constructors must never beat the exact optimum on instances
small enough to solve exactly. Split out of tests/test_sched.py because
the reference solver lives in the attic; the validator-only constructor
checks stayed behind."""

import numpy as np
import pytest

from docengine.sched.blockwise import schedule_blockwise, schedule_taskfirst
from docengine.validator.check import validate

from attic.theory.reference.offline import solve_offline

from test_exact_reference import inst


def total_tau(records):
    return sum(r["tau"] for r in records)


@pytest.mark.parametrize("bits", [0b1111, 0b0101, 0b0000, 0b1010])
def test_constructors_vs_exact(bits):
    X = np.array([[(bits >> 0) & 1, (bits >> 1) & 1],
                  [(bits >> 2) & 1, (bits >> 3) & 1]])
    it = inst([3, 2], [1, 1], [0.5, 0.5], kv_tokens=30)
    # task-first
    recs = schedule_taskfirst(it, X)
    assert validate(it, "task", recs, X) == []
    v_opt, _ = solve_offline(it, "task", X, evict_mode="binary")
    assert total_tau(recs) >= v_opt - 1e-12
    # blockwise k=1 (pipeline) and k=2 (full spec for n=2)
    for k, policy in ((1, "pipe"), (2, "spec")):
        recs = schedule_blockwise(it, X, k)
        assert validate(it, policy, recs, X, kmax=k) == []
        v_opt, _ = solve_offline(it, policy, X, kmax=k, evict_mode="binary")
        assert total_tau(recs) >= v_opt - 1e-12
