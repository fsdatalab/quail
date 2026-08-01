"""The N=10k constructors must emit validator-clean schedules that never beat
the exact optimum on instances small enough to solve exactly."""

import numpy as np
import pytest

from docengine.exact.offline import solve_offline
from docengine.instance import Instance
from docengine.sched.blockwise import schedule_blockwise, schedule_taskfirst
from docengine.validator.check import validate

from test_exact import TOY_MODEL, inst, toy_device


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


def test_constructors_tight_memory():
    """Capacity forces multi-batch schedules with retention/eviction."""
    X = np.array([[1, 1], [1, 0], [0, 1], [1, 1]])
    it = inst([4, 3, 2, 4], [1, 1], [0.7, 0.7], kv_tokens=7)
    recs = schedule_taskfirst(it, X)
    assert validate(it, "task", recs, X) == []
    for k in (1, 2):
        recs = schedule_blockwise(it, X, k)
        policy = "pipe" if k == 1 else "spec"
        assert validate(it, policy, recs, X, kmax=k) == [], recs
        assert len(recs) > 2


def test_constructor_medium_random():
    """Bigger random instance, no exact reference; just validity + coverage."""
    rng = np.random.default_rng(7)
    d = rng.integers(2, 12, size=25).tolist()
    it = inst(d, [2, 1, 2], [0.6, 0.5, 0.7], kv_tokens=40, delta=2)
    X = (rng.random((25, 3)) < 0.6).astype(np.int8)
    recs = schedule_taskfirst(it, X)
    assert validate(it, "task", recs, X) == []
    for k in (1, 2, 3):
        policy = "pipe" if k == 1 else "spec"
        recs = schedule_blockwise(it, X, k)
        assert validate(it, policy, recs, X, kmax=k) == []
