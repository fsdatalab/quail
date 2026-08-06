"""The N=10k constructors must emit validator-clean schedules. The
comparison against the deleted theory program's exact reference
optimum is in git history with the program."""

import numpy as np

from docengine.sched.blockwise import schedule_blockwise, schedule_taskfirst
from docengine.validator.check import validate

from test_exact import inst


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
