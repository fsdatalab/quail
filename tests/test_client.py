"""Checks for the blocked streaming client scheduler, against a stub
engine that answers from planted flags."""

import asyncio

import numpy as np

from docengine.runtime.engine_client import run_filter_chain


class _Out:
    def __init__(self, ids, text):
        self.prompt_token_ids = list(ids)
        self.num_cached_tokens = 0
        o = type("O", (), {})()
        o.text = text
        self.outputs = [o]


class StubEngine:
    """Answers request '<tag>-<doc>-<stage0>' from a planted flag matrix."""

    def __init__(self, flags):
        self.flags = flags
        self.calls = []

    async def generate(self, prompt, sampling_params, request_id):
        _tag, i, j0 = request_id.rsplit("-", 2)
        self.calls.append((int(i), int(j0)))
        await asyncio.sleep(0)
        text = "YES" if self.flags[int(i)][int(j0)] else "NO"
        yield _Out(prompt["prompt_token_ids"], text)


def _setup(N, n, seed=3):
    rng = np.random.default_rng(seed)
    flags = (rng.random((N, n)) < 0.6).astype(int)
    body_ids = [[100 + i] * int(rng.integers(5, 15)) for i in range(N)]
    q_ids = [[j] * 3 for j in range(n)]
    return flags, body_ids, q_ids


def test_streaming_chain_outcomes():
    flags, body_ids, q_ids = _setup(40, 3)
    eng = StubEngine(flags)
    res = asyncio.run(run_filter_chain(eng, None, body_ids, q_ids,
                                       budget_tokens=10 ** 6))
    want = [i for i in range(40) if all(flags[i])]
    assert res["survivors"] == want
    for (i, j), a in res["answers"].items():
        assert a == flags[i][j - 1]
        assert all(flags[i][:j - 1])       # never asked past a failure
    assert res["requests"] == len(res["answers"])


def test_tiny_budget_completes():
    """Budget smaller than two documents must still finish (one at a
    time), not deadlock, and admission stays in workload order."""
    flags, body_ids, q_ids = _setup(12, 2, seed=5)
    eng = StubEngine(flags)
    res = asyncio.run(run_filter_chain(eng, None, body_ids, q_ids,
                                       budget_tokens=1))
    assert set(res["answers"]) >= {(i, 1) for i in range(12)}
    firsts = [i for i, j0 in eng.calls if j0 == 0]
    assert firsts == sorted(firsts)


def test_lookahead_waste_recorded():
    """lookahead=2 asks the second question ungated, so a doc failing
    stage 1 still has a stage-2 answer recorded (the wasted branch),
    and docs never advance past a failed block."""
    flags, body_ids, q_ids = _setup(30, 4, seed=9)
    eng = StubEngine(flags)
    res = asyncio.run(run_filter_chain(eng, None, body_ids, q_ids,
                                       budget_tokens=10 ** 6, lookahead=2))
    for i in range(30):
        assert (i, 2) in res["answers"]          # block 1 always both
        if flags[i][0] and flags[i][1]:
            assert (i, 3) in res["answers"]
        else:
            assert (i, 3) not in res["answers"]
    want = [i for i in range(30) if all(flags[i])]
    assert res["survivors"] == want
