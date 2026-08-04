"""Checks for the blocked streaming client scheduler, against a stub
engine that answers from planted flags."""

import asyncio

import numpy as np

from docengine.runtime.engine_client import (EngineTags, run_filter_chain,
                                             run_query)


class _Out:
    def __init__(self, ids, text):
        self.prompt_token_ids = list(ids)
        self.num_cached_tokens = 0
        o = type("O", (), {})()
        o.text = text
        self.outputs = [o]


class StubEngine:
    """Answers request '...-<doc>-<stage0>' from a planted flag matrix."""

    def __init__(self, flags):
        self.flags = flags
        self.calls = []
        self.rids = []

    async def generate(self, prompt, sampling_params, request_id,
                       priority=0):
        self.rids.append(request_id)
        _tag, i, j0 = request_id.rsplit("-", 2)
        i, j0 = int(i), int(j0)
        await asyncio.sleep(0)
        if j0 >= len(self.flags[0]):        # end-of-run flush request
            yield _Out(prompt["prompt_token_ids"], "YES")
            return
        self.calls.append((i, j0))
        text = "YES" if self.flags[i][j0] else "NO"
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


def test_engine_tags_protocol():
    """With tags on, each document's first request carries a pin
    directive, releases ride on later requests, outcomes are unchanged,
    and the run ends with a release-all flush."""
    flags, body_ids, q_ids = _setup(25, 3, seed=7)
    ref = asyncio.run(run_filter_chain(StubEngine(flags), None, body_ids,
                                       q_ids, budget_tokens=10 ** 6))
    eng = StubEngine(flags)
    res = asyncio.run(run_filter_chain(eng, None, body_ids, q_ids,
                                       budget_tokens=10 ** 6,
                                       tags=EngineTags()))
    assert res["survivors"] == ref["survivors"]
    assert res["answers"] == ref["answers"]
    pins = [r for r in eng.rids if "|p" in r]
    assert len(pins) == 25                      # one pin per document
    for r in pins:
        assert r.startswith("de1|p") and "|d" in r
    released = set()
    for r in eng.rids:
        for part in r.split("|"):
            if part.startswith("r") and len(part) > 1 and part != "r*":
                released.update(part[1:].split(","))
    assert "r*" in eng.rids[-1].split("|")      # flush is the last call
    assert released <= {str(i) for i in range(25)}


YES_TOK, NO_TOK = 111, 222


class _ChainOut:
    def __init__(self, prompt_ids, snapshots):
        self.prompt_token_ids = list(prompt_ids)
        self.num_cached_tokens = 0
        o = type("O", (), {})()
        o.token_ids = list(snapshots)
        self.outputs = [o]


class ChainStubEngine:
    """Speaks the chain protocol: accepts the registration request,
    then plays each document's whole chain from the planted flags as
    one stream of growing snapshots (the engine-side rewind is
    invisible to the client, which only sees the record grow)."""

    def __init__(self, flags):
        self.flags = flags
        self.rids = []

    async def generate(self, prompt, sampling_params, request_id,
                       priority=0):
        self.rids.append(request_id)
        ids = prompt["prompt_token_ids"]
        await asyncio.sleep(0)
        if "|reg|" in request_id:
            yield _ChainOut(ids, [NO_TOK])
            return
        i = int(request_id.split("|")[2][1:])
        toks = []
        for j in range(len(self.flags[0])):
            toks.append(YES_TOK if self.flags[i][j] else NO_TOK)
            yield _ChainOut(ids, list(toks))
            if not self.flags[i][j]:
                return


def test_run_query_dispatches_to_chain_mode():
    """Multi-filter queries take the chain path: one request per
    document, a registration first, outcomes from the flags."""
    flags, body_ids, q_ids = _setup(30, 3, seed=11)
    eng = ChainStubEngine(flags)
    res = asyncio.run(run_query(eng, None, body_ids, q_ids,
                                budget_tokens=10 ** 6,
                                yes_ids={YES_TOK}))
    assert res["survivors"] == [i for i in range(30) if all(flags[i])]
    assert res["requests"] == 30
    assert "|reg|" in eng.rids[0]
    assert all("|c|" in r for r in eng.rids[1:])
    for (i, j), a in res["answers"].items():
        assert a == flags[i][j - 1]


def test_run_query_single_filter_uses_requests():
    """One filter has nothing to chain: the tagged request path runs,
    with a pin per document and the end-of-run flush."""
    flags, body_ids, q_ids = _setup(15, 1, seed=13)
    eng = StubEngine(flags)
    res = asyncio.run(run_query(eng, None, body_ids, q_ids,
                                budget_tokens=10 ** 6))
    assert res["survivors"] == [i for i in range(15) if flags[i][0]]
    assert sum(1 for r in eng.rids if "|p" in r) == 15
    assert "r*" in eng.rids[-1].split("|")


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


def test_asymmetric_composition():
    """composition=(1,2,1) gates after filter one, speculates filters
    two and three together, gates filter four."""
    flags, body_ids, q_ids = _setup(30, 4, seed=29)
    eng = StubEngine(flags)
    res = asyncio.run(run_filter_chain(eng, None, body_ids, q_ids,
                                       budget_tokens=10 ** 6,
                                       composition=(1, 2, 1)))
    assert res["survivors"] == [i for i in range(30) if all(flags[i])]
    for i in range(30):
        if flags[i][0]:
            assert (i, 3) in res["answers"]   # block two both asked
        else:
            assert (i, 2) not in res["answers"]
