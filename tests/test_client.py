"""Checks for the blocked streaming client scheduler, against a stub
engine that answers from planted flags."""

import asyncio

import numpy as np

from quail.runtime.engine_client import (EngineTags, run_filter_chain,
                                         run_filter_chain_engine)


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
    def __init__(self, prompt_ids, snapshots, cached=0):
        self.prompt_token_ids = list(prompt_ids)
        self.num_cached_tokens = cached
        o = type("O", (), {})()
        o.token_ids = list(snapshots)
        self.outputs = [o]


class ChainStubEngine:
    """Speaks the chain protocol: accepts the registration request,
    then plays each document's whole chain from the planted flags as
    one stream of growing snapshots (the engine-side rewind is
    invisible to the client, which only sees the record grow). A
    failed stage ends the stream, as the gate does in the engine."""

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
        parts = request_id.split("|")[1:-1]
        i = int(next(p[1:] for p in parts
                     if p.startswith("d") and len(p) > 1))
        toks = []
        for j in range(len(self.flags[0])):
            toks.append(YES_TOK if self.flags[i][j] else NO_TOK)
            yield _ChainOut(ids, list(toks))
            if not self.flags[i][j]:
                return


def test_chain_mode_runs_one_request_per_document():
    """Chain mode: a registration first, then one request per
    document that carries the whole filter chain, with outcomes from
    the flags and no document asked past a failure."""
    flags, body_ids, q_ids = _setup(30, 3, seed=11)
    eng = ChainStubEngine(flags)
    res = asyncio.run(run_filter_chain_engine(eng, None, body_ids, q_ids,
                                              budget_tokens=10 ** 6,
                                              yes_ids={YES_TOK}))
    assert res["survivors"] == [i for i in range(30) if all(flags[i])]
    assert res["requests"] == 30
    assert "|reg|" in eng.rids[0]
    assert all("|c|" in r for r in eng.rids[1:])
    for (i, j), a in res["answers"].items():
        assert a == flags[i][j - 1]
        assert all(flags[i][:j - 1])       # never asked past a failure
