"""Checks for the synchronous step-driven client, against a stub
engine that answers from planted flags."""

import numpy as np

from quail.runtime.engine_client import (EngineTags, run_filter_chain,
                                         run_filter_chain_engine)


class _Out:
    def __init__(self, rid, ids, text):
        self.request_id = rid
        self.finished = True
        self.prompt_token_ids = list(ids)
        self.num_cached_tokens = 0
        o = type("O", (), {})()
        o.text = text
        o.token_ids = []
        self.outputs = [o]


class StubEngine:
    """Answers request '...-<doc>-<stage0>' from a planted flag matrix.
    Every queued request finishes on the next step, so the client's
    admit-step-route loop runs many rounds on a tiny budget."""

    def __init__(self, flags):
        self.flags = flags
        self.calls = []
        self.rids = []
        self.request_params = {}
        self._queue = []

    def add_request(self, request_id, prompt, sampling_params, priority=0):
        self.rids.append(request_id)
        self.request_params[request_id] = sampling_params
        _tag, i, j0 = request_id.rsplit("-", 2)
        i, j0 = int(i), int(j0)
        if j0 < len(self.flags[0]):
            self.calls.append((i, j0))
        self._queue.append((request_id, prompt, i, j0))

    def step(self):
        out, self._queue = [
            _Out(rid, prompt["prompt_token_ids"],
                 "YES" if j0 >= len(self.flags[0]) or self.flags[i][j0]
                 else "NO")
            for rid, prompt, i, j0 in self._queue], []
        return out


def _setup(N, n, seed=3):
    rng = np.random.default_rng(seed)
    flags = (rng.random((N, n)) < 0.6).astype(int)
    body_ids = [[100 + i] * int(rng.integers(5, 15)) for i in range(N)]
    q_ids = [[j] * 3 for j in range(n)]
    return flags, body_ids, q_ids


def test_streaming_chain_outcomes():
    flags, body_ids, q_ids = _setup(40, 3)
    eng = StubEngine(flags)
    res = run_filter_chain(eng, None, body_ids, q_ids,
                           budget_tokens=10 ** 6)
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
    res = run_filter_chain(eng, None, body_ids, q_ids, budget_tokens=1)
    assert set(res["answers"]) >= {(i, 1) for i in range(12)}
    firsts = [i for i, j0 in eng.calls if j0 == 0]
    assert firsts == sorted(firsts)


def test_engine_tags_protocol():
    """With tags on, each document's first request carries a pin
    directive, releases ride on later requests, outcomes are unchanged,
    and the run ends with a release-all flush."""
    flags, body_ids, q_ids = _setup(25, 3, seed=7)
    ref = run_filter_chain(StubEngine(flags), None, body_ids, q_ids,
                           budget_tokens=10 ** 6)
    eng = StubEngine(flags)
    res = run_filter_chain(eng, None, body_ids, q_ids,
                           budget_tokens=10 ** 6, tags=EngineTags())
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


class _SP:
    """Stands in for SamplingParams: copyable, with extra_args."""

    def __init__(self):
        self.extra_args = None


def test_short_documents_opt_out_of_a_capped_store():
    """With a store length threshold, requests for short documents
    carry max_offload_tokens=0 and long documents' requests keep the
    original params untouched."""
    flags, body_ids, q_ids = _setup(20, 2, seed=13)
    eng = StubEngine(flags)
    sp = _SP()
    run_filter_chain(eng, sp, body_ids, q_ids, budget_tokens=10 ** 6,
                     store_min_tokens=10)
    for rid, params in eng.request_params.items():
        _tag, i, j0 = rid.rsplit("-", 2)
        i, j0 = int(i), int(j0)
        if j0 >= len(flags[0]):
            continue
        if len(body_ids[i]) < 10:
            kv = params.extra_args["kv_transfer_params"]
            assert kv["max_offload_tokens"] == 0
        else:
            assert params is sp and sp.extra_args is None


YES_TOK, NO_TOK = 111, 222


class _ChainOut:
    def __init__(self, rid, prompt_ids, snapshot, finished):
        self.request_id = rid
        self.finished = finished
        self.prompt_token_ids = list(prompt_ids)
        self.num_cached_tokens = 0
        o = type("O", (), {})()
        o.token_ids = list(snapshot)
        self.outputs = [o]


class ChainStubEngine:
    """Speaks the chain protocol: finishes the registration request at
    once, then plays each document's chain one stage per step as a
    growing snapshot (the engine-side rewind is invisible to the
    client, which only sees the record grow). A failed stage ends the
    chain, as the gate does in the engine."""

    def __init__(self, flags):
        self.flags = flags
        self.rids = []
        self._chains = {}            # rid -> (prompt_ids, doc, stage)

    def add_request(self, request_id, prompt, sampling_params, priority=0):
        self.rids.append(request_id)
        ids = prompt["prompt_token_ids"]
        if "|reg|" in request_id:
            self._chains[request_id] = (ids, None, 0)
            return
        parts = request_id.split("|")[1:-1]
        i = int(next(p[1:] for p in parts
                     if p.startswith("d") and len(p) > 1))
        self._chains[request_id] = (ids, i, 0)

    def step(self):
        out = []
        for rid in list(self._chains):
            ids, i, stage = self._chains[rid]
            if i is None:                       # registration
                out.append(_ChainOut(rid, ids, [NO_TOK], True))
                del self._chains[rid]
                continue
            n = len(self.flags[0])
            toks = [YES_TOK if self.flags[i][j] else NO_TOK
                    for j in range(stage + 1)]
            done = not self.flags[i][stage] or stage + 1 == n
            out.append(_ChainOut(rid, ids, toks, done))
            if done:
                del self._chains[rid]
            else:
                self._chains[rid] = (ids, i, stage + 1)
        return out


def test_chain_mode_runs_one_request_per_document():
    """Chain mode: a registration first, then one request per
    document that carries the whole filter chain, with outcomes from
    the flags and no document asked past a failure."""
    flags, body_ids, q_ids = _setup(30, 3, seed=11)
    eng = ChainStubEngine(flags)
    res = run_filter_chain_engine(eng, None, body_ids, q_ids,
                                  yes_ids={YES_TOK})
    assert res["survivors"] == [i for i in range(30) if all(flags[i])]
    assert res["requests"] == 30
    assert "|reg|" in eng.rids[0]
    assert all("|c|" in r for r in eng.rids[1:])
    for (i, j), a in res["answers"].items():
        assert a == flags[i][j - 1]
        assert all(flags[i][:j - 1])       # never asked past a failure
