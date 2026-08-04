"""Checks for the blocked streaming client scheduler, against a stub
engine that answers from planted flags."""

import asyncio

import numpy as np

from docengine.runtime.engine_client import (EngineTags, run_filter_chain,
                                             run_filter_chain_engine,
                                             run_query, run_shared_scan)


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


class SharedChainStubEngine:
    """Speaks the shared-scan protocol: per-query registrations (Q
    parts), chain requests answered from that query's planted flags,
    and the scheduler's memory story mimicked client-visibly - a
    document prefills cold once, stays resident while its pin's
    release mentions are outstanding, and later chains for it report
    the body as cached tokens."""

    def __init__(self, flags_by_query, body_lens):
        self.flags = flags_by_query      # one (N, n_k) matrix per query
        self.body_lens = body_lens
        self.rids = []
        self.registered = set()          # query ids seen registering
        self.resident = {}               # doc -> release mentions left
        self.admissions = {}             # doc -> cold prefill count
        self.pins = 0
        self.frees = 0
        self.star_freed = 0              # docs release-all had to free
        self.bad_release = 0             # releases for absent docs

    def _release(self, keys):
        for key in keys:
            if key == "*":
                self.star_freed += len(self.resident)
                self.resident.clear()
                continue
            doc = int(key)
            if doc not in self.resident:
                self.bad_release += 1
                continue
            self.resident[doc] -= 1
            if self.resident[doc] == 0:
                del self.resident[doc]
                self.frees += 1

    async def generate(self, prompt, sampling_params, request_id,
                       priority=0):
        self.rids.append(request_id)
        ids = prompt["prompt_token_ids"]
        parts = request_id.split("|")[1:-1]
        qid = next((p[1:] for p in parts
                    if p.startswith("Q") and len(p) > 1), "0")
        if "reg" in parts:     # dispatched before directive parsing,
            self.registered.add(qid)  # like the scheduler's add_request
            await asyncio.sleep(0)
            yield _ChainOut(ids, [NO_TOK])
            return
        doc, uses, rel = None, 1, []
        for part in parts:
            if part.startswith("d") and len(part) > 1:
                doc = int(part[1:])
            elif part.startswith("u") and part[1:].isdigit():
                uses = int(part[1:])
            elif part.startswith("r") and len(part) > 1:
                rel = ["*"] if part[1:] == "*" else part[1:].split(",")
        self._release(rel)
        if "c" not in parts:               # release-only flush request
            await asyncio.sleep(0)
            yield _ChainOut(ids, [NO_TOK])
            return
        assert qid in self.registered, "chain before its registration"
        if doc in self.resident:
            cached = self.body_lens[doc]
        else:
            cached = 0
            self.resident[doc] = uses
            self.admissions[doc] = self.admissions.get(doc, 0) + 1
            self.pins += 1
        row = self.flags[int(qid)][doc]
        toks = []
        for f in row:
            toks.append(YES_TOK if f else NO_TOK)
            await asyncio.sleep(0)
            yield _ChainOut(ids, list(toks), cached=cached)
            if not f:
                return


def _setup_shared(N, ns, seed):
    """Distinct flag matrices and question sets per query (query k may
    have its own filter count ns[k])."""
    rng = np.random.default_rng(seed)
    flags = [(rng.random((N, n)) < 0.7).astype(int) for n in ns]
    body_ids = [[100 + i] * int(rng.integers(5, 15)) for i in range(N)]
    all_q = [[[50 * (k + 1) + j] * 3 for j in range(n)]
             for k, n in enumerate(ns)]
    return flags, body_ids, all_q


def _shared_against_alone(N, ns, seed):
    """Run the shared scan, then each query alone in chain mode, and
    check outcomes, single admission, and pin/free balance."""
    flags, body_ids, all_q = _setup_shared(N, ns, seed)
    nq = len(ns)
    queries = [dict(q_ids=all_q[k], yes_ids={YES_TOK})
               for k in range(nq)]
    eng = SharedChainStubEngine(flags, [len(b) for b in body_ids])
    res = asyncio.run(run_shared_scan(eng, None, body_ids, queries,
                                      budget_tokens=10 ** 6))
    for k in range(nq):
        ref = asyncio.run(run_filter_chain_engine(
            ChainStubEngine(flags[k]), None, body_ids, all_q[k],
            10 ** 6, {YES_TOK}, tag=f"a{k}"))
        assert res["queries"][k]["survivors"] == ref["survivors"]
        assert res["queries"][k]["answers"] == ref["answers"]
        assert res["queries"][k]["wall"] <= res["wall"]
    # one admission per document: the body prefilled once, then served
    # from cache to every later query's chain
    assert eng.admissions == {i: 1 for i in range(N)}
    corpus = sum(len(b) for b in body_ids)
    assert res["cached_tokens"] == (nq - 1) * corpus
    assert res["read_multiplier"] >= 1.0
    # pins and frees balanced: every pin got exactly one release per
    # query, and the end-of-run release-all found nothing left
    assert eng.pins == N and eng.frees == N
    assert eng.resident == {} and eng.star_freed == 0
    assert eng.bad_release == 0
    assert eng.registered == {str(k) for k in range(nq)}
    return eng, res


def test_shared_scan_two_queries():
    eng, res = _shared_against_alone(40, [3, 3], seed=17)
    chains = [r for r in eng.rids if "|c|" in r]
    assert len(chains) == 80                 # one chain per doc, per query
    assert all("|p" in r and "|u2|" in r for r in chains)
    regs = [r for r in eng.rids if "|reg|" in r]
    assert sorted(p for r in regs for p in r.split("|")[1:-1]
                  if p.startswith("Q")) == ["Q0", "Q1"]
    assert "r*" in eng.rids[-1].split("|")   # flush is the last call


def test_shared_scan_three_queries():
    """Three queries with different filter counts share one pass."""
    eng, res = _shared_against_alone(35, [3, 2, 4], seed=23)
    chains = [r for r in eng.rids if "|c|" in r]
    assert len(chains) == 105
    assert all("|u3|" in r for r in chains)
    assert res["requests"] == 105


def test_shared_scan_single_query_matches_chain_mode():
    """q=1 is plain chain mode with a pin per document released once;
    no consumer-count part is emitted (the default of one applies)."""
    eng, res = _shared_against_alone(20, [3], seed=31)
    assert all("|u" not in r for r in eng.rids if "|c|" in r)


def test_shared_scan_tiny_budget_completes():
    """A budget below two documents' cost still finishes one at a
    time; outcomes and the balance invariants are unchanged."""
    flags, body_ids, all_q = _setup_shared(12, [2, 3], seed=37)
    queries = [dict(q_ids=all_q[k], yes_ids={YES_TOK}) for k in range(2)]
    eng = SharedChainStubEngine(flags, [len(b) for b in body_ids])
    res = asyncio.run(run_shared_scan(eng, None, body_ids, queries,
                                      budget_tokens=1))
    for k in range(2):
        want = [i for i in range(12) if all(flags[k][i])]
        assert res["queries"][k]["survivors"] == want
    assert eng.admissions == {i: 1 for i in range(12)}
    assert eng.pins == 12 and eng.frees == 12
    assert eng.resident == {} and eng.star_freed == 0
