"""The executor obeys the plan: mode, budget, pins, decode window."""

import asyncio

from docengine.exec import run_plan
from docengine.plan import plan_query
from docengine.configs import DEVICES, MODELS
from tests.test_client import (ChainStubEngine, StubEngine, YES_TOK,
                               _setup)


def _plan(n, body_ids, **kw):
    return plan_query(n, [len(b) for b in body_ids],
                      MODELS["Qwen3-4B-FP8"],
                      DEVICES["H100-SXM-80GB"], **kw)


def test_run_plan_chain_dispatch():
    flags, body_ids, q_ids = _setup(20, 3, seed=21)
    p = _plan(3, body_ids)
    assert p.mode == "chain"
    eng = ChainStubEngine(flags)
    res = asyncio.run(run_plan(eng, None, body_ids, q_ids, p,
                               yes_ids={YES_TOK}))
    assert res["survivors"] == [i for i in range(20) if all(flags[i])]
    assert res["requests"] == 20


def test_run_plan_single_filter_requests():
    flags, body_ids, q_ids = _setup(10, 1, seed=23)
    p = _plan(1, body_ids)
    assert p.mode == "requests"
    eng = StubEngine(flags)
    res = asyncio.run(run_plan(eng, None, body_ids, q_ids, p))
    assert res["survivors"] == [i for i in range(10) if flags[i][0]]
