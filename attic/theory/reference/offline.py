"""Offline exact solver: on-demand Dijkstra over scheduler states
(paper sec. 8.4). Valid with evict/recompute cycles because every nonempty
batch has positive cost."""

import heapq
from typing import Optional

from docengine.instance import Instance

from . import engine


def solve_offline(inst: Instance, policy: str, X, kmax: int = 1,
                  evict_mode: str = "suffix", max_states: int = 2_000_000):
    """Minimum-makespan schedule for realized outcomes X.

    Returns (value, schedule) where schedule is a list of
    (batch, stats, cost, outcomes, post_eviction_state) records suitable for
    manifest emission. X is an N x n 0/1 matrix (row i = doc i)."""
    inst.check()
    start = engine.initial_state(inst)
    dist = {start: 0.0}
    pred: dict = {start: None}
    done = set()
    heap = [(0.0, start)]
    target = None

    while heap:
        c0, state = heapq.heappop(heap)
        if state in done:
            continue
        done.add(state)
        if engine.terminal(inst, state):
            target = state
            break
        if len(dist) > max_states:
            raise RuntimeError("state explosion: raise delta or shrink instance")
        for b in engine.candidate_batches(inst, policy, kmax, state):
            ev = engine.evaluate_batch(inst, policy, state, b)
            if ev is None:
                continue
            st, cost, _ops = ev
            comps = engine.completions(inst, policy, state, b)
            (_prob, outcomes), = engine.outcome_branches(inst, comps, X=X)
            mid = engine.apply_batch(inst, policy, state, b, outcomes)
            for nxt in engine.eviction_choices(inst, policy, mid, evict_mode):
                nd = c0 + cost
                if nd < dist.get(nxt, float("inf")) - 1e-15:
                    dist[nxt] = nd
                    pred[nxt] = (state, b, st, cost, outcomes)
                    heapq.heappush(heap, (nd, nxt))

    if target is None:
        raise RuntimeError("no feasible schedule found")

    schedule = []
    s = target
    while pred[s] is not None:
        prev, b, st, cost, outcomes = pred[s]
        schedule.append(dict(batch=b, stats=st, cost=cost,
                             outcomes=outcomes, state_before=prev,
                             state_after=s))
        s = prev
    schedule.reverse()
    return dist[target], schedule
