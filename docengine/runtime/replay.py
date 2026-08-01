"""LP-guided replay: convert expected-flow rates into a finite validated
schedule (revised paper sec. 'From the LP to a finite 10,000-document
execution').

The replay is an ordinary continuous-batching loop. It quantizes the real
cache inventory to the nearest LP state, prefers the action with the largest
state-conditioned deficit mu*(a|s) * visits(s) - uses(s,a) among actions
whose inputs are actually ready, executes it with REAL documents and REAL
lengths, applies the realized outcomes, and emits validator-format manifest
records. Batches are tagged with a fixed phase rule:

  fill    batches before any second-stage work is ready (prompt pinning,
          first prefill wave),
  core    batches that execute an LP-supported action at 50 percent or more
          of its target token volume,
  repair  batches from the fixed safe fallback (state outside LP support,
          or target action not ready),
  drain   underfull batches after the admission stream is exhausted.

The realized latency sum(tau_t) is the constructed finite latency; its
signed difference from N / lambda* is a deviation, not a guaranteed
overhead (the fluid target is not a finite-N bound)."""

from collections import deque
from typing import Dict, List

import numpy as np

from ..costmodel import kv_capacity_tokens
from ..instance import Instance
from ..optimizer.state_actions import LengthType, MethodModel
from ..optimizer.steady_state_lp import LPResult
from ..sched.blockwise import _BatchBuilder


def _type_index(types: List[LengthType]):
    exact = {t.d: i for i, t in enumerate(types)} \
        if all(t.lo == t.hi for t in types) else None

    def idx(d):
        if exact is not None:
            return exact[d]
        for i, t in enumerate(types):
            if t.lo <= d <= t.hi:
                return i
        return len(types) - 1
    return idx


class _Mixer:
    """Deficit tracker over the LP mixture, grouped by state."""

    def __init__(self, model: MethodModel, lp: LPResult):
        self.by_state: Dict[object, list] = {}
        for a in model.actions:
            y = lp.y.get(a.key, 0.0)
            if y > 0:
                self.by_state.setdefault(a.state, []).append([a, y, 0])
        for s, rows in self.by_state.items():
            tot = sum(r[1] for r in rows)
            for r in rows:
                r[1] /= tot
        self.visits: Dict[object, int] = {}

    def pick(self, state, ready) -> object:
        rows = self.by_state.get(state)
        if not rows:
            return None
        self.visits[state] = self.visits.get(state, 0) + 1
        n_s = self.visits[state]
        best, score = None, -1e18
        for r in rows:
            if not ready(r[0]):
                continue
            sc = r[1] * n_s - r[2]
            if sc > score:
                best, score = r, sc
        if best is None:
            return None
        best[2] += 1
        return best[0]


def replay(inst: Instance, X, model: MethodModel, lp: LPResult):
    if model.method == "task":
        return _replay_task(inst, X, model, lp)
    if model.method == "fullspec":
        return _replay_spec(inst, X, model, lp)
    if model.method == "pipe":
        return _replay_pipe(inst, X, model, lp)
    raise ValueError(model.method)


def _phases_init():
    return dict(fill=[], core=[], repair=[], drain=[])


# ---------------------------------------------------------------- task-first

def _replay_task(inst: Instance, X, model: MethodModel, lp: LPResult):
    n = inst.n
    idx = _type_index(model.types)
    queues: Dict[tuple, deque] = {}
    for i in range(inst.N):
        queues.setdefault((idx(inst.d[i]), 1), deque()).append(i)
    mixer = _Mixer(model, lp)
    records, phases = [], _phases_init()
    pins = set()
    t = 1

    b = _BatchBuilder(inst, t, "task", 0)
    for j in range(1, n + 1):
        b.add_prompt(j)
        pins.add(j)
    records.append(b.finish([0] * inst.N, pins, {}))
    phases["fill"].append(t)

    def remaining():
        return sum(len(q) for q in queues.values())

    while remaining():
        t += 1
        resident = sum(inst.p[j - 1] for j in pins)
        b = _BatchBuilder(inst, t, "task", resident)

        def ready(a):
            return any(queues.get((bb, j)) and c > 0
                       for (bb, j), c in _task_counts(a))
        act = mixer.pick("S0", ready)
        target_tokens = act.U if act else 0
        outcomes = {}
        exec_tokens = 0
        stages_used = set()
        plan = _task_counts(act) if act else \
            [((bb, j), len(q)) for (bb, j), q in queues.items()]
        promote = []      # outcomes reveal at the batch END (gate rule)
        for (bb, j), c in plan:
            q = queues.get((bb, j))
            take = min(c, len(q) if q else 0)
            for _ in range(take):
                i = q.popleft()
                stages_used.add(j)
                b.read_resident(("prompt", j), inst.p[j - 1])
                b.add_chunk(i, 0, inst.d[i], cached=inst.p[j - 1], stage=j)
                exec_tokens += inst.d[i]
                passed = bool(X[i][j - 1])
                outcomes[str(i)] = 1 if passed else 0
                if passed and j < n:
                    promote.append(((bb, j + 1), i))
        for key, i in promote:
            queues.setdefault(key, deque()).append(i)
        if b.U == 0:
            raise RuntimeError("task replay: empty batch")
        records.append(b.finish([0] * inst.N, pins, outcomes))
        if act is None:
            phases["repair"].append(t)
        elif exec_tokens >= 0.5 * target_tokens:
            phases["core"].append(t)
        else:
            phases["drain"].append(t)
    return records, phases


def _task_counts(a):
    return [((int(k.split(",")[0]), int(k.split(",")[1])), c)
            for k, c in a.detail["counts"].items()]


# ----------------------------------------------------------- full speculation

def _replay_spec(inst: Instance, X, model: MethodModel, lp: LPResult):
    n = inst.n
    idx = _type_index(model.types)
    queues: Dict[int, deque] = {}
    for i in range(inst.N):
        queues.setdefault(idx(inst.d[i]), deque()).append(i)
    mixer = _Mixer(model, lp)
    records, phases = [], _phases_init()
    t = 0
    while any(queues.values()):
        t += 1
        b = _BatchBuilder(inst, t, "spec", 0)

        def ready(a):
            return any(queues.get(int(k)) and c > 0
                       for k, c in a.detail["counts"].items())
        act = mixer.pick("S0", ready)
        target_tokens = act.U if act else 0
        outcomes = {}
        exec_tokens = 0
        plan = [(int(k), c) for k, c in act.detail["counts"].items()] \
            if act else [(bb, len(q)) for bb, q in queues.items()]
        for bb, c in plan:
            q = queues.get(bb)
            take = min(c, len(q) if q else 0)
            for _ in range(take):
                i = q.popleft()
                b.add_chunk(i, 0, inst.d[i], cached=0)
                for j in range(1, n + 1):
                    b.add_branch(i, j, inst.d[i], 0)
                exec_tokens += inst.d[i] + sum(inst.p)
                passes = 0
                for j in range(n):
                    if X[i][j]:
                        passes += 1
                    else:
                        break
                outcomes[str(i)] = passes
        if b.U == 0:
            raise RuntimeError("spec replay: empty batch")
        records.append(b.finish([0] * inst.N, set(), outcomes))
        if act is None:
            phases["repair"].append(t)
        elif exec_tokens >= 0.5 * target_tokens:
            phases["core"].append(t)
        else:
            phases["drain"].append(t)
    return records, phases


# ------------------------------------------------------------------ pipeline

def _replay_pipe(inst: Instance, X, model: MethodModel, lp: LPResult):
    assert inst.n == 2
    cap = kv_capacity_tokens(inst.model, inst.device)
    quantum = model.meta["quantum"]
    cap_docs = model.meta["cap_docs"]
    rho = model.meta["rho"]
    p1, p2 = inst.p
    mixer = _Mixer(model, lp)
    records, phases = [], _phases_init()

    new = deque(range(inst.N))
    unc: deque = deque()          # evicted survivors awaiting F2 recompute
    resident: deque = deque()     # doc ids with full doc KV awaiting F2
    t = 0
    admitted_done = False
    while new or unc or resident:
        t += 1
        if t > 20 * inst.N + 100:
            raise RuntimeError("pipe replay failed to terminate")
        m = len(resident)
        state = min(cap_docs, (m // quantum) * quantum)
        res_tokens = sum(inst.d[i] for i in resident)
        b = _BatchBuilder(inst, t, "pipe", res_tokens)

        def ready(a):
            det = a.detail
            if det["v"] > 0 and m == 0:
                return False
            if det["u"] > 0 and not new:
                return False
            if det["rec"] > 0 and not unc:
                return False
            return True
        act = mixer.pick(state, ready)
        fallback = act is None
        if fallback:
            frac, rec_share = 1.0, (0.5 if unc else 0.0)
        else:
            frac = act.detail["v"] / max(state, 1) if state else 0.0
            rec_share = 0.5 if act.detail["rec"] > 0 else 0.0

        outcomes = {}
        room = rho * cap - res_tokens
        v_int = int(round(m * frac)) if m else 0
        drained = []
        for _ in range(v_int):
            i = resident.popleft()
            drained.append(i)
            b.read_resident(("doc", i), inst.d[i])
            b.add_branch(i, 2, inst.d[i], inst.d[i])
            outcomes[str(i)] = 1 if X[i][1] else 0
            room -= p2
        rec_used = 0
        if rec_share > 0:
            budget = room * rec_share
            while unc and budget >= inst.d[unc[0]] + p2:
                i = unc.popleft()
                budget -= inst.d[i] + p2
                room -= inst.d[i] + p2
                b.add_chunk(i, 0, inst.d[i], cached=0)
                b.add_branch(i, 2, inst.d[i], 0)
                outcomes[str(i)] = 1 if X[i][1] else 0
                rec_used += 1
        survivors = []
        admitted = 0
        while new and room >= inst.d[new[0]] + p1:
            i = new.popleft()
            room -= inst.d[i] + p1
            b.add_chunk(i, 0, inst.d[i], cached=0)
            b.add_branch(i, 1, inst.d[i], 0)
            outcomes[str(i)] = 1 if X[i][0] else 0
            admitted += 1
            if X[i][0]:
                survivors.append(i)
        if b.U == 0:
            # nothing fits: evict half the resident backlog for recompute
            n_evict = max(1, m // 2)
            for _ in range(n_evict):
                unc.append(resident.pop())
            if records:
                keep = {i: inst.d[i] for i in resident}
                records[-1]["retained_r"] = \
                    [keep.get(i, 0) for i in range(inst.N)]
            t -= 1
            continue

        # retention: keep survivors up to the capacity share
        for i in survivors:
            if sum(inst.d[j] for j in resident) + inst.d[i] <= rho * cap:
                resident.append(i)
            else:
                unc.append(i)
        retained = [0] * inst.N
        for i in resident:
            retained[i] = inst.d[i]
        records.append(b.finish(retained, set(), outcomes))
        if not new and not admitted:
            phases["drain"].append(t)
        elif fallback:
            phases["repair"].append(t)
        elif t <= 1 or (v_int == 0 and rec_used == 0 and m == 0):
            phases["fill"].append(t)
        else:
            phases["core"].append(t)
    return records, phases
