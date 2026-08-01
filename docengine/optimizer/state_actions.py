"""State-action generators for the expected-flow LP (revised paper secs.
'Empirical length types and host queues' and 'Observable cache states and
feasible actions').

Length types: documents are grouped by token length. 'exact' uses one type
per distinct realized length; 'lower'/'upper' use the bin's smallest or
largest realized length as its representative, giving the optimistic and
conservative endpoints of the paper's length-bracket bound (lambda(d-) >=
lambda(d) >= lambda(d+)).

Host-queue types (method-specific):
  task-first       ('task', b, j)   uncached stage-j request, length bin b
  full speculation ('spec', b)      uncached document, length bin b
  pipeline         ('new', b)       document not yet prefilled
                   ('unc', b, j)    evicted survivor awaiting stage j,
                                    document KV must be recomputed

Cache states: task-first and full speculation with atomic requests keep no
per-row KV across batches, so their cache state collapses to one recurrent
state (paper: 'the model then reduces to the simpler LP over batch
frequencies'); the pinned prompt blocks are charged against capacity. The
pipeline records quantized per-bin counts of resident survivors awaiting
their next filter. Transitions project the binomial survivor count onto the
quantized grid by a mean-preserving two-point distribution per bin; counts
above a bin's cap overflow to the ('unc', b, j) host queue (eviction with
later recompute). The quanta are recorded; results are the optimum of this
restricted expected-flow model (paper eq. restricted-full-order).

Batch statistics follow the same write-through conventions as the cost
model, computed in closed form per composition group (count * per-item
statistics) so that large templates need no per-document op objects. The
replay runtime rebuilds real batches per document and the validator
recomputes everything, so any discrepancy here would surface as a replay
deviation, not silent error.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from ..costmodel import BatchStats, a_pairs, kv_capacity_tokens, tau
from ..instance import Instance


@dataclass(frozen=True)
class LengthType:
    d: int          # representative token length
    mass: float     # fraction of the workload
    count: int      # number of real documents in the bin
    lo: int         # smallest real length in the bin
    hi: int         # largest real length in the bin


def make_types(d_values, n_bins: int, rep: str = "upper") -> List[LengthType]:
    """Quantile bins over the realized lengths. rep in {'exact','lower','upper'}."""
    d_sorted = np.sort(np.asarray(d_values))
    N = len(d_sorted)
    if rep == "exact":
        vals, counts = np.unique(d_sorted, return_counts=True)
        return [LengthType(int(v), c / N, int(c), int(v), int(v))
                for v, c in zip(vals, counts)]
    edges = np.quantile(d_sorted, np.linspace(0, 1, n_bins + 1))
    types = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        sel = (d_sorted >= lo) & ((d_sorted < hi) if b < n_bins - 1
                                  else (d_sorted <= hi))
        vals = d_sorted[sel]
        if len(vals) == 0:
            continue
        rep_d = int(vals[0]) if rep == "lower" else int(vals[-1])
        types.append(LengthType(rep_d, len(vals) / N, int(len(vals)),
                                int(vals[0]), int(vals[-1])))
    return types


def bin_of(types: List[LengthType], d: int) -> int:
    """Bin index of a real document length (types are sorted by length)."""
    for b, t in enumerate(types):
        if t.lo <= d <= t.hi:
            return b
    return min(range(len(types)),
               key=lambda b: min(abs(d - types[b].lo), abs(d - types[b].hi)))


@dataclass
class Action:
    key: str
    state: object                    # source cache state
    tau: float
    U: int
    peak_tokens: int
    consume: Dict[object, float]     # host items consumed per use
    produce: Dict[object, float]     # expected host items produced per use
    complete: float                  # expected documents completed per use
    trans: Dict[object, float]       # P(next cache state | this action)
    detail: dict                     # composition, for replay


@dataclass
class MethodModel:
    method: str
    types: List[LengthType]
    states: List[object]
    actions: List[Action]
    host_types: List[object]
    b: Dict[object, float]           # external input mass per host type
    meta: dict = field(default_factory=dict)


@dataclass
class _Tally:
    """Closed-form batch statistics accumulated per composition group."""
    U: int = 0
    A: int = 0
    K_L: int = 0
    K_W: int = 0
    segments: int = 0

    def add(self, count: int, new: int, cached: int, stored: int,
            segments: int = 1):
        self.U += count * new
        self.A += count * a_pairs(cached, new)
        self.K_W += count * stored
        self.segments += count * segments

    def stats(self) -> BatchStats:
        return BatchStats(U=self.U, A=self.A, K_L=self.K_L, K_W=self.K_W,
                          K_tmp=self.U, n_segments=self.segments)


def _finish(inst: Instance, t: _Tally, resident_tokens: int):
    st = t.stats()
    return tau(inst.model, inst.device, st), st.U, resident_tokens + st.K_tmp


# ---------------------------------------------------------------- task-first

def build_taskfirst(inst: Instance, types: List[LengthType],
                    fill: float = 1.0) -> MethodModel:
    """Single recurrent state; atomic [F_j, D] calls; pinned prompt blocks.

    Per call of length d at stage j: U = d, A = a(p_j, d), stored = d - 1
    (the decision leaf is ephemeral). Each stage's pinned prompt block is
    loaded once per batch that uses it (K_L += p_j, deduplicated)."""
    n = inst.n
    pins_tokens = sum(inst.p)
    cap = kv_capacity_tokens(inst.model, inst.device) - pins_tokens
    S0 = "S0"
    host = [("task", b, j) for b in range(len(types)) for j in range(1, n + 1)]
    bmass = {("task", b, 1): types[b].mass for b in range(len(types))}
    actions: List[Action] = []

    def template(counts: Dict[Tuple[int, int], int], tag: str):
        tly = _Tally()
        consume, produce = {}, {}
        complete = 0.0
        stages_used = set()
        for (b, j), c in counts.items():
            if c <= 0:
                continue
            d, p = types[b].d, inst.p[j - 1]
            tly.add(c, new=d, cached=p, stored=d - 1)
            stages_used.add(j)
            consume[("task", b, j)] = consume.get(("task", b, j), 0) + c
            if j < n:
                produce[("task", b, j + 1)] = \
                    produce.get(("task", b, j + 1), 0) + c * inst.s[j - 1]
                complete += c * (1.0 - inst.s[j - 1])
            else:
                complete += c
        if tly.U == 0:
            return
        tly.K_L = sum(inst.p[j - 1] for j in stages_used)
        t, U, peak = _finish(inst, tly, pins_tokens)
        actions.append(Action(
            key=f"task:{tag}", state=S0, tau=t, U=U, peak_tokens=peak,
            consume=consume, produce=produce, complete=complete,
            trans={S0: 1.0},
            detail=dict(counts={f"{b},{j}": c for (b, j), c in counts.items()
                                if c > 0})))

    for b in range(len(types)):
        for j in range(1, n + 1):
            c = max(1, int(fill * cap) // types[b].d)
            template({(b, j): c}, f"pure-b{b}j{j}")
    pis = [1.0]
    for j in range(1, n):
        pis.append(pis[-1] * inst.s[j - 1])
    weights = {(b, j): types[b].mass * pis[j - 1]
               for b in range(len(types)) for j in range(1, n + 1)}
    tok = sum(w * types[b].d for (b, j), w in weights.items())
    if tok > 0:
        scale = fill * cap / tok
        template({(b, j): max(0, int(w * scale))
                  for (b, j), w in weights.items()}, "mixed")

    return MethodModel(method="task", types=types, states=[S0],
                       actions=actions, host_types=host, b=bmass,
                       meta=dict(cap=cap, pins_tokens=pins_tokens))


# ----------------------------------------------------------- full speculation

def build_fullspec(inst: Instance, types: List[LengthType],
                   fill: float = 1.0) -> MethodModel:
    """Single recurrent state; fused doc + all n branches per document.

    Per job of length d: U = d + sum p_j, A = a(0, d) + sum_j a(d, p_j),
    stored = d + sum_j (p_j - 1); no resident reads."""
    n = inst.n
    cap = kv_capacity_tokens(inst.model, inst.device)
    S0 = "S0"
    host = [("spec", b) for b in range(len(types))]
    bmass = {("spec", b): types[b].mass for b in range(len(types))}
    job_tokens = [types[b].d + sum(inst.p) for b in range(len(types))]
    actions: List[Action] = []

    def template(counts: Dict[int, int], tag: str):
        tly = _Tally()
        consume = {}
        complete = 0.0
        for b, c in counts.items():
            if c <= 0:
                continue
            d = types[b].d
            tly.add(c, new=d, cached=0, stored=d)
            for j in range(1, n + 1):
                tly.add(c, new=inst.p[j - 1], cached=d,
                        stored=inst.p[j - 1] - 1)
            consume[("spec", b)] = c
            complete += c
        if tly.U == 0:
            return
        t, U, peak = _finish(inst, tly, 0)
        actions.append(Action(
            key=f"spec:{tag}", state=S0, tau=t, U=U, peak_tokens=peak,
            consume=consume, produce={}, complete=complete,
            trans={S0: 1.0},
            detail=dict(counts={str(b): c for b, c in counts.items() if c > 0})))

    for b in range(len(types)):
        template({b: max(1, int(fill * cap) // job_tokens[b])}, f"pure-b{b}")
    tok = sum(types[b].mass * job_tokens[b] for b in range(len(types)))
    if tok > 0:
        scale = fill * cap / tok
        template({b: max(0, int(types[b].mass * scale))
                  for b in range(len(types))}, "mixed")

    return MethodModel(method="fullspec", types=types, states=[S0],
                       actions=actions, host_types=host, b=bmass,
                       meta=dict(cap=cap))


# ------------------------------------------------------------------ pipeline

def _two_point(value: float, quantum: int, cap: int):
    """Mean-preserving two-point distribution of `value` on the quantized
    grid {0, q, 2q, ..., cap}; returns ({level: prob}, overflow_mean)."""
    over = max(0.0, value - cap)
    v = min(value, float(cap))
    lo = int(v // quantum) * quantum
    hi = min(lo + quantum, cap)
    if hi == lo:
        return {lo: 1.0}, over
    p_hi = (v - lo) / (hi - lo)
    out = {}
    if p_hi < 1.0:
        out[lo] = 1.0 - p_hi
    if p_hi > 0.0:
        out[hi] = out.get(hi, 0.0) + p_hi
    return out, over


def build_pipeline(inst: Instance, types: List[LengthType],
                   levels: int = 8, rho: float = 0.7,
                   fill: float = 1.0) -> MethodModel:
    """Two-stage pipeline with a one-dimensional quantized survivor state.

    Under the paper's length-independent-selectivity assumption, the length
    mix of F1 survivors equals the admission mix exactly, so the cache state
    needs only the TOTAL count of resident survivors awaiting F2; every
    per-survivor cost uses the exact mass-weighted length mix. Compositions
    are fractional (this is a fluid rate model; integers reappear in the
    replay), so host flows balance exactly with no rounding slack. The
    expected survivor count is projected onto the quantized grid by a
    mean-preserving two-point distribution; counts above the cap overflow to
    the aggregate ('unc',) host queue (eviction with later recompute, also
    at the admission mix). Batch coefficients are expectations over the mix,
    which the paper permits when declared (sec. 'Observable cache states').
    """
    assert inst.n == 2, "pipeline LP generator currently covers n=2"
    s1 = inst.s[0]
    p1, p2 = inst.p
    cap = kv_capacity_tokens(inst.model, inst.device)
    d_mean = sum(t.mass * t.d for t in types)
    a2_mean = sum(t.mass * a_pairs(t.d, p2) for t in types)
    cap_docs = max(1, int(rho * cap / d_mean))
    quantum = max(1, math.ceil(cap_docs / levels))
    cap_docs = max(quantum, (cap_docs // quantum) * quantum)
    states = list(range(0, cap_docs + 1, quantum))
    host = [("new", b) for b in range(len(types))] + [("unc",)]
    bmass = {("new", b): types[b].mass for b in range(len(types))}
    actions: List[Action] = []

    def make_action(m, drain_frac, rec_share, tag):
        resident = m * d_mean
        room = fill * cap - resident
        v = m * drain_frac
        room -= v * p2
        if room < 0:
            return
        rec = 0.0
        if rec_share > 0:
            rec = (room * rec_share) / (d_mean + p2)
            room -= rec * (d_mean + p2)
        u = room / (d_mean + p1)
        if u + v + rec <= 1e-9:
            return
        tly = _Tally()
        # expectations over the exact length mix, scaled by fractional counts
        for t_ in types:
            w = t_.mass
            tly.U += int(round((u + rec) * w * t_.d))
            tly.A += int(round((u + rec) * w * a_pairs(0, t_.d)))
            tly.A += int(round(u * w * a_pairs(t_.d, p1)))
            tly.K_W += int(round((u + rec) * w * t_.d))
        tly.U += int(round(u * p1 + (v + rec) * p2))
        tly.A += int(round((v + rec) * a2_mean))
        tly.K_W += int(round(u * (p1 - 1) + (v + rec) * (p2 - 1)))
        tly.K_L += int(round(v * d_mean))       # resident doc KV per F2 branch
        tly.segments = int(math.ceil(2 * u + v + 2 * rec))
        t, U, peak = _finish(inst, tly, int(resident))
        # per-type rounding of the fluid composition can overshoot by a few
        # tokens; real feasibility is enforced by the replay and validator
        if peak > cap + max(64, len(types)):
            return
        consume = {("new", b): u * types[b].mass
                   for b in range(len(types)) if u > 0}
        if rec > 0:
            consume[("unc",)] = rec
        produce = {}
        complete = v + rec + u * (1.0 - s1)
        target = m - v + s1 * u
        dist, over = _two_point(target, quantum, cap_docs)
        if over > 1e-12:
            produce[("unc",)] = over
        actions.append(Action(
            key=f"pipe:m{m}:{tag}", state=m, tau=t, U=U, peak_tokens=peak,
            consume=consume, produce=produce, complete=complete,
            trans=dict(dist), detail=dict(u=u, v=v, rec=rec)))

    for m in states:
        for frac, ftag in ((1.0, "drainall"), (0.5, "half"), (0.0, "keep")):
            for rshare, rtag in ((0.0, ""), (0.5, "+rec")):
                make_action(m, frac, rshare, f"{ftag}{rtag}")

    return MethodModel(method="pipe", types=types, states=states,
                       actions=actions, host_types=host, b=bmass,
                       meta=dict(cap=cap, cap_docs=cap_docs, quantum=quantum,
                                 rho=rho, levels=levels, d_mean=d_mean))


def _product(lists):
    if not lists:
        yield ()
        return
    for head in lists[0]:
        for rest in _product(lists[1:]):
            yield (head,) + tuple(rest)


# --------------------------------------- n-filter blockwise (lookahead k) LP

def build_blockwise_lp(inst: Instance, types: List[LengthType], k: int,
                       levels: int = 5, rho: float = 0.85,
                       fill: float = 1.0) -> MethodModel:
    """n-filter document-first execution with lookahead-k blocks.

    Filters are grouped into contiguous blocks of k (the last block may be
    shorter). A document's admission runs its prefill fused with block 1;
    survivors of a non-final block wait, with resident document KV, at the
    waiting point after that block; a drain runs their next whole block.
    k = 1 is the strict pipeline, k = n is full speculation.

    The cache state is one quantized survivor count per waiting point
    (blocks minus one dimensions), each with a capacity share proportional
    to its steady inflow. As in build_pipeline, survivor length mix equals
    admission mix under length-independent selectivity, so per-survivor
    costs use the exact mass-weighted mix and compositions are fractional.
    Overflow at waiting point w becomes the ('unc', w) host queue, consumed
    by recompute actions (re-prefill fused with the next block).
    """
    n = inst.n
    blocks = []
    j = 1
    while j <= n:
        blocks.append(tuple(range(j, min(j + k - 1, n) + 1)))
        j += k
    B = len(blocks)
    W = B - 1
    s = inst.s
    rho_b = [float(np.prod([s[jj - 1] for jj in blk])) for blk in blocks]
    P_b = [sum(inst.p[jj - 1] for jj in blk) for blk in blocks]
    nb = [len(blk) for blk in blocks]
    d_mean = sum(t.mass * t.d for t in types)
    a0_mean = sum(t.mass * a_pairs(0, t.d) for t in types)
    a_blk = [sum(t.mass * sum(a_pairs(t.d, inst.p[jj - 1]) for jj in blk)
                 for t in types) for blk in blocks]
    cap = kv_capacity_tokens(inst.model, inst.device)

    if W == 0:
        states = [()]
        quanta, cap_docs = [], []
    else:
        pi_w = []
        acc = 1.0
        for b in range(W):
            acc *= rho_b[b]
            pi_w.append(acc)
        tot = sum(pi_w)
        quanta, cap_docs = [], []
        for w in range(W):
            cb = max(1, int(rho * cap * (pi_w[w] / tot) / d_mean))
            q = max(1, math.ceil(cb / levels))
            quanta.append(q)
            cap_docs.append(max(q, (cb // q) * q))
        grids = [list(range(0, cap_docs[w] + 1, quanta[w])) for w in range(W)]
        states = [tuple(m) for m in _product(grids)]

    host = [("new", b) for b in range(len(types))] + \
        [("unc", w) for w in range(W)]
    bmass = {("new", b): types[b].mass for b in range(len(types))}
    actions: List[Action] = []

    def make_action(state, drain_frac, rec_w, tag, only_w: int = -1):
        m = list(state)
        resident = sum(m[w] * d_mean for w in range(W))
        room = fill * cap - resident
        if only_w >= 0:
            v = [m[w] * drain_frac if w == only_w else 0.0 for w in range(W)]
        else:
            v = [m[w] * drain_frac for w in range(W)]
        room -= sum(v[w] * P_b[w + 1] for w in range(W))
        if room < 0:
            return
        rec = 0.0
        if rec_w is not None:
            rec = (room * 0.5) / (d_mean + P_b[rec_w + 1])
            room -= rec * (d_mean + P_b[rec_w + 1])
        u = room / (d_mean + P_b[0])
        if u + sum(v) + rec <= 1e-9:
            return
        tly = _Tally()
        for t_ in types:
            w_ = t_.mass
            tly.U += int(round((u + rec) * w_ * t_.d))
            tly.K_W += int(round((u + rec) * w_ * t_.d))
        tly.A += int(round(u * (a0_mean + a_blk[0])))
        tly.U += int(round(u * P_b[0]))
        tly.K_W += int(round(u * (P_b[0] - nb[0])))
        for w in range(W):
            b = w + 1
            tly.U += int(round(v[w] * P_b[b]))
            tly.A += int(round(v[w] * a_blk[b]))
            tly.K_W += int(round(v[w] * (P_b[b] - nb[b])))
            tly.K_L += int(round(v[w] * d_mean))
        if rec > 0:
            b = rec_w + 1
            tly.U += int(round(rec * P_b[b]))
            tly.A += int(round(rec * (a0_mean + a_blk[b])))
            tly.K_W += int(round(rec * (P_b[b] - nb[b])))
        tly.segments = int(math.ceil(u * (1 + nb[0]) + sum(v) * 2 + rec * 2))
        t, U, peak = _finish(inst, tly, int(resident))
        if peak > cap + max(64, len(types)):
            return
        consume = {("new", b): u * types[b].mass
                   for b in range(len(types)) if u > 0}
        if rec > 0:
            consume[("unc", rec_w)] = rec
        produce = {}
        # completions and inflows to the next waiting point
        complete = u * (1.0 - rho_b[0]) if B > 1 else u
        inflow = [0.0] * W
        if W > 0:
            inflow[0] = u * rho_b[0]
        for w in range(W):
            b = w + 1
            amt = v[w] + (rec if rec_w == w else 0.0)
            if b == B - 1:
                complete += amt                      # final block: all done
            else:
                complete += amt * (1.0 - rho_b[b])
                inflow[b] += amt * rho_b[b]
        per_dim = []
        for w in range(W):
            target = m[w] - v[w] + inflow[w]
            dist, over = _two_point(target, quanta[w], cap_docs[w])
            per_dim.append(dist)
            if over > 1e-12:
                produce[("unc", w)] = produce.get(("unc", w), 0.0) + over
        trans: Dict = {}
        if W == 0:
            trans[()] = 1.0
        else:
            for combo in _product([list(dd.items()) for dd in per_dim]):
                nxt = tuple(level for level, _pr in combo)
                pr = 1.0
                for _level, p in combo:
                    pr *= p
                if pr > 1e-12:
                    trans[nxt] = trans.get(nxt, 0.0) + pr
        actions.append(Action(
            key=f"blk{k}:{'-'.join(map(str, state))}:{tag}", state=state,
            tau=t, U=U, peak_tokens=peak, consume=consume, produce=produce,
            complete=complete, trans=trans,
            detail=dict(u=u, v=list(v), rec=rec, rec_w=rec_w)))

    for state in states:
        for frac, ftag in ((1.0, "drainall"), (0.5, "half"), (0.0, "keep")):
            make_action(state, frac, None, ftag)
        for w in range(W):
            make_action(state, 1.0, w, f"drainall+rec{w}")
            if W > 1:
                make_action(state, 1.0, None, f"drain-w{w}", only_w=w)

    return MethodModel(method=f"blockwise{k}", types=types, states=states,
                       actions=actions, host_types=host, b=bmass,
                       meta=dict(cap=cap, cap_docs=cap_docs, quanta=quanta,
                                 rho=rho, levels=levels, k=k, blocks=blocks,
                                 d_mean=d_mean))
