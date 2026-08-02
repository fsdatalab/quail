"""Exact solvers for reasoning filters (small N certification layer).

The state is per document and integer, decisions are made at event
boundaries, and randomness enters through filter outcomes exactly as
in the paper: thinking lengths and document lengths are known, a
stage-j call passes with probability s_j. Two solvers over the same
transition system:

  solve_offline(inst, X, family)   outcomes known (coupled scenario);
                                   deterministic shortest path by
                                   memoized recursion
  solve_online(inst, family)       outcomes revealed at call
                                   completion; expectimax value
                                   iteration

Families restrict the action set of the same solver: "task" re-reads
the task template plus document for every call and retains nothing;
a stage composition tuple runs the document-first template launching
each composition block's calls together; "opt" is the unrestricted
document-first family (adaptive speculation: any number of next
stages may be launched at any boundary).

Semantics, chosen to mirror the engine and the fluid layer: documents
prefill atomically; a call generates g_j tokens, one per step, and all
active calls advance together; an epoch runs to the next call
completion, during which chosen pending prefill work is co-scheduled;
epoch cost is the max of compute and bandwidth seconds over the whole
epoch; memory admission is by peak footprint (document plus p + g per
active call).

State per document:
  ("P",)            not prefilled (document-first) or between calls
                    (task family, nothing retained)
  ("I", j)          resident, next stage j (1-indexed), no active call
  ("C", branches)   active call(s); branches is a tuple of
                    (stage, remaining_tokens)
  ("D",)            dead or finished
"""

from .model import t_pre

_MEMO_LIMIT = 500_000


def _epoch_cost(inst, active_ctx, U):
    """Cost of one epoch: advance every active call by delta = min
    remaining tokens, co-scheduling U prefill tokens. active_ctx is a
    tuple of (context_now, remaining) per active call."""
    mdl, dev = inst.model, inst.device
    if not active_ctx:
        if U == 0:
            return 0.0, 0
        return max(t_pre(inst, U),
                   mdl.kappa * U / dev.BW), 0
    delta = min(r for _, r in active_ctx)
    m = len(active_ctx)
    comp = 2.0 * mdl.P * (m * delta + U) / inst.R_C
    attn = sum(delta * c + delta * (delta - 1) / 2.0
               for c, _ in active_ctx)
    bw = (delta * mdl.W_run
          + mdl.kappa * (attn + U + m * delta)) / dev.BW
    return max(comp, bw), delta


def _footprint(inst, doc_state, family):
    """Peak KV tokens this document holds in this state."""
    kind = doc_state[0]
    if kind in ("P", "D"):
        return 0.0
    resident = 0.0 if family == "task" else inst.d
    if kind == "I":
        return resident
    extra = sum(inst.p[j - 1] + inst.g[j - 1]
                for j, _r in doc_state[1])
    if family == "task":
        j = doc_state[1][0][0]
        extra += inst.fj(j - 1) + inst.d
    return resident + extra


def _launch_options(inst, j, family):
    """Sets of stages a document at next-stage j may launch together."""
    n = inst.n
    if family == "task":
        return [(j,)]
    if family == "opt":
        return [tuple(range(j, j + k)) for k in range(1, n - j + 2)]
    # fixed composition: j must begin one of its blocks
    pos = 1
    for k in family:
        if pos == j:
            return [tuple(range(j, j + k))]
        pos += k
    raise AssertionError(f"stage {j} not a block start of {family}")


def _ctx0(inst, family, j):
    base = inst.d + inst.p[j - 1]
    if family == "task":
        base += inst.fj(j - 1)
    return base


class _Solver:
    def __init__(self, inst, family, X=None):
        self.inst = inst
        self.family = family
        self.X = X                       # None: online (expectimax)
        self.memo = {}

    def prefill_tokens(self, launches):
        """Prefill tokens charged when `launches` (doc, stages) start."""
        toks = 0.0
        for _i, stages, needs_doc in launches:
            for j in stages:
                toks += self.inst.p[j - 1]
                if self.family == "task":
                    toks += self.inst.fj(j - 1) + self.inst.d
            if needs_doc:
                toks += self.inst.d
        return toks

    def value(self, state):
        if all(s[0] == "D" for s in state):
            return 0.0
        key = state
        if key in self.memo:
            return self.memo[key]
        if len(self.memo) > _MEMO_LIMIT:
            raise RuntimeError("exact solver state space exceeded the "
                               "memo limit; shrink N, n, or g")
        best = float("inf")
        for nxt, cost in self.transitions(state):
            v = cost + (self.expect(nxt) if self.X is None
                        else self.value(nxt))
            if v < best:
                best = v
        self.memo[key] = best
        return best

    def expect(self, state):
        """State may contain completed calls awaiting outcomes (marked
        by remaining == 0); branch on them."""
        for i, s in enumerate(state):
            if s[0] == "C" and any(r == 0 for _j, r in s[1]):
                branches = list(s[1])
                for idx, (j, r) in enumerate(branches):
                    if r == 0:
                        j0 = j
                        branches.pop(idx)
                        break
                rest = tuple(branches)
                sj = self.inst.s[j0 - 1]
                out_pass = self.resolve(state, i, j0, rest, True)
                out_fail = self.resolve(state, i, j0, rest, False)
                return sj * self.expect(out_pass) \
                    + (1 - sj) * self.expect(out_fail)
        return self.value(state)

    def resolve(self, state, i, j0, rest, passed):
        lst = list(state)
        if not passed:
            lst[i] = ("D",)              # dead; in-flight branches wasted
        elif rest:
            lst[i] = ("C", rest)
        else:
            nj = max(j for j, _ in state[i][1]) + 1
            if nj > self.inst.n:
                lst[i] = ("D",)          # survived every stage
            elif self.family == "task":
                lst[i] = ("P", nj)
            else:
                lst[i] = ("I", nj)
        return tuple(lst)

    def resolve_known(self, state, i):
        """Offline: apply known outcomes to completed calls of doc i."""
        s = state[i]
        done = sorted(j for j, r in s[1] if r == 0)
        rest = tuple((j, r) for j, r in s[1] if r > 0)
        lst = list(state)
        for j in done:
            if not self.X[i][j - 1]:
                lst[i] = ("D",)
                return tuple(lst)
        if rest:
            lst[i] = ("C", rest)
            return tuple(lst)
        nj = max(done) + 1
        if nj > self.inst.n:
            lst[i] = ("D",)
        elif self.family == "task":
            lst[i] = ("P", nj)
        else:
            lst[i] = ("I", nj)
        return tuple(lst)

    def transitions(self, state):
        inst, family = self.inst, self.family
        N = len(state)
        # documents that can launch now
        launchable = []
        for i, s in enumerate(state):
            if s[0] == "I":
                launchable.append((i, s[1], False))
            elif s[0] == "P":
                nj = s[1] if len(s) > 1 else 1
                launchable.append((i, nj, family != "task"))
        active = [(i, s) for i, s in enumerate(state) if s[0] == "C"]

        options = [[]]
        for i, j, needs_doc in launchable:
            new = []
            for base in options:
                new.append(base)                       # do not launch
                for stages in _launch_options(inst, j, family):
                    new.append(base + [(i, stages, needs_doc)])
            options = new

        out = []
        for launches in options:
            if not launches and not active:
                continue
            # memory admission by peak footprint
            occ = sum(_footprint(inst, s, family) for s in state)
            for _i, stages, needs_doc in launches:
                occ += (inst.d if (needs_doc or family == "task") else 0.0)
                occ += sum(inst.p[j - 1] + inst.g[j - 1] for j in stages)
                if family == "task":
                    occ += inst.fj(stages[0] - 1)
            if occ > inst.cap:
                continue
            lst = list(state)
            for i, stages, _nd in launches:
                lst[i] = ("C", tuple((j, inst.g[j - 1]) for j in stages))
            U = self.prefill_tokens(launches)
            ctxs = []
            for i, s in enumerate(lst):
                if s[0] == "C":
                    for j, r in s[1]:
                        ctxs.append((_ctx0(inst, family, j)
                                     + (inst.g[j - 1] - r), r))
            cost, delta = _epoch_cost(inst, tuple(ctxs), U)
            if delta == 0 and not launches:
                continue
            nstate = []
            for s in lst:
                if s[0] == "C":
                    nstate.append(("C", tuple((j, r - delta)
                                              for j, r in s[1])))
                else:
                    nstate.append(s)
            nstate = tuple(nstate)
            if self.X is not None:
                for i, s in enumerate(nstate):
                    if s[0] == "C" and any(r == 0 for _j, r in s[1]):
                        nstate = self.resolve_known(nstate, i)
            out.append((nstate, cost))
        return out


def _start(inst, family):
    if family == "task":
        return tuple(("P", 1) for _ in range(inst.N))
    return tuple(("P",) for _ in range(inst.N))


def solve_offline(inst, X, family):
    s = _Solver(inst, family, X=X)
    return s.value(_start(inst, family))


def solve_online(inst, family):
    s = _Solver(inst, family, X=None)
    return s.expect(_start(inst, family))
