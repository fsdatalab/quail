"""Constructive feasible schedules for large N (paper sec. 8.6 claim 2).

Two constructors, both emitting validator-format manifest records:

  schedule_taskfirst(inst, X)      -- stage waves under the [F_j][D_i] template
  schedule_blockwise(inst, X, k)   -- document-first with lookahead-k blocks:
                                      k=1 is the pipeline, k=n is full
                                      speculation, 1<k<n is partial speculation

Both constructions are ONLINE-implementable: no batch uses an outcome revealed
at its own end, and eviction after batch t depends only on outcomes revealed
by batch t. Evaluated under a realized X they therefore upper-bound both
OPT_off(d, X) and, averaged over coupled scenarios, the online policy value.

Packing: batches are filled to the free-KV capacity (K_tmp <= cap - resident,
eq. 22) streaming docs in workload order, splitting a document at chunk-
quantum boundaries when it straddles a batch (paper sec. 8.1). Under tau_0
larger batches never cost more (D and H are maxima of additive terms), so
capacity-filling is the right packing shape; gaps to LB_res are reported,
not assumed away. When retention pressure leaves a batch with no feasible
work (high survival on a small-capacity device), the scheduler evicts part
of the branch-owing backlog and recomputes those documents later -- the
eviction/recompute path of the model, not a failure.
"""

from ..costmodel import attn_time, dense_time, kv_capacity_tokens
from ..instance import Instance, survival


def _a(c, q):
    return c * q + q * (q + 1) // 2


def _tau_fields(inst, U, A, K_L, K_W, peak_tokens):
    m = inst.model
    D = dense_time(m, inst.device, U)
    H = attn_time(m, inst.device, A, K_L, K_W)
    return dict(U=U, A=A, K_L=K_L, K_W=K_W,
                M_peak=m.W_mem + m.kappa * peak_tokens, D=D, H=H, tau=D + H)


def _floor_delta(x, delta):
    return (x // delta) * delta


class _BatchBuilder:
    def __init__(self, inst, t, policy, resident_start):
        self.inst = inst
        self.rec = dict(t=t, policy=policy, ops=[])
        self.U = self.A = self.K_W = self.K_tmp = 0
        self.reads = {}
        self.resident_start = resident_start

    def room(self, cap):
        return cap - self.resident_start - self.K_tmp

    def add_prompt(self, j):
        p = self.inst.p[j - 1]
        self.rec["ops"].append(dict(kind="prompt_prefill", doc=-1, stage=j,
                                    token_start=0, token_end=p))
        self.U += p; self.A += _a(0, p); self.K_W += p; self.K_tmp += p

    def add_chunk(self, i, start, end, cached, stage=0):
        self.rec["ops"].append(dict(kind="doc_chunk", doc=i, stage=stage,
                                    token_start=start, token_end=end))
        q = end - start
        self.U += q; self.A += _a(cached, q); self.K_W += q; self.K_tmp += q
        if stage > 0 and end == self.inst.d[i]:
            self.K_W -= 1   # task decision leaf is ephemeral (write-through)

    def add_branch(self, i, j, d_i, resident_prefix):
        p = self.inst.p[j - 1]
        self.rec["ops"].append(dict(kind="branch", doc=i, stage=j,
                                    token_start=0, token_end=p))
        # interior branch tokens are stored (write-through); the final
        # decision position is the only ephemeral leaf
        self.U += p; self.A += _a(d_i, p); self.K_W += p - 1; self.K_tmp += p
        if resident_prefix > 0:
            self.reads[("doc", i)] = resident_prefix

    def read_resident(self, block, tokens):
        self.reads[block] = tokens

    def finish(self, retained_r, pins, outcomes):
        K_L = sum(self.reads.values())
        self.rec.update(_tau_fields(self.inst, self.U, self.A, K_L, self.K_W,
                                    self.resident_start + self.K_tmp))
        self.rec["K_tmp"] = self.K_tmp
        self.rec["retained_r"] = list(retained_r)
        self.rec["retained_pins"] = sorted(pins)
        self.rec["outcomes"] = outcomes
        return self.rec


def schedule_taskfirst(inst: Instance, X, cap: int = None):
    """Stage waves; wave j prefills [F_j][D_i] for every doc with Y_ij = 1.
    Task-doc KV is consumed by its own completion, so nothing but the pinned
    p_j prompt blocks survives a wave. `cap` overrides the analytic KV
    capacity, for driving a real engine whose pool size is measured."""
    Y = survival(X)
    cap = cap if cap is not None else kv_capacity_tokens(inst.model,
                                                         inst.device)
    records = []
    pins = set()
    t = 0
    for j in range(1, inst.n + 1):
        queue = [(i, 0) for i in range(inst.N) if Y[i][j - 1]]
        if not queue:
            break
        first = True
        while queue:
            t += 1
            resident = sum(inst.p[jj - 1] for jj in pins) \
                + sum(pr for _i, pr in queue if pr > 0)
            b = _BatchBuilder(inst, t, "task", resident)
            if first:
                b.add_prompt(j)
                pins.add(j)
                first = False
            else:
                b.read_resident(("prompt", j), inst.p[j - 1])
            carry = []
            while queue:
                i, prog = queue[0]
                need = inst.d[i] - prog
                room = b.room(cap)
                take = need if need <= room else max(
                    0, _floor_delta(room, inst.delta))
                if take <= 0:
                    break
                queue.pop(0)
                if prog > 0:
                    b.read_resident(("taskdoc", i), prog)
                b.add_chunk(i, prog, prog + take,
                            cached=inst.p[j - 1] + prog, stage=j)
                if take < need:
                    carry.append((i, prog + take))
                    break                    # batch is full
            queue = carry + queue
            if not b.rec["ops"] or (first is False and b.U == 0):
                raise RuntimeError("task-first: no progress in batch")
            outcomes = {}
            retained = [0] * inst.N
            for i, prog in queue:
                retained[i] = prog
            for op in b.rec["ops"]:
                i = op["doc"]
                if i >= 0 and op["token_end"] == inst.d[i]:
                    outcomes[str(i)] = 1 if X[i][j - 1] else 0
            records.append(b.finish(retained, pins, outcomes))
    return records


def schedule_blockwise(inst: Instance, X, k: int, cap: int = None):
    """Document-first with contiguous speculative blocks of size k
    (k=1 pipeline, k=n full speculation). `cap` overrides the analytic KV
    capacity, for driving a real engine whose pool size is measured."""
    Y = survival(X)  # noqa: F841  (outcomes are read directly from X below)
    cap = cap if cap is not None else kv_capacity_tokens(inst.model,
                                                         inst.device)
    policy = "pipe" if k == 1 else "spec"
    records = []
    t = 0
    stream = [(i, 1) for i in range(inst.N - 1, -1, -1)]  # (doc, block start), pop from end
    pending = []          # (doc, block_start): full doc KV resident, owed branches
    partial = None        # (doc, block_start, progress): straddling prefill
    guard = 0

    def branch_tokens(j0):
        kk = min(k, inst.n - j0 + 1)
        return kk, sum(inst.p[jj - 1] for jj in range(j0, j0 + kk))

    while stream or pending or partial is not None:
        t += 1
        guard += 1
        if guard > 20 * inst.N + 200:
            raise RuntimeError("blockwise scheduler failed to terminate")
        resident = sum(inst.d[i] for i, _ in pending) \
            + (partial[2] if partial else 0)
        b = _BatchBuilder(inst, t, policy, resident)
        completed = []                     # (doc, first_stage, kk)
        still_pending = []

        # 1) branches owed to docs retained from earlier batches
        for i, j0 in pending:
            kk, btoks = branch_tokens(j0)
            if b.room(cap) >= btoks:
                b.read_resident(("doc", i), inst.d[i])
                for jj in range(j0, j0 + kk):
                    b.add_branch(i, jj, inst.d[i], inst.d[i])
                completed.append((i, j0, kk))
            else:
                still_pending.append((i, j0))

        # 2) finish a straddling document
        if partial is not None:
            i, j0, prog = partial
            need = inst.d[i] - prog
            room = b.room(cap)
            take = need if need <= room else max(
                0, _floor_delta(room, inst.delta))
            if take > 0:
                b.read_resident(("doc", i), prog)
                b.add_chunk(i, prog, prog + take, cached=prog)
                if take == need:
                    kk, btoks = branch_tokens(j0)
                    if b.room(cap) >= btoks:
                        for jj in range(j0, j0 + kk):
                            b.add_branch(i, jj, inst.d[i], prog)
                        completed.append((i, j0, kk))
                    else:
                        still_pending.append((i, j0))
                    partial = None
                else:
                    partial = (i, j0, prog + take)

        # 3) stream new (or recomputing) docs to capacity
        while partial is None and stream:
            i, j0 = stream[-1]
            kk, btoks = branch_tokens(j0)
            room = b.room(cap)
            if room < min(inst.delta, inst.d[i] + btoks):
                break
            stream.pop()
            take = inst.d[i] if inst.d[i] + btoks <= room else max(
                0, _floor_delta(room, inst.delta))
            take = min(take, inst.d[i])
            if take <= 0:
                stream.append((i, j0))
                break
            b.add_chunk(i, 0, take, cached=0)
            if take == inst.d[i]:
                if b.room(cap) >= btoks:
                    for jj in range(j0, j0 + kk):
                        b.add_branch(i, jj, inst.d[i], 0)
                    completed.append((i, j0, kk))
                else:
                    still_pending.append((i, j0))
            else:
                partial = (i, j0, take)

        # 4) starvation fallback: evict part of the branch-owing backlog for
        # later recompute (the model's eviction/recompute path). The eviction
        # belongs to the PREVIOUS batch boundary, so amend that record's
        # retained_r; no phantom empty batch is emitted.
        if not b.rec["ops"]:
            if not records or not still_pending:
                raise RuntimeError("blockwise: no feasible work and no backlog"
                                   " to evict (capacity too small)")
            n_evict = max(1, len(still_pending) // 2)
            evicted = still_pending[-n_evict:]
            pending = still_pending[:-n_evict]
            stream.extend(reversed(evicted))   # recompute later, same block
            for i, _j0 in evicted:
                records[-1]["retained_r"][i] = 0
            t -= 1
            continue

        # 5) outcomes + retention
        outcomes = {}
        for i, j0, kk in completed:
            passes = 0
            for off in range(kk):
                if X[i][j0 - 1 + off]:
                    passes += 1
                else:
                    break
            outcomes[str(i)] = passes
            if passes == kk and j0 + kk <= inst.n:
                still_pending.append((i, j0 + kk))
        pending = still_pending
        retained = [0] * inst.N
        for i, _j0 in pending:
            retained[i] = inst.d[i]
        if partial is not None:
            retained[partial[0]] = partial[2]
        records.append(b.finish(retained, set(), outcomes))
    return records
