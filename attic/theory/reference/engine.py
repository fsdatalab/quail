"""Shared machinery for the exact DP: states, feasible batches, transitions.

State (paper eqs. 38-40, 48-49), identical shape for all policy families:
    state = (zs, rs, pins)
      zs   : tuple, z_i in 1..n+1 (next unresolved stage; n+1 = row finished)
      rs   : tuple, resident prefix progress r_i
             task-first : doc tokens computed under the CURRENT F_{z_i} prefix
             pipe/spec  : doc-prefix tokens (stage-independent)
      pins : frozenset of j whose shared task-prompt block is resident
             (task-first only; doc-first prompt KV is transient, C3)

An action is (batch, eviction); the online solver splits them around the
outcome reveal (eq. 54). A batch is described by:
    chunks   : tuple of (i, delta)          -- doc-token increments
    branches : tuple of (i, k)              -- contiguous F_{z_i}..F_{z_i+k-1}
    prefills : tuple of j                   -- task-prompt block prefills
Policies: 'task' (no branches; completing the doc under F_{z_i} IS the filter
call), 'pipe' (k=1 only), 'spec' (1 <= k <= kmax). Convention C2: offline
keeps structural gates -- z advances only at batch boundaries, so no batch
contains work that presupposes an outcome revealed at its own end.
"""

from dataclasses import dataclass
from itertools import combinations as _combinations_iter, product
from typing import Iterator, Optional

from docengine.configs import DeviceConfig, ModelConfig
from docengine.costmodel import (BatchStats, Op, batch_stats, memory_ok,
                                 peak_memory, tau)
from docengine.instance import Instance

DONE = "done"  # sentinel


@dataclass(frozen=True)
class Batch:
    chunks: tuple      # ((i, delta), ...)
    branches: tuple    # ((i, k), ...)
    prefills: tuple    # (j, ...)

    @property
    def empty(self) -> bool:
        return not (self.chunks or self.branches or self.prefills)


def initial_state(inst: Instance):
    return (tuple([1] * inst.N), tuple([0] * inst.N), frozenset())


def terminal(inst: Instance, state) -> bool:
    zs, _, _ = state
    return all(z == inst.n + 1 for z in zs)


def resident_tokens(inst: Instance, state) -> int:
    zs, rs, pins = state
    return sum(rs) + sum(inst.p[j - 1] for j in pins)


def _doc_options(inst: Instance, policy: str, kmax: int, z: int, r: int, d: int):
    """Per-doc action alternatives: (delta, k). k>0 requires r+delta == d."""
    opts = [(0, 0)]
    if z > inst.n:
        return opts
    remaining = d - r
    if policy == "task":
        for dl in inst.chunk_options(remaining):
            opts.append((dl, 0))
        return opts
    # pipe / spec / fullspec
    if policy == "pipe":
        k_choices = [1]
    elif policy == "fullspec":
        k_choices = [inst.n - z + 1]     # forced full block: X-independent
    else:
        k_choices = list(range(1, min(kmax, inst.n - z + 1) + 1))
    if remaining > 0:
        for dl in inst.chunk_options(remaining):
            opts.append((dl, 0))
            if r + dl == d:
                for k in k_choices:
                    opts.append((dl, k))
    else:
        for k in k_choices:
            opts.append((0, k))
    return opts


def candidate_batches(inst: Instance, policy: str, kmax: int, state,
                      max_actions: int = 2_000_000) -> Iterator[Batch]:
    """Enumerate every feasible next batch (paper sec. 8.3). Feasibility
    filters (memory, caps) are applied by evaluate_batch; this enumerates the
    combinatorial structure."""
    zs, rs, pins = state
    per_doc = [_doc_options(inst, policy, kmax, zs[i], rs[i], inst.d[i])
               for i in range(inst.N)]
    if policy == "task":
        unpinned = [j for j in range(1, inst.n + 1) if j not in pins]
        prefill_sets = [tuple(c) for m in range(len(unpinned) + 1)
                        for c in _combinations(unpinned, m)]
    else:
        prefill_sets = [()]
    count = 0
    for combo in product(*per_doc):
        chunks = tuple((i, dl) for i, (dl, k) in enumerate(combo) if dl > 0)
        branches = tuple((i, k) for i, (dl, k) in enumerate(combo) if k > 0)
        for pf in prefill_sets:
            b = Batch(chunks=chunks, branches=branches, prefills=pf)
            if b.empty:
                continue
            if policy == "task":
                # every chunk needs its prompt block pinned or prefilled now
                need = {zs[i] for i, _ in chunks}
                if not need.issubset(pins | set(pf)):
                    continue
            count += 1
            if count > max_actions:
                raise RuntimeError("action enumeration exceeded max_actions")
            yield b


def _combinations(items, m):
    return _combinations_iter(items, m)


def evaluate_batch(inst: Instance, policy: str, state, b: Batch):
    """Build ops, check feasibility, return (stats, cost, ops) or None."""
    zs, rs, pins = state
    ops = []
    resident_map = {}
    for j in pins:
        resident_map[("prompt", j)] = inst.p[j - 1]
    for i in range(inst.N):
        if rs[i] > 0:
            key = ("taskdoc", i) if policy == "task" else ("doc", i)
            resident_map[key] = rs[i]

    for j in b.prefills:
        ops.append(Op(kind="prompt_prefill", doc=-1, stage=j,
                      new=inst.p[j - 1], cached_resident=0))
    chunk_delta = dict(b.chunks)
    for i, dl in b.chunks:
        z = zs[i]
        if policy == "task":
            pinned = z in pins
            prefilled = z in b.prefills
            reads = []
            if pinned:
                reads.append(("prompt", z))
            if rs[i] > 0:
                reads.append(("taskdoc", i))
            ops.append(Op(kind="doc_chunk", doc=i, stage=z, new=dl,
                          cached_resident=(inst.p[z - 1] if pinned else 0) + rs[i],
                          cached_inbatch=(inst.p[z - 1] if prefilled else 0),
                          read_blocks=tuple(reads),
                          ephemeral_tail=1 if rs[i] + dl == inst.d[i] else 0))
        else:
            reads = [("doc", i)] if rs[i] > 0 else []
            ops.append(Op(kind="doc_chunk", doc=i, stage=0, new=dl,
                          cached_resident=rs[i], cached_inbatch=0,
                          read_blocks=tuple(reads)))
    for i, k in b.branches:
        z = zs[i]
        dl = chunk_delta.get(i, 0)
        if rs[i] + dl != inst.d[i]:
            return None  # branch without complete prefix
        reads = [("doc", i)] if rs[i] > 0 else []
        for jj in range(z, z + k):
            ops.append(Op(kind="branch", doc=i, stage=jj, new=inst.p[jj - 1],
                          cached_resident=rs[i], cached_inbatch=dl,
                          read_blocks=tuple(reads), ephemeral_tail=1))

    st = batch_stats(ops, resident_map)
    if inst.max_new_tokens is not None and st.U > inst.max_new_tokens:
        return None
    if inst.max_seqs is not None and st.n_segments > inst.max_seqs:
        return None
    res = resident_tokens(inst, state)
    if not memory_ok(inst.model, inst.device, res, st.K_tmp):
        return None
    cost = tau(inst.model, inst.device, st)
    return st, cost, ops


def completions(inst: Instance, policy: str, state, b: Batch):
    """Filter evaluations resolved at this batch's end:
    returns list of (doc, first_stage, k)."""
    zs, rs, _ = state
    out = []
    if policy == "task":
        for i, dl in b.chunks:
            if rs[i] + dl == inst.d[i]:
                out.append((i, zs[i], 1))
    else:
        for i, k in b.branches:
            out.append((i, zs[i], k))
    return out


def outcome_branches(inst: Instance, comps, s=None, X=None):
    """Enumerate joint outcome vectors for the completing docs.

    Offline (X given): a single branch with probability 1.
    Online (s given): per completing doc, outcomes are 'fail at offset f'
    (f in 0..k-1) or 'all k pass'; joint = product across docs.
    Yields (prob, {doc: passes}) where passes = number of stages passed
    before the first failure (passes == k means the whole block passed)."""
    if X is not None:
        res = {}
        for i, j, k in comps:
            passes = 0
            for off in range(k):
                if X[i][j - 1 + off]:
                    passes += 1
                else:
                    break
            res[i] = passes
        yield 1.0, res
        return
    per_doc = []
    for i, j, k in comps:
        alts = []
        pacc = 1.0
        for off in range(k):
            sj = s[j - 1 + off]
            alts.append((pacc * (1.0 - sj), (i, off)))       # fail after `off` passes
            pacc *= sj
        alts.append((pacc, (i, k)))                           # all pass
        per_doc.append(alts)
    for combo in product(*per_doc):
        prob = 1.0
        res = {}
        for pr, (i, passes) in combo:
            prob *= pr
            res[i] = passes
        if prob > 0.0:
            yield prob, res


def apply_batch(inst: Instance, policy: str, state, b: Batch, outcomes: dict):
    """Post-batch, post-reveal state BEFORE eviction. outcomes maps completing
    doc -> passes (see outcome_branches)."""
    zs, rs, pins = state
    zs, rs = list(zs), list(rs)
    for j in b.prefills:
        pins = pins | {j}
    for i, dl in b.chunks:
        rs[i] += dl
    comps = completions(inst, policy, state, b)
    for i, j, k in comps:
        passes = outcomes[i]
        if passes == k and j + k <= inst.n:
            zs[i] = j + k
        elif passes == k:                       # passed through stage n
            zs[i] = inst.n + 1
        else:
            zs[i] = inst.n + 1                  # failed inside the block
        if policy == "task":
            rs[i] = 0                           # next request = new prefix
    # dominance: KV of finished rows has no consumer -> force-evict
    for i in range(inst.N):
        if zs[i] == inst.n + 1:
            rs[i] = 0
    if all(z == inst.n + 1 for z in zs):
        pins = frozenset()
    return tuple(zs), tuple(rs), pins


def eviction_choices(inst: Instance, policy: str, state,
                     evict_mode: str = "suffix") -> Iterator:
    """Enumerate post-batch eviction outcomes E_t as resulting states.
    'suffix': each doc prefix may be truncated to any multiple of delta
    (or kept); pins may be dropped in any subset. 'binary': keep-all or
    evict-to-zero per doc; pins kept. 'none': keep everything -- exact only
    when memory can never bind (ample-capacity instances); the caller is
    responsible for that. Finished rows are already force-evicted."""
    zs, rs, pins = state
    if evict_mode == "none":
        yield state
        return
    per_doc = []
    for i in range(inst.N):
        r = rs[i]
        if r == 0:
            per_doc.append([0])
        elif evict_mode == "binary":
            per_doc.append([r, 0])
        else:
            keep = {r, 0}
            keep.update(x for x in range(inst.delta, r, inst.delta))
            per_doc.append(sorted(keep, reverse=True))
    pin_sets = [pins]
    if pins and evict_mode == "suffix":
        pin_sets = [frozenset(c) for m in range(len(pins), -1, -1)
                    for c in _combinations(sorted(pins), m)]
    for rchoice in product(*per_doc):
        for ps in pin_sets:
            yield zs, tuple(rchoice), ps
