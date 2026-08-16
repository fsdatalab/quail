"""Wave driver: synchronous pre-loading of stored KV.

Applies waves_logic decisions to engine state from inside
QuailScheduler.schedule(). QUAIL_WAVES=1 enables pre-loading;
QUAIL_SPILL=1 additionally puts preempted requests at the front of
the wave order, so a document evicted under a choked pool reloads
through the same channel before it is rescheduled - spill reload and
pre-load are one mechanism with two orderings.

Territory. The vendor's per-request load path and the wave channel
must never load the same bytes. The vendor defers an external-hit
request at zero token cost (load_kv_async schedules no new tokens),
so it can sweep the whole waiting queue in one step - a skip window
cannot contain it; measured 1.6x duplicated load bytes when it
shared territory with waves. So the split is enforced at the store
lookup instead: every wave document is CLAIMED, and a claimed
request's lookup answers None - the vendor's own not-ready path,
which sets the request aside and re-asks next step, charging
nothing. After the wave registers, the request comes back through
and cache-hits locally. Two regions are left to the vendor's load
path:

  the seam   the first 2 x step-budget tokens of the query, spent
             once, never claimed. The head is scheduled before any
             wave lands, and the per-request path moves it at
             channel speed rather than prefilling it. The budget is
             per query, not per step: requests reach the scheduler
             over many steps (the client feeds a separate engine
             process), and a per-step window would roll over each
             batch of arrivals until it had handed most of the
             corpus to the per-request path - measured 77 percent.
  the floor  max_num_seqs blocks + one step budget of blocks stay
             free so the vendor pass never starves for allocation.

Waves are planned front-first beyond the seam, all at once while
pool headroom lasts, so registrations land in the order the vendor
walks the queue and the copy frontier runs ahead of the sweep.

The cycle for one wave:

  plan       allocate GPU blocks, claim the documents, pin the CPU
             blocks, hand the worker one copy list (it rides this
             step's connector metadata; the copy starts on the wave
             transfer stream immediately)
  copy done  the worker reports the wave's job finished; the next
             step's driver pass registers the blocks in the prefix
             cache, releases the claims, and queues the wave's gate.
             Registering only after the copy lands means the gate's
             event wait is already satisfied - the compute stream
             never stalls on it; the gate stays as insurance for
             the report-vs-event edge
  later      as wave documents get scheduled, the driver drops its
             extra block references; a fully consumed wave vanishes

If a claimed document is ever consumed before its wave registers
(possible only through a duplicate-content local cache hit), it
computed or reused its own KV and the wave's copy of it is freed
unregistered: a wasted copy, never a wrong answer. Known limits, on
purpose for now: wave blocks left unconsumed are evicted by the
engine's ordinary reuse path, which single-tenant strict mode would
refuse - run waves with QUAIL_SINGLE_TENANT=0 - and duplicate
documents could cache-hit a wave block before its copy lands, which
distinct-document corpora cannot trigger.
"""

import os

from vllm.v1.kv_offload.base import GPULoadStoreSpec

from .waves_logic import plan_waves


class WaveDriver:
    def __init__(self, scheduler):
        self.s = scheduler
        self.on = os.environ.get("QUAIL_WAVES", "0") == "1"
        self.spill = os.environ.get("QUAIL_SPILL", "0") == "1"
        self._wave_tokens = None
        self._seam_left = None     # one-time head budget, refilled at drain
        self._handled = set()      # request ids waved or rejected
        self._doc_wave = {}        # request id -> wave id
        self._pending_reg = []     # (wave_id, [(req, blocks, n_full)])
        self._live = {}            # wave id -> {rid: blocks} unconsumed
        self._consumed_early = set()   # prefilled before wave registered
        self._next_wave = 0
        self._errors = 0
        self.stats = dict(waves=0, docs=0, blocks=0, reloads=0)
        if self.on:
            print(f"[quail-waves] on, spill {self.spill}", flush=True)

    def _cs(self):
        c = getattr(self.s, "connector", None)
        cs = getattr(c, "connector_scheduler", None)
        return cs if hasattr(cs, "queue_wave") else None

    def before(self):
        """Runs before the vendor scheduling pass."""
        if not self.on:
            return
        try:
            self._before()
        except Exception as e:
            self._errors += 1
            print(f"[quail-waves] step error {self._errors}: "
                  f"{type(e).__name__}: {e}", flush=True)
            if self._errors >= 3:
                self.on = False
                # claimed requests answer None at the store lookup;
                # with the driver dead nothing would ever release
                # them, so release them all to the vendor's paths
                cs = self._cs()
                if cs is not None:
                    cs.wave_claimed.clear()
                print("[quail-waves] disabled after repeated errors",
                      flush=True)

    def _before(self):
        cs = self._cs()
        if cs is None:
            return
        pool = self.s.kv_cache_manager.block_pool
        block_size = self.s.block_size
        step_budget = self.s.scheduler_config.max_num_batched_tokens
        if self._wave_tokens is None:
            self._wave_tokens = (
                int(os.environ.get("QUAIL_WAVE_TOKENS", "0"))
                or 2 * step_budget)

        # 1. register waves whose copy the worker reported finished.
        # From now on their documents cache-hit locally, so each gate
        # is queued in this same step - and is already satisfied.
        # get_new_blocks left one reference on each block at planning
        # time; that allocation reference IS the driver's pin (a
        # block with ref_cnt > 0 cannot be evicted), released in
        # after() when the document consumes the wave. A document the
        # vendor prefilled meanwhile computed its own KV; its blocks
        # are freed here unregistered, because a duplicate cache
        # entry must not serve a hit.
        if self._pending_reg:
            still = []
            for wave_id, members in self._pending_reg:
                if wave_id not in cs.finished_waves:
                    still.append((wave_id, members))
                    continue
                cs.finished_waves.discard(wave_id)
                left = {}
                for req, blocks, n_full in members:
                    cs.wave_claimed.discard(req.request_id)
                    if req.request_id in self._consumed_early:
                        self._consumed_early.discard(req.request_id)
                        pool.free_blocks(blocks)
                        continue
                    pool.cache_full_blocks(
                        request=req, blocks=blocks, num_cached_blocks=0,
                        num_full_blocks=n_full, block_size=block_size,
                        kv_cache_group_id=0)
                    left[req.request_id] = blocks
                if left:
                    self._live[wave_id] = left
                cs.queue_gates([wave_id])
            self._pending_reg = still

        # 2. plan waves over the stored tail of the waiting queue
        floor_blocks = (self.s.scheduler_config.max_num_seqs
                        + -(-step_budget // block_size))
        budget_blocks = pool.get_num_free_blocks() - floor_blocks
        if budget_blocks <= 0:
            return
        try:
            waiting = list(self.s.waiting)
        except TypeError:
            self.on = False
            print("[quail-waves] waiting queue not iterable; disabled",
                  flush=True)
            return
        if self._seam_left is None:
            self._seam_left = 2 * step_budget
        order = waiting
        if self.spill:
            # a preempted document is mid-chain: its next visit is
            # the soonest, so it reloads first. Preempted documents
            # inside the seam stay the vendor's - it is about to
            # recompute them, and a wave would duplicate that.
            pre = [r for r in order if r.num_preemptions]
            for r in pre:
                rid = r.request_id
                if rid in self._handled and rid not in self._doc_wave:
                    self._handled.discard(rid)
                    self.stats["reloads"] += 1
            order = pre + [r for r in order if not r.num_preemptions]
        cands, by_id = [], {}
        for r in order:
            rid = r.request_id
            if rid in self._handled or r.num_computed_tokens:
                continue
            if self._seam_left > 0:
                self._seam_left -= r.num_tokens
                self._handled.add(rid)
                continue
            if not r.block_hashes:
                self._handled.add(rid)
                continue
            # a document whose last full block is already in the GPU
            # prefix cache needs no wave: the local hit wins anyway,
            # and a wave for it is a wasted copy (the write query's
            # documents all look like this)
            if pool.get_cached_block(r.block_hashes[-1], [0]):
                self._handled.add(rid)
                continue
            cands.append((rid, len(r.block_hashes) * block_size))
            by_id[rid] = r
        planned = plan_waves(cands, self._wave_tokens,
                             budget_blocks * block_size)
        for wave_members in planned:
            wave_id = self._next_wave
            members, all_keys, gpu_ids = [], [], []
            starved = False
            for rid in wave_members:
                req = by_id[rid]
                self._handled.add(rid)
                keys = cs.wave_keys_for(req)
                if not keys:
                    continue
                n_full = len(keys)
                if pool.get_num_free_blocks() < n_full + floor_blocks:
                    starved = True
                    break
                blocks = pool.get_new_blocks(n_full)
                members.append((req, blocks, n_full))
                all_keys.extend(keys)
                gpu_ids.extend(b.block_id for b in blocks)
                self._doc_wave[rid] = wave_id
                cs.wave_claimed.add(rid)
            if members:
                src = cs.prepare_wave(all_keys)
                dst = GPULoadStoreSpec(block_ids=gpu_ids,
                                       group_sizes=[len(gpu_ids)],
                                       block_indices=[0])
                cs.queue_wave(wave_id, all_keys, src, dst)
                self._pending_reg.append((wave_id, members))
                self._next_wave += 1
                self.stats["waves"] += 1
                self.stats["docs"] += len(members)
                self.stats["blocks"] += len(gpu_ids)
                print(f"[quail-waves] wave {wave_id}: "
                      f"{len(members)} documents, {len(gpu_ids)} blocks",
                      flush=True)
            if starved:
                break

    def after(self, scheduled_ids):
        """Runs after the vendor pass: waves whose documents got
        scheduled lose the driver's extra block references - the
        requests hold their own now."""
        if not self.on or not self._doc_wave:
            return
        try:
            for rid in scheduled_ids:
                wave_id = self._doc_wave.pop(rid, None)
                if wave_id is None:
                    continue
                left = self._live.get(wave_id)
                blocks = left.pop(rid, None) if left else None
                if blocks:
                    self.s.kv_cache_manager.block_pool.free_blocks(blocks)
                elif wave_id not in self._live:
                    # the vendor prefilled it before its wave
                    # registered: the wave frees this member's blocks
                    # at registration
                    self._consumed_early.add(rid)
                if left is not None and not left:
                    del self._live[wave_id]
        except Exception as e:
            print(f"[quail-waves] release error: {e}", flush=True)

    def drain(self):
        """Release every reference the driver still holds. Called at
        quiesce points only (a prefix-cache reset between queries):
        wave copies must be long finished, because freed blocks
        rejoin the pool while an in-flight copy would still write
        them."""
        if not (self._live or self._pending_reg or self._doc_wave):
            return
        pool = self.s.kv_cache_manager.block_pool
        n = 0
        try:
            for wave_id, members in self._pending_reg:
                for _req, blocks, _n_full in members:
                    pool.free_blocks(blocks)
                    n += len(blocks)
            for left in self._live.values():
                for blocks in left.values():
                    pool.free_blocks(blocks)
                    n += len(blocks)
            cs = self._cs()
            if cs is not None:
                cs.wave_claimed.clear()
                cs.finished_waves.clear()
        except Exception as e:
            print(f"[quail-waves] drain error: {e}", flush=True)
        self._pending_reg = []
        self._live = {}
        self._doc_wave = {}
        self._consumed_early = set()
        self._handled = set()
        self._seam_left = None
        if n:
            print(f"[quail-waves] drained {n} blocks", flush=True)
