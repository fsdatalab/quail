"""Wave driver: synchronous pre-loading of stored KV.

Applies waves_logic decisions to engine state from inside
QuailScheduler.schedule(). QUAIL_WAVES=1 enables pre-loading;
QUAIL_SPILL=1 additionally puts preempted requests at the front of
the candidate order, so a document evicted under a choked pool
reloads through the same channel before it is rescheduled - spill
reload and pre-load are one mechanism with two orderings.

The step cycle for one wave:

  step k    pick documents from the waiting queue, allocate their GPU
            blocks, pin the CPU blocks, hand the worker one copy list
            (it rides step k's connector metadata; the copy starts on
            the transfer stream at step k's start)
  step k+1  register the blocks in the prefix cache and queue the
            wave's gate; any later step that schedules one of the
            wave's documents makes the compute stream wait on the
            wave's CUDA event first, so correctness is pure stream
            ordering - no request parking, no polling
  later     as wave documents get scheduled, the driver drops its
            extra block references; a fully consumed wave vanishes

A document scheduled before its wave registers simply prefills as if
there were no store: a wasted copy, never a wrong answer. Known
limits, on purpose for now: wave blocks left unconsumed are evicted
by the engine's ordinary reuse path, which single-tenant strict mode
would refuse - run waves with QUAIL_SINGLE_TENANT=0 - and duplicate
documents could cache-hit a wave block before its copy lands, which
distinct-document corpora cannot trigger.
"""

import os

from vllm.v1.kv_offload.base import GPULoadStoreSpec

from .waves_logic import plan_wave


class WaveDriver:
    def __init__(self, scheduler):
        self.s = scheduler
        self.on = os.environ.get("QUAIL_WAVES", "0") == "1"
        self.spill = os.environ.get("QUAIL_SPILL", "0") == "1"
        self.max_in_flight = 2
        self._wave_tokens = None
        self._next_wave = 0
        self._handled = set()      # request ids waved or rejected
        self._doc_wave = {}        # request id -> wave id
        self._pending_reg = []     # (wave_id, [(req, blocks, n_full)])
        self._live = {}            # wave id -> {rid: blocks} unconsumed
        self._consumed_early = set()   # scheduled before wave registered
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
                print("[quail-waves] disabled after repeated errors",
                      flush=True)

    def _before(self):
        cs = self._cs()
        if cs is None:
            return
        pool = self.s.kv_cache_manager.block_pool
        block_size = self.s.block_size
        if self._wave_tokens is None:
            self._wave_tokens = (
                int(os.environ.get("QUAIL_WAVE_TOKENS", "0"))
                or 2 * self.s.scheduler_config.max_num_batched_tokens)

        # 1. register last step's wave; from now on its documents can
        # cache-hit, so the gate must be queued in this same step.
        # get_new_blocks already left one reference on each block -
        # that allocation reference IS the driver's pin (a block with
        # ref_cnt > 0 cannot be evicted), released in after() when the
        # document consumes the wave. A document consumed before its
        # wave registered computed its own KV; its blocks are freed
        # here unregistered, because their copy may still be in
        # flight and a duplicate cache entry must not serve a hit.
        for wave_id, members in self._pending_reg:
            left = {}
            for req, blocks, n_full in members:
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
        self._pending_reg = []

        # 2. plan the next wave
        if len(self._live) + len(self._pending_reg) >= self.max_in_flight:
            return
        try:
            waiting = list(self.s.waiting)
        except TypeError:
            self.on = False
            print("[quail-waves] waiting queue not iterable; disabled",
                  flush=True)
            return
        if self.spill:
            # a preempted document is mid-chain: its next visit is the
            # soonest, so it reloads first (the rotation order)
            for req in waiting:
                if (req.num_preemptions
                        and req.request_id in self._handled
                        and req.request_id not in self._doc_wave):
                    self._handled.discard(req.request_id)
                    self.stats["reloads"] += 1
            waiting.sort(key=lambda r: 0 if r.num_preemptions else 1)
        # the head of the queue belongs to the on-demand path: those
        # documents get scheduled in the next step or two, and a wave
        # for them loses the race - measured 1.8x duplicated loads
        # when waves targeted the head. Waves own everything beyond
        # the imminent region; the scan is bounded so a deep queue
        # costs nothing per step.
        head_skip = 2 * self.s.scheduler_config.max_num_batched_tokens
        scan_budget = 3 * self._wave_tokens * self.max_in_flight
        cands, skipped, scanned = [], 0, 0
        for r in waiting:
            if scanned >= scan_budget:
                break
            rid = r.request_id
            if rid in self._handled or r.num_computed_tokens:
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
            tokens = len(r.block_hashes) * block_size
            if skipped < head_skip:
                skipped += tokens        # not handled: on-demand's seam
                continue
            scanned += tokens
            cands.append((rid, r))
        sized = [(rid, len(r.block_hashes) * block_size)
                 for rid, r in cands]
        picked = set(plan_wave(
            sized, self._wave_tokens,
            in_flight=len(self._live) + len(self._pending_reg),
            max_in_flight=self.max_in_flight))
        if not picked:
            return
        wave_id = self._next_wave
        members, all_keys, gpu_ids = [], [], []
        for rid, req in cands:
            if rid not in picked:
                continue
            self._handled.add(rid)
            keys = cs.wave_keys_for(req)
            if not keys:
                continue
            n_full = len(keys)
            # leave allocation headroom for the vendor pass right after
            if pool.get_num_free_blocks() < n_full + 64:
                break
            blocks = pool.get_new_blocks(n_full)
            members.append((req, blocks, n_full))
            all_keys.extend(keys)
            gpu_ids.extend(b.block_id for b in blocks)
            self._doc_wave[rid] = wave_id
        if not members:
            return
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
        print(f"[quail-waves] wave {wave_id}: {len(members)} documents, "
              f"{len(gpu_ids)} blocks", flush=True)

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
                    # scheduled before its wave registered: the wave
                    # frees this member's blocks at registration
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
        except Exception as e:
            print(f"[quail-waves] drain error: {e}", flush=True)
        self._pending_reg = []
        self._live = {}
        self._doc_wave = {}
        self._consumed_early = set()
        self._handled = set()
        if n:
            print(f"[quail-waves] drained {n} blocks", flush=True)
