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
        # cache-hit, so the gate must be queued in this same step
        for wave_id, members in self._pending_reg:
            left = {}
            for req, blocks, n_full in members:
                pool.cache_full_blocks(
                    request=req, blocks=blocks, num_cached_blocks=0,
                    num_full_blocks=n_full, block_size=block_size,
                    kv_cache_group_id=0)
                pool.touch(blocks)
                left[req.request_id] = blocks
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
        cands = [(r.request_id, r) for r in waiting
                 if r.request_id not in self._handled
                 and not r.num_computed_tokens]
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
                if left is not None and not left:
                    del self._live[wave_id]
        except Exception as e:
            print(f"[quail-waves] release error: {e}", flush=True)
