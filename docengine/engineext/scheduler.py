"""Query-aware scheduler extension for vLLM v1 (phase C).

Subclasses vLLM's scheduler so the query plan, not the engine's
heuristics, governs memory. Live documents' prefix blocks are pinned
by holding an extra block-pool reference, dead documents' blocks are
stripped and freed the instant the plan learns of the death, and every
block a planned request leaves behind that is not pinned has its cache
entry removed on free. Everything uses the block pool's public
interface; vendor code is not modified.

Single tenant mode (env DOCENGINE_SINGLE_TENANT, default on) is the
shipping configuration: requests that do not carry a plan tag are
refused, so unplanned memory cannot exist, and the engine's
keep-the-most-recent eviction rule becomes unreachable by
construction. If it ever fires anyway, that means memory escaped the
plan's accounting, and the scheduler raises instead of silently
falling back to heuristic behavior. Set the env var to 0 for shared
card experiments, where foreign traffic is legitimate and the recency
rule is its default policy.

The scheduler lives in the engine core process, so client directives
ride inside request ids. Protocol, fields separated by "|":

    de1|p<tokens>|d<doc>|r<doc,doc,...>|<suffix>

  p<tokens>  pin the document prefix covering the first <tokens> tokens
             when this request's blocks are freed
  d<doc>     document key that owns the pin
  r<...>     release the pins of these document keys; "*" releases all
"""

import os

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus


def parse_tag(request_id):
    """Return (pin_tokens, doc_key, releases) for a tagged id, else None."""
    if not request_id.startswith("de1|"):
        return None
    pin, doc, rel = 0, None, []
    for part in request_id.split("|")[1:]:
        if part.startswith("p") and part[1:].isdigit():
            pin = int(part[1:])
        elif part.startswith("d") and len(part) > 1:
            doc = part[1:]
        elif part.startswith("r") and len(part) > 1:
            rel = ["*"] if part[1:] == "*" else part[1:].split(",")
    return pin, doc, rel


class DocEngineScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._de_pins = {}          # doc key -> list of pinned blocks
        self._de_pinned_ids = set()  # block ids currently pinned
        self._de_intent = {}        # request id -> (pin_tokens, doc key)
        self._de_strict = os.environ.get(
            "DOCENGINE_SINGLE_TENANT", "1") == "1"
        self._de_plan_evicting = False
        self._de_stats = dict(pinned=0, released=0, blocks=0,
                              foreign_rejected=0, heuristic_evictions=0)

        pool = self.kv_cache_manager.block_pool
        orig_evict = pool._maybe_evict_cached_block

        def guarded(block):
            evicted = orig_evict(block)
            if evicted and not self._de_plan_evicting:
                self._de_stats["heuristic_evictions"] += 1
                if self._de_strict:
                    raise RuntimeError(
                        "DocEngine single-tenant invariant violated: the "
                        "recency rule evicted a cached block, so some "
                        "memory escaped the plan's accounting")
            return evicted

        pool._maybe_evict_cached_block = guarded

    def _de_evict(self, block_ids):
        """Plan-initiated cache-entry removal (not a heuristic event)."""
        if not block_ids:
            return
        pool = self.kv_cache_manager.block_pool
        self._de_plan_evicting = True
        try:
            pool.evict_blocks(block_ids)
        finally:
            self._de_plan_evicting = False

    def add_request(self, request):
        tag = parse_tag(request.request_id)
        if tag is None and self._de_strict:
            # a single tenant appliance serves only planned work
            super().add_request(request)
            self._de_stats["foreign_rejected"] += 1
            self.finish_requests(request.request_id,
                                 RequestStatus.FINISHED_ABORTED)
            return
        if tag is not None:
            pin, doc, rel = tag
            if rel:
                self._de_release(rel)
            if pin > 0 and doc is not None and doc not in self._de_pins:
                self._de_intent[request.request_id] = (pin, doc)
        super().add_request(request)

    def _de_release(self, docs):
        pool = self.kv_cache_manager.block_pool
        keys = list(self._de_pins) if docs == ["*"] else docs
        for key in keys:
            blocks = self._de_pins.pop(key, None)
            if blocks:
                # dead by the plan's decree: strip cache entries, then
                # drop the pin references; hashless blocks join the head
                # of the free queue and are reusable immediately
                self._de_evict({b.block_id for b in blocks})
                self._de_pinned_ids.difference_update(
                    b.block_id for b in blocks)
                pool.free_blocks(reversed(blocks))
                self._de_stats["released"] += 1
        if docs == ["*"]:
            print(f"[de-sched] release-all: stats {self._de_stats}",
                  flush=True)

    def _free_request_blocks(self, request):
        if parse_tag(request.request_id) is not None:
            groups = self.kv_cache_manager.coordinator.get_blocks(
                request.request_id)
            blocks = groups[0] if groups else []
            intent = self._de_intent.pop(request.request_id, None)
            if (intent is not None
                    and request.num_computed_tokens >= intent[0]
                    and intent[1] not in self._de_pins):
                pin_tokens, doc = intent
                keep = [b for b in blocks[:pin_tokens // self.block_size]
                        if not b.is_null]
                if keep:
                    self.kv_cache_manager.block_pool.touch(keep)
                    self._de_pins[doc] = keep
                    self._de_pinned_ids.update(b.block_id for b in keep)
                    self._de_stats["pinned"] += 1
                    self._de_stats["blocks"] += len(keep)
            # the plan owns its memory: whatever this request leaves
            # behind unpinned has no future consumer, so strip its
            # cache entries rather than leave them to the recency rule
            tail = {b.block_id for b in blocks
                    if not b.is_null
                    and b.block_id not in self._de_pinned_ids}
            self._de_evict(tail)
        super()._free_request_blocks(request)
