"""Query-aware scheduler extension for vLLM v1 (phase C skeleton).

Subclasses vLLM's scheduler to add plan-aware KV retention on top of
the stock engine. A document's prefix blocks are pinned by taking an
extra block-pool reference when the document's first request releases
its own references, so the recency rule can never evict a document the
query plan still needs. Pins are released when the client reports the
document's chain is over. Everything uses the block pool's public
reference-counting interface; no vLLM internals are modified.

The scheduler lives in the engine core process, so client-to-scheduler
messages ride inside request ids. Ids not starting with "de1|" pass
through untouched, which keeps foreign traffic (for example a
co-tenant's requests) on stock behavior. Protocol, fields separated by
"|":

    de1|p<tokens>|d<doc>|r<doc,doc,...>|<suffix>

  p<tokens>  pin the document prefix covering the first <tokens> tokens
             when this request's blocks are freed
  d<doc>     document key that owns the pin
  r<...>     release the pins of these document keys; "*" releases all
"""

from vllm.v1.core.sched.scheduler import Scheduler


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
        self._de_pins = {}        # doc key -> list of pinned KVCacheBlocks
        self._de_intent = {}      # request id -> (pin_tokens, doc key)
        self._de_stats = dict(pinned=0, released=0, blocks=0)

    def add_request(self, request):
        tag = parse_tag(request.request_id)
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
                # The plan knows this document is dead: strip its cache
                # entries first (so nothing can hit them), then drop the
                # pin references. Hashless blocks join the head of the
                # free queue, so the space is reusable immediately
                # instead of waiting for demand-time eviction.
                pool.evict_blocks({b.block_id for b in blocks})
                pool.free_blocks(reversed(blocks))
                self._de_stats["released"] += 1
        if docs == ["*"]:
            print(f"[de-sched] release-all: stats {self._de_stats}",
                  flush=True)

    def _free_request_blocks(self, request):
        intent = self._de_intent.pop(request.request_id, None)
        if (intent is not None
                and request.num_computed_tokens >= intent[0]):
            pin_tokens, doc = intent
            if doc not in self._de_pins:
                groups = self.kv_cache_manager.coordinator.get_blocks(
                    request.request_id)
                blocks = groups[0] if groups else []
                keep = [b for b in blocks[:pin_tokens // self.block_size]
                        if not b.is_null]
                if keep:
                    self.kv_cache_manager.block_pool.touch(keep)
                    self._de_pins[doc] = keep
                    self._de_stats["pinned"] += 1
                    self._de_stats["blocks"] += len(keep)
        super()._free_request_blocks(request)
