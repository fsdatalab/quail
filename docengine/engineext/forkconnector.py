"""Partial-block KV copy for forked speculation.

Cross-request block sharing works in whole 16-token blocks: a forked
sibling reuses the document's full blocks by content hash, but the
last, partly filled block of the shared prefix cannot be hash-matched,
so without this connector every sibling recomputes those tokens.
The fix goes through vLLM's own external-KV interface
(KVConnectorBase_V1): the scheduler half reports the partial block's
tokens as externally computed, and the worker half copies the
parent's block into the sibling's freshly allocated block before the
forward pass reads it. A whole-block copy is sufficient because the
sibling computes its own question tokens immediately after the
copied region, overwriting the stale remainder.

Both halves live in the engine core process. DocEngineScheduler
writes FORK_PARTIALS (sibling request id -> the parent's source
block id and the shared token count) at fork creation; the scheduler
half consumes and clears it. Every guard degrades to "no external
tokens", which means the sibling recomputes the partial block - the
pre-connector behavior - so a surprise can cost speed, never
answers.
"""

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata)

# sibling request id -> (source block id, shared prefix token count)
FORK_PARTIALS = {}

_DEBUG = [3]   # print the first few copies verbatim


class ForkCopyMeta(KVConnectorMetadata):
    def __init__(self, pairs):
        self.pairs = pairs   # [(src block id, dst block id), ...]


class DocEngineForkConnector(KVConnectorBase_V1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = kwargs.get("vllm_config")
        if cfg is None:
            for a in args:
                if hasattr(a, "cache_config"):
                    cfg = a
                    break
        self._block_size = getattr(getattr(cfg, "cache_config", None),
                                   "block_size", 16)
        self._kv = {}        # layer name -> tensor (worker half)
        self._pending = []   # (src, dst) staged this scheduling pass

    # ---- scheduler half ------------------------------------------------

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        entry = FORK_PARTIALS.get(request.request_id)
        if entry is None:
            return 0, False
        _src, shared = entry
        # claim the partial tokens only when the local hits landed
        # exactly on the block boundary below the shared prefix;
        # anything else (evicted entries, unexpected hit depth) means
        # degrade: the sibling recomputes
        aligned = (shared // self._block_size) * self._block_size
        if num_computed_tokens != aligned or shared == aligned:
            FORK_PARTIALS.pop(request.request_id, None)
            return 0, False
        return shared - aligned, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        entry = FORK_PARTIALS.pop(request.request_id, None)
        if entry is None or not num_external_tokens:
            return
        src, shared = entry
        k = shared // self._block_size
        try:
            ids = blocks.get_block_ids()[0]
        except Exception:
            return
        # the destination is the block holding the shared prefix's
        # tail. When `blocks` carries the request's whole allocation
        # the hit blocks precede it (index k); when it carries only
        # the new allocation it comes first (index 0).
        dst = ids[k] if len(ids) > k else (ids[0] if ids else None)
        if dst is None or dst == src:
            return
        self._pending.append((src, dst))
        if _DEBUG[0] > 0:
            _DEBUG[0] -= 1
            print(f"[de-forkcopy] {request.request_id}: shared {shared} "
                  f"tokens, block index {k}, copy {src} -> {dst}, "
                  f"alloc ids {list(ids)[:6]}", flush=True)

    def build_connector_meta(self, scheduler_output):
        pairs, self._pending = self._pending, []
        return ForkCopyMeta(pairs)

    # ---- worker half ---------------------------------------------------

    def register_kv_caches(self, kv_caches):
        self._kv = dict(kv_caches)

    def start_load_kv(self, forward_context, **kwargs):
        meta = self._get_connector_metadata()
        pairs = getattr(meta, "pairs", None)
        if not pairs:
            return
        for src, dst in pairs:
            for t in self._kv.values():
                # block-number dim is the large one of the first two
                # (FlashInfer [blocks, 2, 16, ...] against
                # FlashAttention [2, blocks, 16, ...])
                if t.shape[0] > t.shape[1]:
                    t[dst].copy_(t[src], non_blocking=True)
                else:
                    t[:, dst].copy_(t[:, src], non_blocking=True)

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata,
                      **kwargs):
        pass

    def wait_for_save(self):
        pass
