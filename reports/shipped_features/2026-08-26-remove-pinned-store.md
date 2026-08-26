# Removed: CPU offloading (PinnedStore)

Closes issue #32.

## What changed

The pinned CPU KV store is gone. Document KV now lives only in the
GPU arena, for the duration of one query. Every query recomputes
every document's KV.

Removed:

- `quail/executor/kvstore.py`: `PinnedStore`, `ExtentAllocator`, and
  `alloc_with_reclaim`, plus their tests and the `filter_store_run`
  GPU cell.
- The store save/restore paths in `run_filter` and `run_join`
  (side-stream copies, load events, deferred page frees).
- The planner's store arithmetic: `StoreSpec`, `cpu_memory_gb`,
  the staging-buffer reservation, the restore break-even, the
  store length threshold, the read-vs-restore access decision, and
  the `store_needed_but_disabled` refusal.
- Store setup and store stats in the worker, the session, and the
  coordinator payloads.
- The benchmark's cold/warm protocol. QUAIL-B now runs each query
  once.

Kept on purpose: the `restored` parameter of `FilterAdmission` in
`pack.py` and its tests. If KV persistence in GPU memory is added
later, that is how the scheduler learns a document's KV is already
in the arena.

## Why

The store added a lot of machinery: 8 GiB pinned slabs, a CUDA side
stream, a GPU staging ring buffer, an extent allocator with
cross-dataset reclaim, a break-even calculation, and a length
threshold. The arena already survives between queries inside a warm
worker container, so KV persistence in GPU memory is the simpler
follow-up when we want reuse again.

## Before/after numbers

- Arena capacity for Qwen3 4B on one H100: 345,974 tokens, up from
  313,206. The difference is the 4.8 GB staging ring buffer the
  arena no longer reserves.
- 1,391 lines removed and 249 added across the engine, the
  benchmark, the GPU cells, the tests, and the docs.
- No performance measurement in this change: it removes the
  cross-query reuse path, so single-query walls are unchanged and
  repeated queries recompute their KV.
