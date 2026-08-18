# The wall-minus-GPU gap: the filter loop was CPU-bound

Date: 2026-08-18. One H100 on Modal. Qwen3 4B fp8. All data files are
in `results/`.

The profiling report (`2026-08-18-filter-profiling.md`) left one open
item: a wall-minus-GPU gap of about 2.4 s per 10k-document filter
run, attributed to answer readback and Python scheduling between
chunks. This report closes it.

## Baseline

Re-ran the committed filter gate on this setup: wall 36.01 / 36.08 s,
GPU 33.90 / 34.05 s, gap 2.11 / 2.03 s over 55 chunks (about 37 ms
per chunk). Survivors 1,807, answered 23,113, wrong 6,294 — identical
to the committed gate.

## Where the gap actually was

We added an opt-in `timing` dict to `run_filter` and `pack_chunk`
(host-side `perf_counter` per phase; no behavior change) and ran a
3,000-document cell (`run_filter_timing`). Result: wall 11.06 s, GPU
10.46 s, and per-chunk CPU of about 526 ms against per-chunk GPU of
about 498 ms. The loop was CPU-bound in steady state; the GPU starved
the difference each chunk.

The CPU time, per 3k run:

| Phase | CPU (s) | What it is |
|---|---|---|
| `alloc` | 8.62 | `KVArena.alloc`: one `torch.tensor(rows, device=cuda)` per fresh document (~143/chunk avg) |
| `forward_launch` | 1.82 | CPU dispatch of ~360 kernel launches per chunk |
| `pack` | 0.56 | `pack_chunk`, mostly `block_table`'s one tiny H2D per key |
| `report_wait` | 0.001 | answer readback — never waits |

Two notes against the earlier guesses: answer readback was already
fully overlapped (the profiling report's hypothesis there was wrong),
and the cost was not Python scheduling in general but one specific
pattern — small **pageable** host-to-device copies, which block the
CPU behind whatever the stream is still running. The same disease as
the KV scatter fixed in the profiling report, one layer down.

## The fix

- `KVArena.alloc` keeps row indices on the host. The device copy is
  built lazily on first use (`rows_gpu`, cached per residency), which
  in the filter loop is never — rows are only needed batched, per
  chunk.
- `block_table` builds the whole table flat on the host and stages it
  through pinned memory: one non-blocking copy instead of one
  pageable copy per key.
- Every index tensor in `pack_chunk` (ids, positions, cu tensors,
  cross rows, kv_src/kv_dst) goes through the same pinned,
  non-blocking stage. Pinned staging is safe to drop immediately:
  the caching host allocator defers reuse until the copy's stream
  event fires.

No arithmetic, launch order, or chunk composition changed.

## Results

| Gate | Before | After | Answers |
|---|---|---|---|
| Filter, 10k docs, 5 filters | 36.01 / 36.08 s | 34.02 / 34.03 s (−5.5%) | bit-identical (1,807 / 23,113 / 6,294) |
| Filter + store, cold | 31.28 s | 27.05 s (−13%) | identical (882 survivors, 5,000 stored) |
| Filter + store, warm | 11.70 s | 9.98 / 10.00 s (−15%) | identical (887 survivors, 5,000 restored) |
| Join, 256k pairs | 108.09 / 107.54 s | 105.56 / 105.29 s (−2.1%) | identical (177,346 yes, 77 chunks) |
| Probe parity gates | 0 disagreements | 0 disagreements | — |

The filter gap fell from 2.05 s to 0.08 s — the drain tail, which is
the irreducible part. Throughput rose from 106.3k to 112.7k tokens/s.
The store cells gained more because their chunks are smaller (62k
tokens per chunk on the cold run against 70-110k on the filter
gate), so the fixed per-chunk CPU cost weighed more against less GPU
time. Peak memory is unchanged everywhere (65.93 GiB filter, 66.83
cold store).

## What is still on the table

- `forward_launch` is now the largest CPU phase (1.82 s per 3k run,
  ~87 ms per chunk of kernel dispatch). It is fully hidden while
  per-chunk GPU time exceeds it, but it sets the floor if chunks
  ever shrink. CUDA graphs per chunk shape would erase it; shapes
  vary, so this is a cache of graphs keyed by token count, not one
  graph.
- GEMMs at 70% of peak (drain chunks), unchanged from the profiling
  report.

## Data files

- `results/m1_filter.json` / `m1_filter_staged.log`: the filter gate
  after the fix. The before numbers are in git history (commit
  `ad3f9be`'s `m1_filter.json`) and `m1_filter_gap_baseline.log`.
- `results/m1_filter_timing.json` / `.log`: the instrumented 3k run.
- `results/m1_filter_store.json` / `m1_filter_store_staged.log`:
  store gate after the fix.
- `results/m1_join.json` / `m1_join_staged.log`: join gate after the
  fix.
- `results/m1_probe.json` / `m1_probe_staged.log`: probe after the
  fix.
