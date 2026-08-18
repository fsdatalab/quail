# The wall-minus-GPU gap: the filter loop was CPU-bound

Date: 2026-08-18. One H100 on Modal. Qwen3 4B fp8. All data files are
in `results/`.

This report covers issue #9 (the 4-to-6-second overhead budget for
the 10k five-filter run) item by item: the wall-minus-GPU gap (item
4), the block_table build (item 2), the attention merge chain (item
3), and the drain-chunk GEMM question (item 1, measured and rejected
below).

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

Two changes landed: the staged-packing fix (items 2 and 4) and the
fused merge kernel (item 3). The staging fix is bit-identical; the
merge kernel carries the near-tie drift described above.

| Gate | Before | After staging | After staging + merge | Answers |
|---|---|---|---|---|
| Filter, 10k docs, 5 filters | 36.01 / 36.08 s | 34.02 / 34.03 s | 33.13 / 33.10 s (−8.1%) | staging: bit-identical; +merge: 1,811 / 23,152 / 6,276 |
| Filter + store, cold | 31.28 s | 27.05 s | 24.21 s (−22.6%) | 887 survivors, 5,000 stored |
| Filter + store, warm | 11.70 s | 9.98 / 10.00 s | 9.58 / 9.57 s (−18.2%) | 889 survivors, 5,000 restored |
| Join, 256k pairs | 108.09 / 107.54 s | 105.56 / 105.29 s | 97.10 / 97.10 s (−9.9%) | 177,406 yes (near-tie drift), 77 chunks |
| Probe parity gates | 0 disagreements | 0 disagreements | 0 disagreements | — |

The filter wall-minus-GPU gap fell from 2.05 s to 0.05-0.08 s — the
drain tail, which is the irreducible part. Filter throughput rose
from 106.3k to 115.8k tokens/s. The merge kernel is worth more on
the join than on the filter (the join is cross-attention-heavy, so
the merge chain was a bigger share of its GPU time). Peak memory is
unchanged everywhere (65.93 GiB filter, 66.83 cold store).

Against the issue's 4-to-6 s budget for the 10k filter: the gap
closure gave 2.0 s (the issue estimated 1 to 1.5), the merge gave
about 0.9 s of wall (the issue estimated 0.5 to 1), and item 1's
drain-chunk estimate did not hold up (about 0.1 s exists, not 2 to
3). Net: 36.0 to 33.1 s. The remaining distance to the ~28 s floor
is GEMM per-shape efficiency, uniform across chunk sizes.

## Item 3: the attention merge chain, fused

The two-call attention path merged its partial outputs with about 9
launches per layer (two LSE transposes, three index_selects, sub,
sigmoid, cast, lerp, index_copy_) — launch-bound tiny kernels, about
1.2 s per 10k run. `lse_merge` (one Triton program per suffix row,
each handling all heads) does the whole merge in one launch per
layer, reading the LSE tensors in their native layout.

Two pitfalls found and fixed during gating:

- The merge is not idempotent — a 2D grid merged every row 32 times
  (survivors collapsed to 1). The launch grid is now (n_suffix_rows,).
- The LSE strides vary with the chunk's token count, and Triton
  specializes integer arguments on div-by-16: the first chunk whose
  stride hit a new key paid a 0.6 s JIT compile inside the measured
  run. The stride arguments are now `do_not_specialize`.

The kernel replicates the unfused chain's rounding (fp32 sigmoid,
bf16 weight, fp32 lerp with torch's |w| < 0.5 branch). The residual
difference against the unfused chain is the fp32 sigmoid itself
(`tl.sigmoid` vs `torch.sigmoid` at the 1-ulp level), which flips a
few near-tie answers downstream: survivors 1,811 against 1,807,
answered 23,152 against 23,113, wrong 6,276 against 6,294 (27.1%
against 27.2%). The probe's parity gates all pass with zero
disagreements. This is the near-tie drift class the milestone 1
report documents between engines; it is not a correctness bug.

## Item 1: drain-chunk GEMMs, measured and rejected

The issue estimated 2 to 3 s from merging drain chunks or admission
look-ahead. The full per-chunk series (now dumped by the timing
cell) says otherwise at the current 110k-token budget:

- Chunks at 57k-110k tokens all run at 8.5 to 8.9 microseconds per
  token. Mid-run chunks are arena-limited, not budget-limited, and
  their GEMMs are as efficient as full chunks'.
- The actual drain tail is 8 chunks holding 16k tokens total. Its
  excess over the mid-run rate is about 70 ms per run.

Merging tail work or adding look-ahead would touch FilterAdmission
and pack for about 0.1 s. Not worth the complexity. The GEMM gap to
the roofline is per-shape kernel efficiency, uniform across chunk
sizes — not drain-driven, and not a scheduling fix.

## What is still on the table

- A rep-0-only tail lump of about 0.24 s (one or two small chunks
  run 5-10x slow on the first rep only, present before these
  changes). Smells like a one-time allocator or compile event in the
  drain phase; not yet traced.
- GEMM per-shape efficiency (~70% of the nominal fp8 roofline),
  uniform across chunk sizes. That is DeepGEMM kernel quality at our
  shapes, not scheduling.

## Data files

- `results/m1_filter.json` / `m1_filter_final.log`: the filter gate
  after both fixes. Baselines: commit `ad3f9be`'s `m1_filter.json`
  (pre-fix) and `m1_filter_staged.log` (staging only).
- `results/m1_filter_timing.json` / `m1_filter_timing_v2b.log`: the
  instrumented 10k run with the full per-chunk series.
  `m1_filter_timing_nomerge.json` isolates the staging fix at the
  same scale.
- `results/m1_filter_store.json` / `m1_filter_store_final.log`:
  store gate after both fixes.
- `results/m1_join.json` / `m1_join_final.log`: join gate after both
  fixes.
- `results/m1_probe.json` / `m1_probe_merge2.log`: probe after both
  fixes.
- `results/profile_filter.json` / `profile_merge.log`: the 3k
  torch-profiler run with the merge kernel.
