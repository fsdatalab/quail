# The wall-minus-GPU gap: the filter loop was CPU-bound

Date: 2026-08-18. One H100 on Modal. Qwen3 4B fp8. All data files are
in `results/`.

This report covers issue #9 (the 4-to-6-second overhead budget for
the 10k five-filter run) item by item: the wall-minus-GPU gap (item
4) and the block_table build (item 2), which landed as one change;
the attention merge chain (item 3) and the drain-chunk GEMM question
(item 1), both measured and rejected below.

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

One change landed: the staged-packing fix (items 2 and 4). The merge
fusion (item 3) was implemented and measured but rejected — see its
section. All numbers below are the packing fix alone, answers
bit-identical everywhere.

| Gate | Before | After | Answers |
|---|---|---|---|
| Filter, 10k docs, 5 filters | 36.01 / 36.08 s | 33.9-34.7 s across the day's runs (−4 to −6%) | bit-identical (1,807 / 23,113 / 6,294) |
| Filter + store, cold | 31.28 s | 27.05 s (−13%) | identical (882 survivors, 5,000 stored) |
| Filter + store, warm | 11.70 s | 9.98 / 10.00 s (−15%) | identical (887 survivors, 5,000 restored) |
| Join, 256k pairs | 108.09 / 107.54 s | 105.56 / 105.29 s (−2.1%) | identical (177,346 yes, 77 chunks) |
| Probe parity gates | 0 disagreements | 0 disagreements | — |

The filter wall-minus-GPU gap fell from 2.05 s to 0.05-0.08 s — the
drain tail, which is the irreducible part. Filter throughput rose
from about 106k to about 111k tokens/s. The store cells gained more
because their chunks are smaller (62k tokens per chunk on the cold
run against 70-110k on the filter gate), so the fixed per-chunk CPU
cost weighed more against less GPU time. Peak memory is unchanged
everywhere (65.93 GiB filter, 66.83 cold store).

One honesty note on the filter number: the same code measured
34.02 / 34.03 s on the morning's Modal instance and 34.57 / 34.66 s
on the afternoon's (GPU time itself moved 33.9 to 34.5 s). The
baseline moved less (36.0 s was measured on the morning instance).
The gap closure — the actual claim — is instance-independent: wall
minus GPU is 0.05-0.08 s on every run after the fix, against 2.0-2.1
s before it.

Against the issue's 4-to-6 s budget for the 10k filter: the gap
closure gave about 2 s (the issue estimated 1 to 1.5), item 1's
drain-chunk estimate did not hold up (about 0.1 s exists, not 2 to
3), and item 3's 0.5-1 s was real but rejected. Net landed: 36.0 to
about 34.3 s. The remaining distance to the ~28 s floor is GEMM
per-shape efficiency, uniform across chunk sizes.

## Item 3: the attention merge chain — implemented, measured, rejected

The two-call attention path merges its partial outputs with about 9
launches per layer (two LSE transposes, three index_selects, sub,
sigmoid, cast, lerp, index_copy_) — launch-bound tiny kernels, about
1.2 s per 10k run. We built the fusion: `lse_merge`, one Triton
program per suffix row handling all heads, reading the LSE tensors
in their native layout, replicating torch's lerp branch and rounding.

Measured with the fusion on top of the packing fix: filter 33.1 s
(0.9 s better than the packing fix alone), join 97.1 s (8.4 s
better), probe parity gates all zero disagreements.

Rejected anyway, on a team call. The kernel's fp32 sigmoid
(`tl.sigmoid`) differs from `torch.sigmoid` at the 1-ulp level,
which flips a few near-tie answers downstream: survivors 1,811
against 1,807, wrong rate 27.1% against 27.2%. That is the near-tie
drift class the milestone 1 report documents, and the probe gates
pass — but the team prefers bit-identical answers over the 0.9 s.
The kernel is not in the tree; it is preserved in git history
(commits `fef38aa`, `1ac4fe9`) with the pitfalls found along the
way, in case the trade-off is ever revisited:

- The merge is not idempotent: an early version's stale 2D launch
  grid merged every row 32 times (survivors collapsed to 1).
- The LSE strides vary with the chunk's token count, and Triton
  re-specializes integer arguments on div-by-16: the first such
  chunk paid a 0.6 s JIT compile inside a measured run. The fix was
  `do_not_specialize` on the stride arguments.

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

Final state (packing fix only):

- `results/m1_filter.json` / `m1_filter_final2.log`: the filter gate
  on the final tree. The pre-fix baseline is commit `ad3f9be`'s
  `m1_filter.json` (plus `m1_filter_gap_baseline.log`).
- `results/m1_filter_timing.json` / `m1_filter_timing_final.log`:
  the instrumented 10k run with per-phase CPU times and the full
  per-chunk series.
- `results/m1_filter_store.json` and `results/m1_join.json`: store
  and join gates, measured on executor code byte-identical to the
  final tree (`m1_filter_store_staged.log`, `m1_join_staged.log`).
- `results/m1_probe.json` / `m1_probe_final.log`: probe on the final
  tree.

Merge-fusion evidence (rejected item 3): `m1_filter_timing_v2b.log`
(filter 33.1 s with the kernel), `m1_join_final.log` (join 97.1 s),
`m1_probe_merge2.log` (probe passing), and the kernel itself in
commits `fef38aa` and `1ac4fe9`.
