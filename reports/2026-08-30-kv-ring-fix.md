# The scan ring: KV retention that cannot starve admission

## Setup

The 2026-08-30 discrepancy report
(`2026-08-30-imdb3-bio2-discrepancies.md`) showed why IMDB-3 was the
one QuailB query Quail lost: the planner priced 358,861 of the
362,250-token arena as retained survivor KV while reserving working
headroom for only one document (3,389 tokens), and the filter loop
actually keeps up to two chunk budgets of in-flight document KV
(220,752 tokens). Six chunks in, the arena filled, and every later
admission first evicted one retained document through the
blocked-admission path. The filter degenerated into 2,916 forward
passes of 469 mean tokens; retention cost 35.2 seconds to save 1.3.

This PR changes how retained KV is managed, in three parts:

- **The scan ring.** Before a filter with retention starts, the loop
  reserves pages for two chunk budgets of document KV - one chunk
  executing while the next is packed - and caps retained KV at what
  is left. If an earlier operator's retained KV crowds the ring, the
  least valuable prefixes are evicted once, in bulk, up front. The
  admission path never touches retained KV again.
- **The retained pool** (`quail/executor/retention.py`,
  `RetainedPool`). Passing survivors are offered to a
  fixed-capacity pool. While it has room, every offer is kept. Once
  full, the residents with the least saved recompute per page are
  candidates to make room, and the newcomer replaces them only when
  its recompute value strictly exceeds what the victims lose
  together. Total retained value only rises; equal value never
  swaps; longer documents displace shorter ones. A replacement costs
  a heap operation, not a forward pass: it happens in the
  answer-handling path, never in admission.
- **Planner agreement** (`quail/planner/decide.py`). The keep credit
  now reserves the same two-chunk working headroom
  (`headroom = 2 * chunk`, 220,752 tokens at 4B), so the plan prices
  as resident only what the runtime can actually hold:
  141,498 tokens instead of 358,861.

The old one-victim-at-a-time eviction inside blocked admission
remains only as a safety valve; with the ring in place it should
never fire.

Confirming run: `ablations/discrepancy_timeline.py` (the same
instrumented cell as the discrepancy report; it wraps the engine
without modifying it) rerun with `--out-prefix ringfix`, sf 0.1,
Qwen3 4B fp8, one H100, so the pre-fix files stay intact for
comparison.

## Prediction

Stated before the run:

- IMDB-3: zero eviction calls; the filter runs near-budget chunks
  for the whole scan (tens of chunks, not 2,916) and takes 15 to
  17 seconds, matching IMDB-1's 15.06 seconds for identical work;
  the join stays at about 18.5 to 19.5 seconds; engine wall 34 to
  38 seconds, against 68.9 measured pre-fix in the same harness and
  75.0 recorded in the benchmark.
- IMDB-3 retention: the pool fills to about 8,800 pages
  (141,000 tokens); the join finds about 400 to 500 anchors
  resident (the longest survivors), a similar hit mass to the
  137,397 tokens the pre-fix run got - the fix does not buy more
  hits, it stops paying 35 seconds for them. Regret stays about
  1.2M tokens; the misses just stop costing batch shape.
- BIO-2: unchanged within noise (about 128 to 131 seconds, regret
  0, no evictions) - its plan retains nothing, so only the shared
  code paths could move it.
- Smoke first (sf 0.01): corpus fits beside the ring, everything
  retained, join reuses everything, as pre-fix.

## Result

(to be filled from the run)
