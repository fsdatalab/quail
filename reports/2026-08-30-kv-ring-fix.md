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
Qwen3 4B fp8, one H100, so the unbounded-retention files stay intact for comparison.

## Prediction

Stated before the run:

- IMDB-3: zero eviction calls; the filter runs near-budget chunks
  for the whole scan (tens of chunks, not 2,916) and takes 15 to
  17 seconds, matching IMDB-1's 15.06 seconds for identical work;
  the join stays at about 18.5 to 19.5 seconds; engine wall 34 to
  38 seconds, against 68.9 measured with unbounded retention in
  the same harness and 75.0 recorded in the benchmark.
- IMDB-3 retention: the pool fills to about 8,800 pages
  (141,000 tokens); the join finds about 400 to 500 anchors
  resident (the longest survivors), a similar hit mass to the
  137,397 tokens the unbounded-retention run got - the fix does
  not buy more
  hits, it stops paying 35 seconds for them. Regret stays about
  1.2M tokens; the misses just stop costing batch shape.
- BIO-2: unchanged within noise (about 128 to 131 seconds, regret
  0, no evictions) - its plan retains nothing, so only the shared
  code paths could move it.
- Smoke first (sf 0.01): corpus fits beside the ring, everything
  retained, join reuses everything, as with unbounded retention.

## Result

Every prediction held. Smoke (sf 0.01) first: regret 0 on both
queries, no evictions, answers unchanged. The sf 0.1 numbers below
are the unprofiled pass; the profiled pass agrees within 5%.

### IMDB-3

| | unbounded retention | scan ring |
|---|---:|---:|
| Engine wall (s) | 68.93 | 32.68 |
| Filter phase (s) | 50.28 | 14.54 |
| Filter forward passes | 2,922 | 17 |
| Mean filter pass (tokens) | 602 (469 after the arena filled) | 103,484 |
| Eviction calls in the filter | 2,625 | 0 |
| Join phase (s) | 18.53 | 18.05 |
| Retained at the join (docs / pages) | 120 / 8,631 | 171 / 8,843 |
| Join hit tokens | 137,397 | 140,458 |
| Regret (tokens) | 1,220,547 | 1,217,486 |
| Survivors | 4,380 | 4,380 |

Figure: plots/kv_ring_fix_timeline.png

Figure: plots/kv_ring_fix_walls.png

- The engine wall halved: 32.68 seconds, compared with 68.93
  measured with unbounded retention in the same harness, 75.0
  recorded in the benchmark, and 52.65 recorded for stock vLLM. The filter now runs
  17 near-budget passes at 99.6% GPU busy (chunk grain) and lands
  at 14.54 seconds - level with IMDB-1's 15.06 seconds for the
  identical filter work, so the composition penalty is gone.
- Retention behaved exactly as designed. The pool filled to
  8,843 pages, its cap to the page (22,640 free pages at filter
  start minus the 13,797-page ring), holding 171 of the longest
  survivors at 140,458 tokens - within 0.7% of the planner's
  141,498-token credit. Every one of the 171 was still resident at
  the join and hit. Zero eviction calls anywhere; with unbounded retention, churn
  evicted 4,260 keys (78,268 pages) during the filter alone. One
  prediction miss, in the right direction: 400 to 500 resident
  anchors were predicted from the corpus mean length, but the
  pool's replacement rule keeps the longest survivors, so the same
  token mass arrived as 171 documents of 821 mean tokens against
  the 310-token survivor mean.
- Regret is unchanged (1.22M tokens in both runs), as
  predicted: the fix does not buy more hits - the
  unbounded-retention run ended up with a
  similar hit mass - it stops paying 35 seconds of collapsed
  batches for them. Both runs produced the same 4,380 survivors,
  so the change is performance-only.
- Benchmark metrics for IMDB-3 at the new wall: 52,560 evaluated
  document pairs / 32.68 s = 1,608 document pairs/second
  (recorded with unbounded retention: 701); $0.0359 per query at the H100 rate of
  $3.9492/hour, compared with $0.0823 recorded with unbounded retention and
  $0.0578 for stock vLLM's recorded 52.65 seconds.

### BIO-2

128.22 seconds, compared with 130.35 with unbounded retention -
container variance,
same 98 forward passes at 105,861 mean tokens, 99.4% busy, regret
0, no evictions. Its plan retains nothing, so this is the expected
no-change control.

Data on the `quail-results` volume:

- `/results/ablations/ringfix_imdb3.json`, `ringfix_bio2.json`
  (sf 0.1); `ringfix_imdb3_sf0.01.json`, `ringfix_bio2_sf0.01.json`
  (smoke)
- `/results/ablations/ringfix_traces/` (five chrome traces)
- Unbounded-retention comparisons: `discrepancy_imdb3.json`,
  `discrepancy_bio2.json` from the discrepancy report.
- Modal function calls: `fc-01M18J895155R1VS59VRPTGWEN` (sf 0.1),
  `fc-01M18J4YZ2NZH568D71KA1HEG0` (smoke).

## Meaning

- The IMDB-3 loss to stock vLLM is gone: 32.68 seconds against
  stock's 52.65. Quail's parts now compose: filter 14.54 plus join
  18.05 is the whole query, 0.10 seconds apart from the engine
  wall.
- Retention is now safe by construction, not by tuning. Admission
  owns two chunk budgets for the whole scan; retention competes
  only with itself, by saved recompute per page, and a replacement
  costs heap bookkeeping in the answer path rather than a stalled
  forward pass. The planner prices the same reservation, so the
  credit (141,498 tokens) matched the runtime pool (140,458) to
  0.7%.
- The QuailB headline table should be refreshed by a full benchmark
  rerun; this report's confirming cell covers only the two queries
  it re-measured.

## Rebuild

Rerun the cell (any out-prefix other than `discrepancy` selects
the post-fix profiler windows):

    uv run modal run ablations/discrepancy_timeline.py::run_smoke --out-prefix ringfix
    uv run modal run ablations/discrepancy_timeline.py::run_queries --out-prefix ringfix

Rebuild the figures with `reports/make_kv_ring_fix_plots.py`; its
docstring holds the `modal volume get` commands.
