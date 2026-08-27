# The planner runs the SoL's left deep search, and the engine keeps KV across operators

## What changed

Two things landed together, because the second makes the first true.

1. **The join search in the planner is now the left deep subset DP**
   the SoL estimate runs (`quail/planner/leftdeep.py`, moved out of
   `quail/bench/sol_dp.py`). The state is (joined alias set, cached
   prefix alias set). Stage costs are expected-value `Work` records -
   tokens, attention pairs, KV written, KV read - built from the
   `scan`/`ask`/`stream` operations in `quail/planner/sol.py`, which
   the SoL calculation now imports from the planner instead of
   defining itself. An anchor whose document prefix KV is resident
   pays only its question frame; one without pays preamble + document
   + frame. Candidate plans are ranked by predicted speed-of-light
   seconds from counted model constants and the device datasheet.
   The planner reads no calibration constant anywhere; the two
   measured constants stay as analysis tooling only.

2. **The runtime keeps document KV across operator boundaries.**
   Arena keys for document KV are now `("kv", alias, doc id)`,
   shared by the filter and join loops. A filter chain the plan
   marks `keep_kv` holds its survivors' KV after the chain; the join
   group anchored on that table finds the keys resident and packs no
   prefix tokens for them. A join group marked `keep_anchor_kv`
   holds its gate survivors for a later group on the same table, so
   an exists/anti gate between two same-anchor stages no longer
   forces a recompute. On several GPUs, join anchors follow the
   shards their KV sits on, so kept KV is always on the card that
   will read it.

## The eviction policy

The arena is smaller than some corpora, so keeping is a choice.

- The planner picks what to keep with a length threshold
  (`keep_split`): keep the longest survivors that fit. Resident KV
  of length L saves L dense tokens against the fp8 peak plus
  L(L+1)/2 attention pairs against the bf16 peak - both counted
  from the architecture and the datasheet, no measured constant -
  while occupying bytes linear in L. Saved work per byte rises with
  L under any positive weighting of the two terms, which is what
  orders documents by length; the rise comes from the attention
  term and is small below the dense/attention crossover (about
  12,320 prefix tokens at 4B, 29,760 at 32B - most benchmark
  documents are shorter), so the ordering matters more than the
  spread. What makes longest-first exact rather than a heuristic:
  survival is unknown per document at plan time, so the expected
  kept mass is fractional, and for a fractional knapsack taking by
  value per byte is optimal.
- The runtime never evicts to admit something into the cache.
  Every admission is a computation the query requires; only
  retention is optional, and retaining a document that is already
  resident costs nothing to start. Eviction happens only when kept
  KV starves a required admission: the smallest kept documents go
  first, minus any victim the larger ones make redundant, so no
  document is evicted for pages the admission does not need.
  Evicted documents are simply recomputed later.

## Why

The SoL accounting already credited filter-to-join KV reuse; the
engine recomputed every join anchor's prefix from scratch, and the
planner credited reuse only for consecutive full stages on one
anchor. The committed SoL report
(`reports/2026-08-26-sol-quailb.md`, data at
`/results/sol/sol_quailb_sf0.1.json` on the quail-results volume)
measured the plan-choice half of the gap: seven of the nine
multi-join queries improve under the exact left deep search, FEV-8
most, 0.900 s to 0.787 s at 4B and 7.460 s to 6.520 s at 32B. The
planner now runs that search with expectations instead of labels.
The residency half - not recomputing a 4,146-token BioDEX report at
the join after its filter already computed it - was not counted in
that comparison at all, because both sides of it assumed the reuse.
The engine now does what the model assumed.

## Numbers

Planner-predicted, from the new cost model on a 50-document,
300-token-mean filtered join (unit test
`test_filter_keep_makes_the_join_anchor_resident`): the join stage
costs 4,425 fresh tokens with the kept KV against 23,900 without it
- the difference is the 25 expected surviving documents' prefixes.
No engine run measures this yet; the join cells
(`tests/gpu/join_bench.py`, the QUAIL-B evaluation) are the next
step.

The SoL floor itself does not change: the optimal column is the
same exact-label, unlimited-KV left deep search as before. What
changes on the next SoL regeneration is the current-planner column,
because it simulates whatever plan the planner emits and the
planner changed. The plan-choice part of the gap (seven of nine
multi-join queries) should close, since both searches now cover the
same space. The ratio stays above 1.0 wherever the planner's
selectivity expectations and mean lengths miss the exact survivors,
and wherever arena capacity denies a keep the unlimited-KV optimum
assumes.

`simulate_query` in `reports/make_sol_quailb.py` follows the plan's
keep decisions (`keep_kv`, `keep_anchor_kv`) instead of assuming
unlimited persistence, so the "current planner" SoL column stays an
honest account of what the engine will do.
