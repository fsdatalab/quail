# One join search, run on estimates and on survivors, over retained KV

## What changed

Three pieces, landed together. This merges the two development
branches that split from the SoL work: this branch's plan-time cost
model and multi-GPU coordination, and sol-computation-math's
retention runtime and post-filter planning.

1. **One production join search** (`quail/planner/joins.py`): the
   left deep subset DP records the joined alias set and the current
   anchor group. The actual resident document prefixes are a read
   only input for the next group. The search does not predict later
   evictions. The worker searches again after every group, so the
   next call sees the real resident set. Stage costs are `Work` records
   (tokens, attention pairs, KV written, KV read) built per document
   over the live length lists: a resident anchor prefix pays its
   question frame only, the rest scan preamble + document + frame.
   Gates and n-ary predicates are searched like everything else;
   forced anchors are honored; candidates rank by speed-of-light
   seconds from counted model constants and the device datasheet. No
   calibration constant is read anywhere.

2. **Called again when answers arrive.** `plan_query` calls it with expected
   live counts and the keep credit, and emits the predicted plan -
   explain(), refusals, and sharding run off it.
   After the filter round the worker calls the same function with
   the actual survivors and the KV actually resident. It executes
   one join group, applies the answers, and searches the remaining
   joins again. The parent process does the same on several GPUs.
   The runtime search uses every answer available at that point, and
   the old `pick_runtime_anchor` heuristic is deleted. Stage
   outputs carry written_pos, semantics, and selectivity, so a
   runtime-chosen order assembles into results correctly.

3. **Retained KV across operators** (`executor/arena.py`): every filtered alias the search could
   anchor writes KV, and each survivor's prefix is retained at its
   final TRUE - rewound to preamble + document, evictable at its
   counted recompute value. The join finds the keys resident;
   `activate` grows their pages in place for the frame (no upfront
   reservation, no recompute on a changed anchor). A group whose
   anchor a later group re-uses retains its gate survivors the same
   way, so a gate between two same-anchor stages no longer forces a
   recompute. Thinned documents and KV with no future consumer are
   freed immediately; the arena is empty when the query ends. On
   several GPUs, children retain in their own arenas and join
   anchors follow the shards their KV sits on.

## The eviction policy

The runtime never evicts to admit a cache entry: every admission is
a computation the query requires, only retention is optional, and
retaining an already-resident document costs nothing to start. When
retained KV starves a required admission, the arena evicts document
prefixes in increasing saved work per page. A heap makes retention
and each eviction take logarithmic time in the number of retained
documents. The value of a prefix of length L is L dense
tokens against the fp8 peak plus L(L+1)/2 attention pairs against
the bf16 peak - both counted from the architecture and the
datasheet. The linear dense term dominates below the crossover
(about 12,320 prefix tokens at 4B, 29,760 at 32B), where most
benchmark documents sit. Pages are rounded to 16 tokens before the
ratio is computed. Pinned keys in use by the running operator are
not eviction candidates.

The heap rule is not the exact minimum-loss set. The exact problem
is a minimum knapsack cover problem. Its running time depends on the
number of pages that must be freed. The exact solver remains only as
a small test oracle. It does not run in the planner or worker.

At plan time the same idea appears as a credit, not a rule:
`keep_split` prices in the expected resident fraction the arena can
hold, longest documents first. The byte calculation is fractional,
while the arena allocates whole pages, so the split is an estimate.
The runtime is not bound by the threshold. The runtime tracks each
document and uses the exact page count.

## Why

The SoL accounting credited filter-to-join KV reuse the engine did
not perform, and the committed SoL report
(`reports/2026-08-26-sol-quailb.md`, data at
`/results/sol/sol_quailb_sf0.1.json` on the quail-results volume)
measured the plan-choice half of the remaining gap: seven of nine
multi-join queries improve under the exact left deep search, FEV-8
most, 0.900 s to 0.787 s at 4B and 7.460 s to 6.520 s at 32B. The
engine now performs the reuse the model assumed, and decides its
join plan with the best information available
at each moment. It uses estimates before anything runs, exact
survivors after the filters, and new exact survivors after every
join group.

## Numbers

Planner-predicted, from the cost model on a 50-document,
300-token-mean filtered join (unit test
`test_filter_keep_makes_the_join_anchor_resident`): the join stage
costs 4,425 fresh tokens with the retained KV against 23,900
without it - the difference is the 25 surviving documents'
prefixes.

Measured, on the source branch of the retention runtime: one H100!
check of a filter followed by two joins retained all 7 filter
survivors, hit all 7 at the report-anchored join, evicted nothing,
and matched a separate CPU recombination row for row
(`reports/2026-08-26-filter-join-kv-retention.md`, data at
`/results/runs/run_1787795777696587173.json`). That check ran the
lifecycle this branch ports verbatim, under the worker it was built
in. The merged worker re-integrates it and needs its own run. The
local production planner mirror completed 70 query and model runs
in 10.679 seconds, including the ground truth simulation between
planner calls. The join cells
(`tests/gpu/join_bench.py`, the QUAIL-B evaluation) are the next
step, and the run report's new `kv_manager` block (retained counts,
anchor hits and misses, evictions with their summed value) and
`join_optimizer` block (search size, chosen sequence) are what to
check against the prediction.

The SoL floor does not change. The exact values for all 35 queries
and both models match the previous output field by field. The SoL
script no longer calls or simulates the production planner. Its
full local run decreased from 59.13 seconds to 18.01 seconds.
