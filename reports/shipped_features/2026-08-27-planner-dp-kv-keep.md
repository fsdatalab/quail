# One join search, run on estimates and on survivors, over retained KV

## What changed

Three pieces, landed together. This merges the two development
branches that split from the SoL work: this branch's plan-time cost
model and multi-GPU coordination, and sol-computation-math's
retention runtime and post-filter planning.

1. **One join search** (`quail/planner/joins.py`): the left deep
   subset DP over (joined alias set, cached prefix alias set), the
   same search the SoL estimate runs. Stage costs are `Work` records
   (tokens, attention pairs, KV written, KV read) built per document
   over the live length lists: a resident anchor prefix pays its
   question frame only, the rest scan preamble + document + frame.
   Gates and n-ary predicates are searched like everything else;
   forced anchors are honored; candidates rank by speed-of-light
   seconds from counted model constants and the device datasheet. No
   calibration constant is read anywhere.

2. **Called twice per query.** `plan_query` calls it with expected
   live counts and the keep credit, and emits the predicted plan -
   explain(), refusals, sharding, and the SoL comparison run off it.
   After the filter round the worker (the parent process on several
   GPUs) calls the same function with the actual survivors and the
   KV actually resident, and executes its answer. The runtime call
   wins wherever they disagree, because it has real data. Barriers
   reuse its answer: no order-relevant information arrives below a
   stage boundary, so nothing is re-decided per chunk, and the old
   `pick_runtime_anchor` heuristic is deleted as subsumed. Stage
   outputs carry written_pos, semantics, and selectivity, so a
   runtime-chosen order assembles into results correctly.

3. **Retained KV across operators** (`executor/arena.py`,
   `executor/retention.py`): every filtered alias the search could
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
retained KV starves a required admission, the arena evicts the
minimum-loss victim set - the least total recompute-seconds cover
for exactly the pages needed (`minimum_loss_victims`, state capped
at the pages needed). The value of a prefix of length L is L dense
tokens against the fp8 peak plus L(L+1)/2 attention pairs against
the bf16 peak - both counted from the architecture and the
datasheet. The linear dense term dominates below the crossover
(about 12,320 prefix tokens at 4B, 29,760 at 32B), where most
benchmark documents sit. Pinned keys - in use by the running
operator - are untouchable.

At plan time the same idea appears as a credit, not a rule:
`keep_split` prices in the expected resident fraction the arena can
hold, longest documents first - the fractional knapsack take, since
survival is fractional in expectation; optimal in bytes, an
estimate under page rounding. The runtime is not bound by the
threshold; the credit keeps the prediction and the SoL comparison
honest. The search itself grants residency credit only while what
a candidate plan holds - retained aliases some remaining stage can
still anchor, beside the stage's largest tuple - fits the arena:
past a plan's first stage over that budget, every prefix prices as
a scan, so re-using an anchor after an intervening group that
could not have kept it is not priced too cheaply.

## Why

The SoL accounting credited filter-to-join KV reuse the engine did
not perform, and the committed SoL report
(`reports/2026-08-26-sol-quailb.md`, data at
`/results/sol/sol_quailb_sf0.1.json` on the quail-results volume)
measured the plan-choice half of the remaining gap: seven of nine
multi-join queries improve under the exact left deep search, FEV-8
most, 0.900 s to 0.787 s at 4B and 7.460 s to 6.520 s at 32B. The
engine now performs the reuse the model assumed, and decides its
join plan from the same search with the best information available
at each moment: estimates before anything runs, exact survivors
after the filters.

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
in; the merged worker re-integrates it and needs its own run. Two
more things no run has measured: the search's capacity check on
residency credit is all-or-nothing past a plan's first overflow (it
does not model which victim the minimum-loss eviction picks, so
under mild pressure it can still misprice in either direction), and
the eviction cover's CPU cost under a full arena. The join cells
(`tests/gpu/join_bench.py`, the QUAIL-B evaluation) are the next
step, and the run report's new `kv_manager` block (retained counts,
anchor hits and misses, evictions with their summed value) and
`join_optimizer` block (search size, chosen sequence) are what to
check against the prediction.

The SoL floor does not change. `simulate_query` in
`reports/make_sol_quailb.py` now runs the worker's own search on the
exact filter survivors and follows its retention lifecycle (no
eviction pressure assumed - stated in the JSON), so the
current-planner column is the engine's behavior by construction.
The plan-choice part of its gap to the optimal should close on
regeneration; what remains is expectation error inside the runtime
search's later-stage thinning and any eviction pressure the
simulation does not model.
