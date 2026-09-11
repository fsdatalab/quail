# Multiple pairwise joins, the rest: plan graph, joint search, barriers

Date: 2026-08-24. Slices 2-5 of issue #38, on top of the first slice
(same PR). The shared-anchor requirement is gone: any connected set
of full joins plans and executes, on one GPU or several. Gated by
the CPU suite plus one GPU smoke of the barrier path (below).

## What changed

- **The physical plan is a dataflow graph, not a flat list**
  (`planner/plan.py`, `planner/decide.py`). `PhysicalPlan.nodes`
  holds DocScan, FilterChain, JoinGroup, Barrier, Recombine, and
  Sink nodes; each names its inputs as (producer id, port) edges.
  Exactly two things flow on edges: a table's live document ids and
  one stage's passing pairs. The worker executes the structure the
  planner emitted - the worker's own grouping code (`_stage_groups`)
  is deleted, so groups and barriers have one authority.
  `explain()` prints every node with its edges.
- **Order and anchors are one search** (`plan_joins` in
  `decide.py`). The planner costs every (stage order, anchor per
  stage) sequence with a live-count walk: a group-opening stage pays
  the anchor's KV, a group-continuing stage pays only its naming
  line, every stage pays its pair stream, and the gate rule thins
  the live counts. Anchor switches in the cheapest sequence become
  groups and Barrier nodes. This replaces `choose_anchor`,
  `choose_shared_anchor`, and `order_joins`, and removes both
  NotImplementedError paths (no shared table; forced anchors that
  split the stages).
- **Barriers execute** (`backends/quail/coordinator.py`,
  `backends/quail/distributed.py`). At a Barrier the live sets thin to the
  documents in some surviving pair of every finished full stage
  (`thin_survivors` - cost only; results are enforced at
  recombination). The multi-GPU coordinator runs one round per
  JoinGroup node; an anchor with no filter shard gets fresh balanced
  shards over its live documents (`join_group_payloads`). The
  re-shard moves no KV between GPUs: a partner table never owned
  any, so only token ids ship and each GPU computes its new anchor
  slice fresh.
- **Barrier-time re-planning** (`pick_runtime_anchor` in
  `decide.py`). A one-stage group with no user-forced anchor
  re-picks its anchor from the measured live counts, restricted to
  anchors whose worst-case tuple fits the chunk budget. The payload
  ships block labels and naming lines for every table of a stage
  (`frames`/`labels` maps), so the re-pick needs no re-tokenization;
  `stage_for_anchor` materializes the child-facing spec for
  whichever anchor a round uses.
- **Recombination generalizes to per-stage anchors**
  (`runtime/session.py`, `_assemble`). Output tuples come from
  equi-joining every full stage's passing pairs on whatever aliases
  a stage shares with the assignments built so far - each stage
  keyed on its own anchor. The survivor-set checks (filters,
  exists/anti gates) and the LIMIT-on-final-rows rule (#39) are
  unchanged.

## Why

Issue #38: a chain like `ai(a,b), ai(b,c), ai(c,d)` has no table in
every predicate, so no single anchor exists - it needs groups and
barriers. And the flat operator list could not carry a barrier's
edge ("the live ids of table c, measured from group 1's pairs"), so
the plan had to become the graph the execution actually follows.

## Not in this PR

Re-ordering the remaining stages at a barrier (the plan's stage
order is fixed at compile time; only a one-stage group's anchor and
every group's shards are re-decided from measured counts).

## Benchmarks

The measured cells for issue #38's benchmark slice - the gate
against its formula, and the re-shard trade against the forced
shared-anchor baseline - are in
`reports/2026-08-24-join-gate-and-reshard.md` (removed 2026-08-29; git history) (script
`experiments/cells/join_bench.py` (removed 2026-08-29; git history), summary `results/join_bench.json` (removed 2026-08-29; git history)).
Headlines: gate mechanics exact and bit-for-bit reproducible; the
two-group barrier plan measured 11.3x fewer tokens and 11.3x faster
than the forced shared anchor, with plan-predicted token counts
matching measurement to 0.03%; and an accuracy finding - anchor
orientation flipped a planted stage from exact to all-TRUE - that
feeds issue #43's orientation check.

## GPU smoke of the barrier path

`experiments/cells/barrier_smoke.py` (Modal, qwen3-4b-fp8): a filter on 10
reports, then `ai(r, c)` anchored r and `ai(c, g)` anchored g -
forced anchors, so the plan is two JoinGroups with one Barrier.
Reports draw from only 4 of the 6 colors while candidates cover all
6, so the unused colors' candidates can match no report and the
thinning check cannot pass vacuously; the run fails outright if
every candidate survives the barrier. Run on 1 GPU (the worker's
plan-node walk and thinning) and 2 GPUs (per-group rounds, the
parent's thinning, and the re-shard of the unfiltered stage-2
anchor). Prediction, stated before the run: the returned rows equal
a CPU brute-force recombination of the worker's own answer rows on
both GPU counts; if the model matches the planted truth, 8 of 12
candidates survive the barrier and stage 2 evaluates 6 x 8 = 48
tuples.

Result, both GPU counts: 14 rows, exactly matching the brute-force
recombination; 8 of 12 candidates survived the barrier (the planted
count exactly); stage 2 evaluated 48 tuples = 6 labels x 8 thinned
candidates. The model matched the planted truth 14 of 14 with both
stages at the planted 1/6 selectivity. The store worked across the
barrier: the 7 filter-surviving report documents restored in the
join round instead of recomputing, on both GPU counts. Summary
committed as `results/barrier_smoke.json`; the teed log is
`results/barrier_smoke.log` (not committed, like other logs).

## Two review fixes, after that run

- The re-shard condition was wrong. `join_group_payloads` followed
  any shard present in the payload, and engine payloads ship a scan
  shard for every alias, so the fresh balance over live documents
  never ran: an unfiltered anchor kept its static scan shard however
  the live set thinned, and after heavy thinning one GPU could hold
  every live anchor. The condition is now "the anchor ran the filter
  round" - that is the only case where a GPU already holds its KV -
  and any other anchor balances fresh over its live documents.
  Results were never affected, only load balance. Unit test: an
  unfiltered anchor whose whole live set sat in one scan shard now
  splits across workers (`test_coordinator.py`).
- The smoke's thinning gate could not fail on that corpus draw: all
  12 candidates had a stage-1 match, so "stage 2 evaluated labels x
  thinned candidates" held with nothing thinned. Reports now draw
  from only 4 of the 6 colors while candidates keep all 6, so the
  unused colors' candidates can match no report, and the run fails
  outright if every candidate survives the barrier. The re-run
  measured exactly the prediction from the drawn seed: the same 7
  filter survivors, 8 of 12 candidates surviving the barrier, stage
  2 at 6 x 8 = 48 tuples, and 14 rows equal to brute force on both
  GPU counts - the smoke section above reports it.

## Tests

`uv run pytest tests/ -q` from `quail/`: 181 passed. New coverage:
the joint search splitting a chain into groups when the long side
anchors and keeping one group when the shared table is longest, a
three-join chain planning two groups and one barrier, forced-anchor
remarks, `pick_runtime_anchor` cost and chunk-fit behavior, the
barrier's thinning (single stage and intersection across stages),
`gate_group`, `stage_for_anchor`, the re-shard of an anchor with no
filter round (both with no shard entry at all and with a scan shard
present that must be ignored), `derive_plan_nodes`, and end-to-end
recombination across a barrier checked against the shared-anchor
chain's expected triples.
