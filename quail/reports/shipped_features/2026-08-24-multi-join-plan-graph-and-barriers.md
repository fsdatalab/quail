# Multiple pairwise joins, the rest: plan graph, joint search, barriers

Date: 2026-08-24. Slices 2-5 of issue #38, on top of the first slice
(same PR). CPU-side code and CPU tests only; no engine run, so no
measurements in this note. The shared-anchor requirement is gone:
any connected set of full joins plans and executes, on one GPU or
several.

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
- **Barriers execute** (`runtime/coordinator.py`,
  `runtime/worker.py`). At a Barrier the live sets thin to the
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
Benchmarks for the chain and star shapes, and one case where
token-length asymmetry makes the re-shard win, are issue #38's last
slice and need engine runs.

## Tests

`uv run pytest tests/ -q` from `quail/`: 180 passed. New coverage:
the joint search splitting a chain into groups when the long side
anchors and keeping one group when the shared table is longest, a
three-join chain planning two groups and one barrier, forced-anchor
remarks, `pick_runtime_anchor` cost and chunk-fit behavior, the
barrier's thinning (single stage and intersection across stages),
`gate_group`, `stage_for_anchor`, the re-shard of an anchor without
filter shards, `derive_plan_nodes`, and end-to-end recombination
across a barrier checked against the shared-anchor chain's expected
triples.
