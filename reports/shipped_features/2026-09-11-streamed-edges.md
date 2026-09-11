# Survivor streams are the normal edge between GPU operators

Date: 2026-09-11

## What changed

- The physical operators are named for what they are: `Scan`,
  `AiFilter`, `AiJoin`, `Barrier`, `Exchange`, `Recombine`, `Project`,
  and `Limit`. `DocumentInput`, `PackedFilter`, and `AnchoredJoin` were
  renamed, and the old `Exchange` node, which materialized survivors,
  pruned them with finished join answers, and switched the anchor, is
  now `Barrier`. Type names changed with them (`quail.scan`,
  `quail.ai_filter`, `quail.ai_join`, `quail.barrier`).
- A new `Exchange` node appears only in plans for several GPUs, once
  before each join group. It names the alias whose documents go to the
  GPU that holds their KV. On one GPU it passes the ids through; the
  routing itself is still done by the coordinator's join round.
- An `AiFilter` that feeds the join anchored on it now returns its
  survivor port as a `SurvivorStream` and a `finalize` callable, instead
  of running to completion. The consuming `AiJoin` drives the chain one
  chunk at a time and writes what it learned into the stream's holder;
  the runner calls `finalize` after the whole graph has run and stores
  the filter's answers and metrics as its own result. The runner no
  longer has a streamed-edge special case: `stream_anchor`,
  `StreamedInput`, `NodeResult.produced`, and the deferred-producer
  logic are gone.
- The planner records the edge kind on the producer: `AiFilter.pin_survivors`
  says its survivors stay pinned for a join, and `hold_tokens` is the
  join's largest frame, so a pinned document's pages already cover it.
  Both are derived from the graph shape, not chosen.
- The multi-GPU path follows the same contract: the coordinator's
  `AiFilter` execution returns a stream when survivors are pinned, and
  the join round fills its holder with the merged answers.

## Why

- The table of operators in the design discussion is now the table of
  operators in the code, and an edge between two GPU operators streams
  by default. Only `Barrier` materializes.
- Removing the special case makes the next step, a caller-driven
  `Handoff` barrier, an ordinary node.

## Numbers

- No execution change: the chain and join drivers are the ones measured
  in [the streamed filter-join report](../2026-09-11-streamed-filter-join.md).
  CONFIRM_PLACEHOLDER
