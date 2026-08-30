# KV regret in every benchmark report

Date: 2026-08-30

## What changed

- The engine worker (`quail/runtime/worker.py`) now counts KV regret
  for every query and returns `regret_tokens` next to `fresh_tokens`.
  KV regret is the fresh tokens spent recomputing a document prefix
  whose KV the same query already computed once; with unlimited KV
  space every regret token would have been a hit. Both the single-GPU
  path and the per-GPU child path count it: a `seen` set records
  every arena key computed in any phase, and before each join round
  every anchor is classified as resident (hit), seen-but-evicted
  (regret, counted at its prefix length), or first computation.
- The stock vLLM benchmark runner (`baselines/stock_vllm/run.py`)
  computes the matching number from vLLM's own per-request
  `num_cached_tokens`. A `prior` dict maps (alias, global row) to the
  prompt token-id lists that computed that document's KV earlier in
  the query. Per join request, the would-be hit is the longest common
  token prefix with any earlier prompt for the same document (pairs
  after the first can hit the whole prefix their pair 0 computed),
  rounded down to the cache block size; regret is the part vLLM did
  not actually serve. Each join step and each per-query entry
  reports `regret_tokens`.
- Session reports (`quail/runtime/session.py`) and QUAIL-B benchmark
  rows (`quail/bench/quailb.py`) carry the field through, so every
  benchmark query reports regret with no profiler attached.

## Why

- Regret separates avoidable recompute (bad retention) from
  unavoidable first computations. Until now it existed only in the
  standalone instrumentation cells (`ablations/profile_quail.py`,
  `ablations/profile_stock.py`), which are run on demand per query.
  The 2026-08-30 discrepancy report needed those cells to learn that
  pre-fix Quail wasted 1,220,547 regret tokens on IMDB-3 and stock
  vLLM 1,326,432; now that number comes out of every benchmark run.
- Filters never produce regret - every document there is a first
  computation - so the accounting only runs at joins.

## Cost and numbers

- Bookkeeping is one set insert per document per phase in the engine,
  plus one pass over join anchors; the stock runner additionally
  keeps surviving documents' prompt token ids and computes one
  longest-common-prefix per anchor, outside the timed step walls. No
  timing change expected, and this change ships no new measurements.
