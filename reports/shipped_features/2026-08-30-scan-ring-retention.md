# KV retention that cannot starve admission

## What changed

The 2026-08-30 discrepancy report showed why IMDB-3 was the one
QuailB query Quail lost to stock vLLM: retained survivor KV was
allowed to fill the arena down to one document of headroom, while
the filter loop needs about two chunk budgets of in-flight document
KV. Once the arena filled, every admission first evicted one
retained document through the blocked-admission path, and the
filter collapsed into one-document forward passes. Three changes:

- `run_filter` (`quail/executor/loop.py`): before a filter with
  retention starts, reserve pages for two chunk budgets of document
  KV - the scan ring - and cap retained KV at what is left. If an
  earlier operator's retained KV crowds the ring, evict the prefixes
  with the fewest tokens per page once, in bulk, up front. The old
  one-victim-at-a-time eviction inside blocked admission remains
  only as a safety valve.
- `RetainedPool` (`quail/executor/retention.py`): passing survivors
  are offered to a fixed-capacity pool. While it has room, every
  offer is kept. Once full, the newcomer replaces the residents
  with the fewest prefix tokens per page only when it contains more
  prefix tokens than the victims contain together. Total retained
  prefix tokens only rise, and equal token counts never swap.
  Replacement happens in the answer path, never in admission.
- Planner keep credit (`quail/planner/decide.py`): the working
  headroom the credit reserves is now the same two chunk budgets
  (220,752 tokens at 4B), replacing the one-document headroom
  (3,389 tokens on IMDB-3) that priced 358,861 of the 362,250-token
  arena as resident. The credit represents a fraction of survivors
  with the corpus length distribution.

`FilterAdmission.report` (`quail/executor/pack.py`) also credits a
kept document's rewound question-tail pages back to the admission
pool at the keep, instead of stranding them.

## Why

Retention is optional work; admission is required work. The old
design let the optional work take pages the required work needed
back, and the reclaim path freed one blocked document's shortfall
at a time. On IMDB-3 that cost 35.2 seconds of collapsed batches to
save 1.3 seconds of recompute. The ring makes the two claims
disjoint: admission owns its working set for the whole scan, and
retention competes only with itself, by exact prefix tokens.

## Before/after numbers

IMDB-3 (F1 filter into the reviews x aspects join), sf 0.1, Qwen3
4B fp8, one H100, measured by the same instrumented cell as the
discrepancy report (`experiments/discrepancy_timeline.py`, since
generalized into `experiments/profile_quail.py`,
`--out-prefix ringfix_tokens_head`):

- Filter phase: 50.28 s in 2,922 forward passes (602 mean tokens)
  before; 14.59 s in 17 passes (103,484 mean tokens) after - the
  same 1,759,233 fresh tokens, and now level with IMDB-1's 15.06 s
  for the identical work.
- Eviction calls during the filter: 2,625 before; 0 after.
- Join phase: 18.53 s before, 18.15 s after.
- Engine wall: 68.93 s before, 32.79 s after (2.10x), against the
  52.65 s recorded for stock vLLM on this query - the one QuailB
  loss is gone.
- Retained at the join: 309 documents and 140,212 prefix tokens on
  8,843 pages, exactly the pool cap. All 309 hit. Regret stayed at
  1.22M tokens; the
  same 4,380 survivors, so answers are untouched.
- BIO-2 unchanged: 128.22 s after against 130.35 s before, regret
  0, no evictions (its plan retains nothing).

Full report with the plot and volume paths:
`reports/2026-08-30-kv-ring-fix.md`.
