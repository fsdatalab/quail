# BIO-6's REACTION_SEVERE join: a smaller terms table to fix ground-truth OOM

**Superseded on 2026-08-26.** The fix described here was the wrong one and
has been reverted. Shrinking BIO-6's second join to 64 terms changed the
benchmark to work around a limit in the offline labeling job. The
out-of-memory crash came from the judge's vLLM settings, which were copied
from the 4B engine runs: `gpu_memory_utilization=0.92` left 5.38 GiB free on
an 80 GiB H100 while a 5.41 GiB prefill activation needed room. The judge now
runs at 0.85 and BIO-6's second join is back to the full 614-term table. The
diagnosis below ("the real cause was scale") is wrong: `rows_per_call(614)`
is 1, so the batch-halving test it cites never changed anything, and vLLM
re-chunks whatever it is handed anyway, so the 122,800 pairs were never
resident on the GPU at once. See
`reports/shipped_features/2026-08-26-judge-identity-and-bio6-full-terms.md`.


## Result

`REACTION_SEVERE` (BIO-6's second join) now has ground truth, and all 4 predicates needed for the new multi-join queries are labeled: `ASPECT_SENTIMENT`, `ASPECT_RELATED`, `REACTION_SEVERE`, `REFUTE`.

| predicate | workload | pairs | TRUE | % TRUE |
|---|---|---|---|---|
| `ASPECT_SENTIMENT` | imdb | 60,000 | 4,086 | 6.81% |
| `ASPECT_RELATED` | imdb | 144 | 0 | 0% |
| `REACTION_SEVERE` | biodex | 12,800 | 1,158 | 9.05% |
| `REFUTE` | fever | 5,700 | 201 | 3.53% |

Each row above was verified by reading every saved answer file directly and counting rows and TRUE answers by hand, not by trusting the code's own completeness check (see "Known separate issue" below for why that check can't be trusted right now).

The committed aggregate summary is `results/benchmark/20260826T040743Z-new-multijoin-predicates-ground-truth.json`. The raw per-row data is on the `quail-results` Modal volume, one path per predicate, listed in that summary.

Figure: plots/20260826T040743Z-new-multijoin-predicates-ground-truth-selectivity.png

These numbers differ from ones mentioned earlier in this work: an earlier run this session used QUAIL-B's templates from before PR #56 ("Strip the redundant answer cue from the QUAIL-B templates"), which is now fixed and no longer matches the current, correct templates. Every join predicate's answers shift meaningfully under that fix — even `DISCUSS_ASPECT`, an existing predicate untouched by this PR, moved from 1.9% to over 5% TRUE. The numbers above are labeled under the current, correct templates.

## Problem

`REACTION_SEVERE` asks, for a report and a candidate reaction term, whether the report describes that term as a severe or life-threatening occurrence. BIO-6 joins it against the full `terms` table (614 terms) under a second alias, same as its first join (`REACTION`).

Generating ground truth for that full cross product — 200 reports x 614 terms, each report up to about 15,000 tokens — crashed with a CUDA out-of-memory error every time:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 5.41 GiB.
GPU 0 has a total capacity of 79.18 GiB of which 5.38 GiB is free.
Process 1 has 73.79 GiB memory in use.
```

Two different fixes were tried and ruled out, because the crash reproduced with identical numbers regardless:

- Halving the batch size (4 reports per call to 2).
- Resetting the model's KV cache between workloads.

The real cause was scale: 200 reports x 614 terms is 122,800 pairs, and admitting that many long-report prompts at once exceeded vLLM's fixed KV cache reservation, independent of batching.

## Fix

`REACTION_SEVERE`'s question is also only meaningful for terms a report actually discusses — asking "was this severe" about a term the report never mentions is close to a trivial FALSE. So instead of shrinking the batch further, `severe_terms` was added: the 64 most common reaction terms (already frequency-sorted by the existing vocabulary-building code), a fixed subset of `terms`. BIO-6's second join and `REACTION_SEVERE`'s ground truth now both use `severe_terms` instead of `terms`, cutting the cross product to 200 x 64 = 12,800 pairs.

One thing this fix had to avoid: the ground-truth pipeline tracks a `corpus_id`, a fingerprint over the source tables, which determines whether a predicate's existing labels can be reused. Adding a new table to that fingerprint's table list would have changed `corpus_id` for every predicate, forcing every one of the other 22 to be relabeled from scratch. `severe_terms` is deliberately left out of that fingerprint and read separately, so only `REACTION_SEVERE` needed new work.

## Setup

- Scale factor: 0.1.
- Corpus id: `c_df45ef585738f42e4a7a731306f1b9fc` (unchanged by this fix).
- Model: `Qwen/Qwen3-32B-FP8` at revision `aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`.
- Hardware: one H100 per workload (imdb/biodex/fever/lepard run as separate, parallel containers, per PR #57).
- Modal app: `quail-milestone1`.
- Cell: `quail.bench.judge_pass.judge_workload`.

## Known separate issue

The labeling job's completeness check (which counts saved rows per predicate and compares to the expected count) is currently unreliable for predicates that have been relabeled more than once across the project's history: `F1`, `F7`, `SUPPORT`, `LEP1`, `REACTION`, and `DISCUSS_ASPECT` all have leftover answer files from earlier code versions sitting alongside current ones, so counting them together comes out inflated (in most cases exactly double). This crashed 3 of the 4 workload jobs (`imdb`, `fever`, `lepard`) during this work, and prevented the ground-truth collection from being finalized and activated.

This is not caused by anything in this PR - none of the affected predicates were touched here, and the bug predates this change. The 4 new predicates in this PR are unaffected because they've only ever been generated once, under the current code, so there's no earlier-version data to conflict with. Fixing the underlying issue (cleaning up or deduplicating the stale files) is a separate task.
