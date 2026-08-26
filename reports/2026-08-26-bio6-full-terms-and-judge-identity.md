# BIO-6 back to the full terms table, and a judge identity that only hashes what changes an answer

Status: the labeling pass is running. Setup and prediction below are
fixed; the Result section is filled in from the measured run.

## Problem

The ground-truth labeling job crashed on BIO-6's second join:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 5.41 GiB.
GPU 0 has a total capacity of 79.18 GiB of which 5.38 GiB is free.
```

The fix that shipped in PR #58 cut BIO-6's second join from the full
614-term table to a 64-term subset, dropping that stage from 122,800
pairs to 12,800. That changed the benchmark to fit the labeling job.
It also cut the stage 10x in the one dimension pipelining and
token-based admission exist to speed up, which flatters our own
numbers. It has been reverted. See
`reports/old/2026-08-26-bio6-severe-terms-fix.md`.

Three things were wrong with the original diagnosis:

- It read "halving the batch changed nothing" as evidence that total
  job size was the cause. `rows_per_call(614)` is 1, so on the
  614-term path the job already submitted one report at a time and the
  halving was a no-op.
- vLLM re-chunks whatever it is handed against `max_num_batched_tokens`
  and `max_num_seqs`. The 122,800 pairs were never resident on the GPU
  at once.
- The judge's vLLM settings were copied from the 4B engine runs
  (`baselines/vllm_opbench/config.py`, `tests/gpu/milestone1.py`), but
  the judge loads Qwen3 32B FP8. At `gpu_memory_utilization=0.92` the
  KV reservation left 5.38 GiB free, and a full 25,305-token prefill
  chunk had nowhere to put its activations.

## What changed

1. The judge runs at `gpu_memory_utilization=0.85`, which frees about
   5.5 GiB more on the same H100 - roughly twice the 5.41 GiB
   allocation that failed. `max_num_seqs` went from 2,648 to 4,096;
   neither value binds, because `rows_per_call` caps a submission at
   256 prompts for filters and 614 for the reports-by-terms joins.
2. `severe_terms` is gone. BIO-6's second join and `REACTION_SEVERE`
   read the full `terms` table under a second alias (`m2`), the shape
   IMDB-8 and FEV-7 already use. BIO-6 is character-for-character the
   query PR #51 defined.
3. `JUDGE_SPEC` no longer contains `max_num_batched_tokens` or
   `max_num_seqs`. That dict feeds `JUDGE_ID`, which feeds
   `label_set_id`, which is both a directory name and a stored column
   on every label row. With capacity knobs inside it, an
   out-of-memory fix invalidated all 23 predicates' labels. Those
   knobs cannot change a greedy one-token decode restricted to TRUE
   and FALSE.
4. `judge_pass.rehash_label_sets` moves a label set to its new id
   without calling the model: `example_full_hash` is stored on every
   row, so `judgment_id` recomputes from it.
5. `_parts_stats` and `_compact_label_parts` take the exact list of
   part files the writers produce instead of globbing the label
   directory. A directory can hold more than one generation of parts,
   because a change in filter-group membership or in a join's
   right-hand table moves the boundaries and leaves the older files
   behind. The imdb `F1` directory holds 79 files covering its 5,000
   rows twice, which is what crashed 3 of 4 workload jobs in the last
   pass.

## Setup

- Scale factor 0.1, data seed 20260818, corpus
  `c_df45ef585738f42e4a7a731306f1b9fc` (unchanged by any of this;
  `severe_terms` was never in `CORPUS_COLUMNS`).
- `Qwen/Qwen3-32B-FP8` at revision
  `aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`.
- Four H100s, one per workload, 96 GiB host memory each. Modal app
  `quail-milestone1`, cell `quailb_judge_pass`.
- Superseded collection: `gt_306dac4fc83883c7a5bcc86f4d103f32`.
- New collection: `gt_04231c5de83cdf9e7e68fc03849959d6`.

## Prediction

Stated before the run, from `judge_pass.PREDICTION`:

- 434,201 labels in the finished collection: 394,138 Qwen3 32B
  judgments and 40,063 source labels.
- The 15 filter predicates (17,057 labels) are already on the volume
  under their current ids after the rehash, so this pass writes the 8
  joins: 377,081 model judgments, 63 FEVER annotation labels, and
  LePaRD's 40,000 source labels.
- biodex is the long pole with 245,600 report-length prompts, 1.8x the
  136,200 it ran in 1,056 seconds last pass, so 27 to 35 minutes
  including boot.
- $5 to $9 at current Modal prices.
- 0 answer differences on the deterministic rerun sample.
- No out-of-memory crash at `gpu_memory_utilization=0.85`.

## Result

To be filled in from the run.

## Rehash: what moved without the model

15 filter predicates, 17,057 labels, copied to their new ids with no
GPU time. Verified row by row against the originals: answers,
`example_id`, content hashes, source labels and row counts identical
for all 15, and every `judgment_id` recomputes from its stored
`example_full_hash`.

The 8 join predicates could not be rehashed, and this is not caused by
the BIO-6 change. Their prompts moved in PR #56 (answer-cue strip) and
PR #52 (shared join prompts). The volume's `REACTION` labels carry
`predicate_full_hash` `b61a0026...`; main at `1b6309d` computes
`c710b1b3...` and at `004b142` computes `0d0f4ae0...`. Across all four
collections on the volume, no label set matches the current code for
any join predicate, and exactly one does for every filter predicate.
So the active ground truth's join labels were already stale against
main. Only `REACTION_SEVERE`'s size change is from this work.

## What BIO-6 is, across the three states

| | J1 | J2 | pairs per stage |
|---|---|---|---|
| PR #51, when added (`74c0d8d`) | `terms` (614), REACTION | `terms` (614), REACTION_SEVERE | 122,800 / 122,800 |
| PR #58, reverted here (`1b6309d`) | `terms` (614), REACTION | `severe_terms` (64), REACTION_SEVERE | 122,800 / 12,800 |
| now | `terms` (614), REACTION | `terms` (614), REACTION_SEVERE | 122,800 / 122,800 |

`REACTION_SEVERE` exists because BIO-6's two edges share an anchor and
a table. If J2 also used `REACTION`, its prompts would be identical to
J1's and the second stage would not be a second stage. BIO-D avoids
this without a new predicate because its edges have different operands
(`r1`x`m1`, `r2`x`m1`, `r2`x`m2`). `REACTION_SEVERE` has no BioDEX
label behind it - it is an engine-stress predicate, and BIO-6 should be
read as a throughput and scheduling measurement, not as a real BioDEX
task.
