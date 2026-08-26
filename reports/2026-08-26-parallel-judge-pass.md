# Regenerating QUAIL-B ground truth: four containers, and a cue that was suppressing positives

## Result

Two changes landed together in this run, and only one of them was
supposed to matter.

The intended change was mechanical: the judge pass now runs one Modal
container per workload instead of one container for all four. That cut
the wall time from a reconstructed 43.0 minutes to 33.8 minutes, a
speedup of 1.27x. I predicted 1.7x and was wrong about why.

The change that mattered was the templates. PR #56 removed a trailing
`ANSWER=` from all 24 QUAIL-B templates, which had been duplicating the
`\nANSWER:` the engine already appends. Relabelling with the corrected
templates moved 14 of the 19 predicates, every one of them toward more
positives, and three of them by more than 2x. The BioDEX reaction join
went from 81 true pairs out of 122,800 to 19,777, a factor of 244.

The 2026-08-25 report flagged that 0.066% positive rate as something to
audit before treating the labels as ground truth. The duplicated cue was
the cause.

The new collection is
`gt_80e7582534b349bc61c087595f2e0a51` and is the active ground truth for
corpus `c_df45ef585738f42e4a7a731306f1b9fc`. The committed summary is
`results/parallel_judge_pass_sf0.1.json`. The 245,557 labels themselves
are on the `quail-results` volume at
`/results/ground_truth/quailb/schema_v1/collections/gt_80e7582534b349bc61c087595f2e0a51`.

## Setup

- Scale factor 0.1, corpus seed 20260818, corpus
  `c_df45ef585738f42e4a7a731306f1b9fc`.
- `Qwen/Qwen3-32B-FP8` at revision
  `aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`.
- Four H100s, one per workload, 96 GiB host memory each. Modal app
  `quail-milestone1`, cell `quailb_judge_pass`.
- Temperature 0, one output token, only `TRUE` or `FALSE` permitted.
- Corpus built once on a CPU container and read by all four GPU
  containers, so every container judges the same rows.
- Finalize call `fc-01M0XP3ENNX7XX5HNVXR3378K8`.

## Prediction

- 205,457 Qwen judgments and 40,100 source labels, 245,557 total.
- 30 to 45 minutes of wall time including model load.
- About 1.7x faster than one container doing the same work.
- 0 answer differences on the deterministic rerun.

## Measured

### Label counts matched

| | predicted | measured |
|---|---|---|
| total labels | 245,557 | 245,557 |
| Qwen3 32B judgments | 205,457 | 205,494 |
| source labels | 40,100 | 40,063 |
| answer differences on rerun | 0 | 0 of 288 |

The 37-label gap is the same one the 2026-08-25 report found: only 63 of
the 100 sampled FEVER claims have their annotated page inside the
57-page evidence pool, so Qwen judged the other 37 pairs rather than
taking a FEVER annotation. My prediction counted 100. The total is
unaffected because the same rows are labelled either way.

The rerun re-submits 288 saved prompts in reverse order and compares.
Nothing changed, so the pass is repeatable. That says nothing about
whether the answers are right.

### Wall time was in range, the speedup was not

Figure: plots/judge_pass_wall.png

| workload | model load | judging | wall |
|---|---|---|---|
| BioDEX | 656.3 s | 1,374.3 s | 2,030.7 s |
| IMDB | 280.9 s | 707.9 s | 988.8 s |
| FEVER | 331.4 s | 187.5 s | 518.9 s |
| LePaRD | 279.3 s | 30.7 s | 309.9 s |

33.8 minutes against a predicted 30 to 45. But 1.27x, not 1.7x, for two
reasons.

BioDEX is 60% of the judging on its own: 1,374 of 2,300 seconds. Even
with a free model load, splitting this work four ways cannot beat 1.56x.
The 1.7x I predicted was never available.

The rest is model load contention. BioDEX took 656 seconds to load
against 279 for LePaRD and 281 for IMDB. Four containers pull the same
32B checkpoint off the shared Hugging Face cache volume at the same
time, and the loser waits an extra 6.3 minutes. That sits directly on
the critical path.

The serial figure of 43.0 minutes is reconstructed as one model load
(279 seconds, the least contended) plus all four workloads' judging
time. The measured serial run this replaces took 58.2 minutes, but it
stopped partway and resumed, so it is not a fair baseline.

Parallelism also costs GPU time rather than saving it: 3,848 GPU-seconds
against a reconstructed 2,580. That is 49% more compute to remove 9
minutes of waiting.

### Removing the cue moved 14 of 19 predicates

Figure: plots/judge_pass_selectivity.png

| workload | predicate | rows | before | after | change |
|---|---|---|---|---|---|
| BioDEX | REACTION | 122,800 | 0.07% | 16.11% | 244x |
| FEVER | SUPPORT | 5,700 | 0.63% | 2.61% | 4.1x |
| IMDB | DISCUSS_ASPECT | 60,000 | 1.90% | 5.54% | 2.9x |
| LePaRD | LEP3 | 200 | 5.00% | 9.50% | 1.9x |
| LePaRD | LEP4 | 200 | 6.50% | 9.00% | 1.4x |
| LePaRD | LEP5 | 200 | 3.00% | 3.50% | 1.2x |
| FEVER | F11 | 100 | 59.00% | 64.00% | 1.08x |
| IMDB | F1 | 5,000 | 77.46% | 80.08% | 1.03x |
| LePaRD | LEP2 | 200 | 51.50% | 54.00% | 1.05x |
| IMDB | F4 | 5,000 | 23.10% | 24.36% | 1.05x |
| BioDEX | F7 | 200 | 59.50% | 61.50% | 1.03x |
| BioDEX | F8 | 200 | 68.00% | 70.00% | 1.03x |
| BioDEX | F9 | 200 | 63.00% | 64.00% | 1.02x |
| IMDB | F5 | 5,000 | 56.78% | 57.08% | 1.01x |

F12, F13, LEP1 and LEPS1 did not move. LEPJOIN did not move either, and
could not: it is labelled from LePaRD's own passage ids and never
reaches the model. That it came out identical is a check that the corpus
and the pairing are unchanged.

Two things stand out.

Every change went the same direction. Fourteen predicates moved and all
fourteen produced more positives. A prompt edit that shifted answers at
random would not do that. The duplicated cue was pushing the model
toward `FALSE`.

The joins moved far more than the filters. Filters shifted by 0.3 to 2.6
percentage points. The three model-judged joins shifted by 2.0, 3.6 and
16.0 points. I do not know why, and this run cannot tell me: measuring
it needs the `TRUE`/`FALSE` logit split on the same prompts with and
without the cue, which I did not record. Until that is run, treat the
size of the join effect as observed rather than explained.

## What this means

The SoL analysis in `reports/2026-08-25-sol-quailb.md` is built on these
selectivities and is now out of date. Stage-1 costs are unaffected,
since they depend only on document lengths, but every downstream stage
in a filter chain sees a different number of surviving documents, and
the join selectivities changed by up to 244x. That analysis needs
rerunning against this collection.

Two predicates still need a human check before these labels are treated
as settled:

- The BioDEX reaction join is now 16.11% positive. That is a plausible
  rate where 0.066% was not, but nobody has read the pairs.
- LePaRD's `states_general_rule` is 200 true out of 200, unchanged. With
  no negatives it cannot measure filtering quality as written.

Two fixes worth making before the next full relabel:

- Stagger the container starts, or warm the checkpoint on one container
  before the others begin. That recovers about 6 minutes.
- Shard the BioDEX join. It is the only thing setting the wall time, and
  its 200 report anchors split cleanly across containers. Four shards
  would put BioDEX at about 10 minutes, at which point IMDB's 16.5
  minutes becomes the wall. Treat that as an upper bound on the gain:
  seven containers pulling one checkpoint would make load contention
  worse than the 6.3 minutes measured here.

## Reproducing

    uv run modal run -m quail.bench.judge_pass \
        2>&1 | tee results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-label.log

    uv run --with matplotlib python reports/make_parallel_judge_pass_plots.py

Superseded collection, kept on the volume:
`/results/ground_truth/quailb/schema_v1/collections/gt_42674891c824e01c6d966eb48c9cf8c7`.
