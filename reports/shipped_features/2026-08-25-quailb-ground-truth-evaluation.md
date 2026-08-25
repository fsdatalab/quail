# QUAIL-B ground truth and evaluation

## What changed

QUAIL-B now reads saved ground truth and reports accuracy for every query.
The evaluator reports answer accuracy and final-row precision, recall, F1,
and exact match. It also reports runtime, tokens, H100 cost per token, and
documents per second for each query.

The ground-truth pass uses Qwen3 32B for predicates without exact source
labels. FEVER annotations provide 63 labels. LePaRD passage IDs provide
40,000 labels. BioDEX reactions are relabeled by Qwen3 32B because the source
reaction field is not accepted as truth.

All 26 current queries have complete ground truth. The collection contains
245,557 labels across 19 predicates. Qwen3 32B supplied 205,494 labels, and
the two exact source rules supplied 40,063 labels.

## Why

The benchmark previously reported runtime and token counts without measuring
whether a query returned the correct rows. It also had no stable process for
labeling a predicate added later.

The saved label IDs now include the corpus, prompt, input roles, and label
source. A repeated run skips complete label parts. A completed collection is
marked active, so the evaluator selects the new collection when predicates
are added or changed.

## Files

- `quail/bench/README.md` explains full, one-query, and new-label runs.
- `reports/2026-08-25-quailb-qwen32b-ground-truth.md` records the labeling
  experiment and measured cost.
- `results/benchmark/20260825T082145Z-quailb-qwen32b-ground-truth-sf0.1.json`
  is the committed aggregate summary.
- Raw labels remain under `/results/ground_truth/quailb/schema_v1` on the
  `quail-results` Modal volume.
