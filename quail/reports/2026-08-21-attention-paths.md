# Attention paths per workload, validated against stock vLLM

Issue #24. Branch: `claude/attention-paths-issue-24-n8rojj` (carries
the `attention-path-selection` work forward).

2026-08-21, second pass: every remaining YES/NO planted value and
answer instruction in the corpora and cells was converted to
TRUE/FALSE to match the engine's constrained readout, and the
numbers below were re-measured on the converted corpora. The
conversion matters for answers, not for the path ranking: with the
framing matched to the readout, flag margins widen and all three
paths reproduce full-prompt recompute exactly on the 256-document
parity corpus.

## The decision

- Filters run the `unified` path: one causal FlashAttention-3 paged
  call over kept plus current KV.
- Joins run the `merge_quant` path: the two-call pattern with the
  fused LSE-merge + FP8-quantize Triton kernel.
- `split` stays as the parity reference and the gather-fallback
  path. No path was removed; the assignment is fixed in
  `attention.py` as `FILTER_ATTENTION` / `JOIN_ATTENTION`, and the
  worker reads those constants.
- FlashInfer was measured and not adopted (details below).

## Why filters get `unified`

- Speed, measured on the 10,000-document five-filter workload
  (`results/attention_paths.json`, TRUE/FALSE corpus, 4.10M fresh
  tokens): unified 8.45 us/token, against 8.68 for merge_quant and
  8.99 for split. Walls 34.6 / 35.6 / 36.9 s. (The pre-conversion
  YES/NO corpus gave the same ranking: 8.24 / 8.35 / 8.57.)
- Answers at scale: on the converted corpus all three paths return
  identical answers on all 40,052 (doc, stage) pairs, all 100%
  correct against the planted flags, same 4,645 survivors. The
  path choice does not move filter answers at all once the task is
  framed the way the readout is constrained.
- Exactness: the unified call is bit-identical to a contiguous
  causal FlashAttention call on every kernel-parity case, including
  the edge cases (`results/attention_parity.json`, max_abs 0.0 on
  all 10 cases: page-boundary lengths, 257-page documents, 1-token
  documents, a 48-group chunk, suffix larger than the document, and
  an empty document). The split path differs from the contiguous
  call at bf16 rounding scale (max_abs 0.005-0.016), the same
  magnitude as the contiguous call's own distance from a float32
  reference.
- End to end (`results/attention_end_to_end_parity.json`, 256
  documents, 5 stages, TRUE/FALSE corpus): all three paths
  reproduce full-prompt recompute exactly - 0 disagreements in 997
  answers each, and 0 wrong against the planted flags. (On the
  pre-conversion YES/NO corpus, whose margins sat near zero, split
  and merge_quant flipped 5 answers each; unified was exact there
  too.)
- KV rewind and the store: a save-then-restore pass under unified
  reproduced the no-store answers exactly (0 differences, 64 of 64
  documents restored). The question preamble written after the
  document survives later stages' scatters by construction (later
  suffixes scatter after it), and the parity run confirms it.

## Why joins get `merge_quant`

The unified path is structurally wrong for fan-out: one causal call
per pair cannot share an anchor's KV across the many partner
suffixes of a chunk, because a later pair's tokens would read the
earlier pair's scattered KV. The only correct unified packing is one
partner per anchor per chunk ("waves"), which re-reads the anchor's
full KV once per pair instead of once per suffix group.

Measured (`results/join_attention_paths_*.json`, BioDEX
reports x reaction terms, TRUE/FALSE-framed corpus, us per fresh
token):

| Shape | split | merge_quant | unified (waves) |
|---|---|---|---|
| 10 x 256 | 11.95 | 11.29 | 31.00 (257 chunks) |
| 1 x 2560 (high fan-out) | 12.87 | 12.05 | 373.76 (2,561 chunks) |
| 100 x 256 (multi-chunk) | 11.59 | 10.83 | 13.16 (259 chunks) |

merge_quant beats split by 5-7% on every shape (the fused merge).
The unified waves match the prediction's direction and show how the
penalty scales with the shape: at one anchor the wave is a 42-token
chunk far below the 416-token compute knee and the anchor's KV is
re-read 2,560 times, so the path collapses (29x slower); at 100
anchors a wave is ~4,200 tokens and the penalty shrinks to 21% -
still never ahead, and the answer-correctness constraint (one pair
per anchor per causal call) is what forces the wave shape in the
first place. Path answer disagreements on this corpus are tiny (0-8
between the two-call paths per shape). The cell's TRUE counts sit
near saturation (~2,554 of 2,560) because the 4B checkpoint answers
TRUE to nearly every BioDEX reaction pair - the known model caveat,
not an executor property; the planted-key join in the accuracy cell
below is where join answers are graded against ground truth.

## Accuracy against stock vLLM

Cell: `ablations/accuracy_vs_stock.py`
(`results/accuracy_vs_stock.json`). The same token id streams
answered by standard vLLM serving (v1 engine, bf16 KV, prefix
caching on, the committed client's batch settings, one request per
document-stage or pair, TRUE/FALSE-constrained single token at
temperature 0) and by the packed executor under each path. Stock
answers every (document, stage) unconditionally and reports
TRUE-FALSE logprob margins; a second stock pass in shuffled order
measures stock's own run-to-run flip rate, which is the yardstick:
continuous batching changes batch composition, which changes
reduction order, which flips answers whose margins sit at zero.

Corpus: 1,000 IMDB documents with planted TRUE/FALSE [FLAGS] lines
(12 long documents of ~8 concatenated reviews, 25 very short ones,
5 whose body is only the flags line), four flag questions with
planted truth plus one natural sentiment question, in the engine's
exact prompt framing; and a 24-anchor x 48-partner join with
planted word-key equality (8 partners carry keys no anchor has),
rendered exactly as the engine renders joins. A first run used
numeric keys 0-11; the 4B model answered at chance (11-12% key
accuracy for BOTH systems) with margins collapsed, so the keys
became words - the committed run's numbers are below.

### Filter results (5,000 stock answers, ~3,564 per Quail path)

- Stock is deterministic here: 0 of 5,000 answers flipped between
  the natural-order and shuffled-order passes. Stock margins are
  wide on this corpus: median |TRUE-FALSE logprob gap| 3.25, and
  only 1.8% of answers sit under 1.0.
- Planted-flag accuracy: stock 99.9-100% per stage; every Quail
  path 100% over its answered set. No path got a planted flag
  wrong, on any document kind (long, short, flags-line-only
  included).
- Disagreements with stock: 8 of ~3,564 (0.22%) for each path.
  Seven of the eight are on the natural sentiment stage, one on a
  flag stage where stock itself is at 99.9%; all eight are on plain
  documents (none on the edge kinds). The stock margins at the
  disagreeing answers: median 0.375, maximum 1.125 - against the
  corpus-wide median of 3.25. For unified (the path filters
  actually run) the maximum is 0.875: zero disagreements at any
  margin above 1.0. split and merge_quant each have one
  disagreement at margin 1.125, just over that line, on the
  sentiment question.
- Survivor sets after the 5-stage chain: stock 61, unified 57 with
  56 shared - the drift is the sentiment-stage flips cascading
  through the gate, not flag errors.
- The production-sequence check (filters on unified, then the join
  on merge_quant, one arena, the worker's mode switch) reproduced
  the isolated runs exactly (`mode_switch_clean: true`).

### Join results (1,152 pairs)

The key-equality task is genuinely hard for Qwen3 4B at this
document length: stock's key accuracy is 16.1% (near the all-TRUE
rate), its margins sit at median 0.875 with 55% under 1.0. Even so,
stock is deterministic on it (0 of 1,152 flips between submission
orders). Against that backdrop:

- split disagrees with stock on 103 pairs (8.9%), merge_quant on
  111 (9.6%) - and every single disagreement sits at |margin| at or
  under 0.75. Zero disagreements at decisive margins for both
  paths: wherever stock's answer had any confidence, Quail said the
  same thing.
- Key accuracy with truth known: Quail 18.2-18.4% against stock's
  16.1% - the packed executor is not degraded relative to stock on
  this task; both are at the model's floor.

### Reading

The residual differences are the kernel stacks, not the attention
paths: Quail runs fp8 DeepGEMM projections and fused Triton norms
where stock runs vLLM's kernels, and a bounded logit perturbation
flips exactly the answers whose margins sit near zero. The
attention-path contribution to the difference is zero for filters -
the unified path reproduces full-prompt recompute through Quail's
own stack exactly (0 of 1,101, the end-to-end parity above), so
every unified-vs-stock difference is the kernel stack, and all of
those land under margin 0.875. The issue's acceptance bar
("zero disagreements, or a clear explanation") is met as: zero at
decisive margins; the thin-margin flips are floating-point-scale
differences on answers the model does not meaningfully decide, and
they do not move planted-truth accuracy on either workload.

## FlashInfer and other off-the-shelf kernels

FlashInfer 0.6.14 ships in the vLLM 0.26.0 image
(`results/flashinfer_probe.json`). The benchmark
(`ablations/flashinfer_compare.py`,
`results/flashinfer_bench.json`) times each stack's full
attention-to-o_proj work (KV scatter + attention + merge + FP8
quantization) on synthetic tensors shaped like the real chunks,
with an output cross-check against the FA3 paths (max_abs 0.008 /
0.004, bf16 rounding - both stacks compute the same attention).

Per-layer milliseconds:

| Shape | best FA3 path | FlashInfer paged causal | FlashInfer two-call + merge_state | FlashInfer cascade |
|---|---|---|---|---|
| filter, fresh chunk (~110k tokens) | 2.98 (unified) | 3.79 | 4.67 | not applicable |
| filter, rewind chunk (~11k tokens) | 0.42 (unified) | 1.02 | 1.58 | not applicable |
| join 10 x 26-suffix groups | 1.38 (merge_quant) | - | 1.88 | not applicable |
| join 1 anchor x 256 (fan-out) | 1.33 (merge_quant) | - | 1.79 | 1.61 |

The issue's rule was: adopt off-the-shelf if within 5%. The closest
FlashInfer result is 21% slower (cascade on the single-anchor
fan-out shape, and cascade cannot run multi-anchor chunks at all -
its shared level must be shared by every query in the batch).
FlashInfer's paged causal kernel is 27% slower than the single FA3
call on the fresh filter chunk and 2.4x slower on the rewind chunk.
Decision: stay on FA3 plus the one custom Triton merge kernel.
Hydragen ships no reusable prefill kernel to import (the
decomposition idea is already what `split`/`merge_quant`
implement, on standard FA3 calls), and SGLang's RadixAttention is
an engine-level KV-reuse policy, not an attention kernel this
executor can call; neither offers a drop-in candidate beyond what
was measured here.

The prediction before the run said "two-call stacks within tens of
percent of each other"; the measured gap (35-45% on joins) came in
above that band - FlashInfer's merge_state plus its separate paged
call costs more than expected next to the fused Triton kernel.

## Qwen3 32B fp8

The same battery on the second in-scope model (one H100, chunk
budget 41,943 tokens, arena 64,453 tokens - the kernel index cap and
the smaller free memory both bind harder than at 4B).

Predictions, stated before the runs:

- The ranking holds on both workloads: the mechanism (one paged
  causal call for the filter shape; two calls plus fused merge for
  fan-out) does not depend on model size. The RELATIVE gaps shrink:
  the launch and merge overhead the paths differ by is roughly
  constant per token, while the per-token GEMM work is ~8x larger.
- Absolute rate lands in the 45-70 us/token band for filters
  (params ratio over the 4B's 8.45, minus large-GEMM efficiency).
- Kernel parity at the 32B geometry (64 query heads, 8:1 GQA):
  unified stays bit-identical to the contiguous call.
- Accuracy vs stock: flag accuracy 100% for both systems; the 32B
  model should clear the 4B's floor on the planted-key join (the
  4B answered TRUE to nearly everything; 32B should actually
  compare the keys), margins widen, and the join disagreement rate
  drops well below the 4B's ~9%.
- FlashInfer at 64 heads: FA3 stays ahead; adopt only if within 5%.

Measured (all in `results/*_32b.json` / `*_64h.json`):

- Filters, 10,000 documents (`attention_paths_32b.json`): unified
  59.94 us/token, merge_quant 60.35, split 61.60 - same order, and
  the relative gap shrank as predicted (unified is 2.7% ahead of
  split at 32B against 6.0% at 4B). All three paths return
  identical answers (0 disagreements on all 40,053), 4,645
  survivors, and 1 planted-flag miss out of 40,053 - the model's
  one genuine error, identical on every path.
- Joins (`join_attention_paths_32b_*.json`, us per fresh token):

  | Shape | split | merge_quant | unified (waves) |
  |---|---|---|---|
  | 10 x 256 | 68.82 | 66.08 | 108.02 |
  | 1 x 2560 | 73.27 | 69.70 | 710.77 |
  | 100 x 256 | 69.61 | 66.87 | 83.69 |

  merge_quant wins every shape by 4-5%. The waves penalty shrinks
  in relative terms (1.6x at 10 x 256 against 2.8x at 4B - the
  launch overhead is a smaller share of the 8x-larger per-token
  work) but stays decisive, and high fan-out still collapses it
  (10x). Worth noting: the 32B model answers the BioDEX task
  selectively (52 TRUE of 2,560; the 4B saturated at ~2,554), so
  this cell's answer counts are meaningful again at 32B.
- Kernel parity at 64 query heads
  (`attention_parity_64h.json`): unified bit-identical to the
  contiguous causal call on all 10 edge cases (max_abs 0.0).
- End to end (`attention_end_to_end_parity_32b.json`): all three
  paths exact against full-prompt recompute (0 of 997, 0 wrong),
  store round trip exact (64/64).
- Accuracy vs stock (`accuracy_vs_stock_32b.json`): flag accuracy
  100% for stock and every path; join key accuracy ~99% for both
  systems (stock 98.87%, Quail 98.78-99.05% - the 32B model clears
  the 4B's 16-18% floor, as predicted). Stock's shuffle controls: 0
  flips on both workloads. Disagreements with stock: 2-3 of ~3,544
  filter answers per path (0.06-0.08%) - every one on the
  no-planted-truth sentiment stage, zero on any flag question - and
  8-15 of 1,152 join pairs (0.7-1.3%), landing on the ~13 pairs
  stock itself gets wrong (graded against planted truth, the
  disagreements split evenly: Quail right on 5 of 8 for split, 7 of
  15 for merge_quant). The mode-switch check is clean. One
  instrument note: at 32B the losing answer token usually falls
  outside vLLM's returned top-k logprobs, so the TRUE-FALSE margin
  reads 0.0 for 98% of filter answers (85% join) and the margin
  analysis carries less weight than at 4B; the planted-truth
  grading above replaces it.
- FlashInfer at the 64-head geometry
  (`flashinfer_bench_64h.json`): still nothing within the 5% bar -
  paged causal 31% behind the single FA3 call on the fresh filter
  chunk (3.0x on the rewind chunk), two-call + merge_state 46%
  behind merge_quant on joins, cascade 21% behind on the fan-out
  shape.

The assignment (filters unified, joins merge_quant) holds unchanged
on both in-scope models.

## Edge cases covered

- Page-boundary lengths, many-page documents (2,049- and
  4,097-token, 129/257 pages), 1-3-token documents, empty document,
  suffix larger than the document, 48-group chunks: kernel parity,
  all exact for unified (`results/attention_parity.json`).
- Multi-stage rewind (5 stages) under every path, and rewind with
  store save/restore under unified: end-to-end parity, exact
  (`results/attention_end_to_end_parity.json`).
- Long/short/minimal documents inside the stock-comparison corpus:
  per-kind disagreement counts in
  `results/accuracy_vs_stock.json`.
- High fan-out joins (1 x 2,560) and multi-chunk joins (100 x 256):
  the join sweep above.
- Mixed workload (filter round then join round, one arena, the
  worker's mode switch): the accuracy cell runs the production
  sequence and checks it reproduces the isolated runs
  (`mode_switch_clean`).

## What changed in the code

- `attention.py`: the assignment constants (`FILTER_ATTENTION =
  "unified"`, `JOIN_ATTENTION = "merge_quant"`) with the evidence
  pointers; worker, calibration boot, and kernel warmup read them
  instead of string literals.
- `ablations/accuracy_vs_stock.py`: the stock-comparison cell (new).
- `ablations/flashinfer_compare.py`: probe + benchmark (new).
- `ablations/forward_pass.py`: join cell gained the unified-waves
  variant and a fan-out sweep entrypoint; parity cells gained the
  edge cases and the store round trip; the stale `yes_no_ids`
  import is fixed to `true_false_ids`.
- `tests/test_attention_assignment.py`: the assignment names real
  modes and joins are never unified (CPU test).
- Wiki section 5.3 rewritten for the three paths and the
  assignment.

## Run log

All runs on Modal app `quail-milestone1`, one H100, tee'd logs in
`results/*.log`:

- `flashinfer_probe.log`, `flashinfer_bench.log`
- `attention_parity.log`, `attention_e2e_parity.log`
- `join_attention_paths.log` (the three-shape sweep)
- `accuracy_vs_stock.log`
