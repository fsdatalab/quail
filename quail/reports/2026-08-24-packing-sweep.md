# Packing sweep: the cost constants on real query packings

Issue #25. The cost model prices every fresh token at
`t = a + a2*h` with two measured constants per (model, device) pair,
and assumes the mix inside a chunk does not matter. Both pairs'
committed constants predated the current executor: the 4B anchor came
from the old exploration (its `a2` from single-document prefills - one
document per forward pass, a shape the engine never runs), and the 32B
pair was measured on 2026-08-20, before the attention-path assignment
and the 08-22 bug-fix rounds.

This sweep re-measures both pairs on the packings real benchmark
queries build, instead of a synthetic grid: run the query, record what
the packer put in every chunk, and time every chunk with CUDA events.
Each chunk is one measured point.

## Setup

Seven QUAIL-B workloads at sf=0.1, run exactly as the benchmark's
cold pass runs them (no store; single-stage filters on the fast path
with arena writes off; joins and the chain write KV), on one H100 per
model. The first four ran as one pass; the last three were added
after that pass confirmed the join gap, to cover the corpora and
shapes it left out:

| Query | Shape | What its chunks look like |
|---|---|---|
| BIO-1 | filter over 200 real BioDEX reports | ~26 documents of 833-14,803 tokens mixed in one chunk |
| IMDB-1 | filter over 5,000 real IMDB reviews | ~300 short documents per chunk |
| BIO-2 | join, 200 reports x 614 terms | 1-2 long anchor prefixes + ~1,200 suffixes of ~85 tokens |
| IMDB-2 | join, 5,000 reviews x 12 aspects | ~90 short anchors + ~1,050 suffixes per chunk |
| LEP-2 | self-join, 200 LePaRD citations x 200 passages | ~200-token anchors + ~165-token suffixes (third corpus: legal text) |
| FEV-2 | join, 100 FEVER claims x 57 evidence pages | the mirror of BIO-2: ~11-token claim anchors + ~385-token Wikipedia-passage suffixes |
| BIO-F3 | 3-stage filter chain (F7 -> F8 -> F9) over the 200 reports | stage-1 documents mixed with ~60-token later-stage tails read against each document's kept KV (KV rewind) |

Between them these workloads produce, on their own, the variation
issue #25 asked to sweep: mixed document lengths inside one chunk,
many small pieces per chunk, partly-empty chunks at stream tails,
both join orientations (long anchors with short suffixes and the
reverse), four corpora, and a multi-stage chain. Chunk fill ran
13-100%; pieces per chunk ran 8 to ~1,250.

The cell is `ablations/packing_sweep.py` (runs the executor's
`run_filter` / `run_join` directly, with the new per-chunk `trace`).
The fit is `ablations/packing_sweep_fit.py`. The committed summary the
numbers and plots below read is `results/packing_sweep.json`. Raw
per-chunk records live on the `quail-results` volume:

- `/results/ablations/packing_sweep_qwen3-4b-fp8_full.json` (+ `_ext`, `_rep0..2`)
- `/results/ablations/packing_sweep_qwen3-32b-fp8_full.json` (+ `_ext`, `_rep0..2`)

Function call ids are in `results/packing_sweep_4b.log`,
`packing_sweep_32b.log`, `packing_reps_4b.log`, `packing_reps_32b.log`,
`packing_ext_4b.log`, and `packing_ext_32b.log`.

The fit regresses, over chunks:

    gpu_seconds = a * T + a2c * Sc + a2x * Sx

where `T` is the chunk's fresh tokens, `Sc` sums `n * L` over causal
segments (a token attends ~half its own segment; the /2 is absorbed in
`a2c`, matching how the length-sweep `a2` was always defined), and
`Sx` sums `fresh tokens * kept context` over the pieces that read KV
kept from before - join suffixes reading their anchor, and a chain's
later-stage question tails reading their document. Chunks with no
such reads (`Sx = 0`) pin `(a, a2c)` - the two constants the
calibration files carry. The rest test the cross term.

## Predictions, stated before the run

From the constants loaded before this sweep (per-query us per fresh
token, GPU):

| Query | 4B predicted | 32B predicted |
|---|---|---|
| BIO-1 | 11.04 | 65.23 |
| IMDB-1 | 8.48 | 57.32 |
| BIO-2 | 10.40 | 63.28 |
| IMDB-2 | 8.46 | 57.26 |

Stated expectations: the 4B filter intercept lands at 8.6-9.4
us/token (the 107k-121k tok/s band of earlier runs); join chunks land
above the filter fit, and if the gap is on the GPU it shows up as
roughly 100-200 us per suffix; container-to-container variation needs
re-measuring because the exploration saw up to 45%.

The three added workloads were predicted from the RECALIBRATED
constants plus the join terms the first pass measured (`a2x` at
1.21-1.23x the causal `2*a2c`, ~37 us / ~126 us per suffix), stated
before their run:

| Query | 4B predicted | 32B predicted |
|---|---|---|
| LEP-2 | 8.4 | 58.1 |
| FEV-2 | 8.2 | 57.2 |
| BIO-F3 | 10.4-10.6 | 66-67 |

The two sharp tests: FEV-2's big suffixes against tiny anchors should
land near the causal filter rate (a big suffix prices like a
document), and BIO-F3's later-stage tails should price at the same
cross coefficient `a2x` the joins needed - one term explaining both
shapes.

## Results

Measured, GPU us per fresh token (wall differed from GPU by under
0.3% everywhere - nothing here is host-bound):

| Query | 4B measured | vs predicted | 32B measured | vs predicted |
|---|---|---|---|---|
| BIO-1 | 10.33 | -6.6% | 65.63 | +0.5% |
| IMDB-1 | 8.09 | -4.9% | 57.03 | -0.7% |
| BIO-2 | 12.23 | +17.6% | 71.87 | +13.6% |
| IMDB-2 | 8.55 | +1.0% | 58.55 | +2.2% |

Figure: plots/packing_sweep_queries.png

Fits over the per-chunk points:

| | 4B | 32B |
|---|---|---|
| `a` (us/token) | 7.867 (was 8.261, -4.8%) | 56.22 (was 56.64, -0.7%) |
| `a2c` (s/token^2) | 4.343e-10 (was 4.934e-10, -12%) | 1.563e-09 (was 1.528e-09, +2.3%) |
| filter-fit R^2 (chunks used) | 0.9995 (24 of 25) | 0.9992 (63 of 64) |
| `a2x` (s/token^2, cross) | 1.070e-09 = 1.23x the causal 2*a2c | 3.793e-09 = 1.21x |
| leftover per suffix (joint fit) | ~37 us | ~126 us |
| container band (filters, 4 containers) | 2.9-3.1% | 2.6-6.6% |

One chunk per model was excluded from the filter fit: each
container's first measured chunk paid a leftover kernel compile once
(+14% on that one chunk at 32B, cached afterward). The fit script
excludes any chunk more than 5% off its own GPU time and reports the
count; the all-chunk fit is kept alongside in the summary
(`fits.filter.all_chunks`).

Figure: plots/packing_sweep_context.png

What the figure shows: with the x axis counting cross context in full
and a segment's own context at half, equal per-attended-token cost
would put join chunks on the same line as filter chunks. They sit
above it.

## What the numbers mean

- **The two-constant model holds on filter packings.** One line fits
  all 25 filter chunks of both corpora at R^2 = 0.9998, across mixed
  lengths (833 to 14,803 tokens in the same chunk), 8 to 300 pieces
  per chunk, and fill from 66% to 100%. Adding a per-chunk fixed cost
  changes nothing (7 ms per ~1 s chunk). Packing density does not
  need a term.
- **Both committed constants were stale, in opposite ways.** The 4B
  executor now runs 4.8% faster than its anchor (127.1k tok/s against
  the anchor's 121.0k; the anchor also predated the 08-21/08-22
  changes, which beat the 8.6-9.4 us/token band this report
  predicted). The 32B filter constants were nearly right (within 1%).
  Both files are refreshed from this sweep's filter fit.
- **Join chunks cost more than the causal model says, on the GPU.**
  Reading a kept anchor's KV from arena pages costs 1.23x per
  attended token compared with in-chunk causal attention (`a2x =
  1.070e-09` against `2*a2c = 8.685e-10` at 4B), plus ~37 us per
  suffix. On BIO-2's shape that is +18% wall; on IMDB-2's short
  anchors it is +1%. This is the term the model was missing - not
  chunk fill, not segment count.
- **Container-to-container variation collapsed.** The exploration
  measured up to 45%. Four containers here: 2.9-3.1% at 4B; at 32B,
  2.6% on the short-document filter and 6.6% on the long-document one
  (the full run's container was the fastest of its four). Committed
  constants are fine - the worst band is 6.6% against decision
  margins of 32% and larger - so no boot-time calibration and no
  worst-case constants are needed.
- **No break-even flips.** Restore-vs-recompute at 4B: loading KV
  costs 5.33 us/token at the pinned 27.7 GB/s channel against 7.87
  us/token to recompute, so restore still wins at every length
  (margin was 55% under the old `a`, now 32%). At 32B the margin
  stays ~6x. The planner's token-count join ordering compares
  same-shape alternatives, so the join surcharge cancels there;
  `choose_anchor` for a two-way join is unaffected because both
  candidates price the same tuple count. The surcharge matters only
  if a future decision needs join wall accuracy - then price suffix
  context at `a2x` and add the per-suffix constant.

## Changes shipped with this report

- `quail/calibration/qwen3-4b-fp8_h100-sxm.json` and
  `qwen3-32b-fp8_h100-sxm.json` refreshed from the filter fits, with
  provenance pointing at this sweep.
- `run_join` gained the same per-chunk `trace` option `run_filter`
  already had; both traces now record per-piece composition
  (see `reports/shipped_features/2026-08-24-join-chunk-trace.md`).

## Reproducing

    uv run modal run ablations/packing_sweep.py --model qwen3-4b-fp8 \
        2>&1 | tee results/packing_sweep_4b.log
    # pull the raw records, then:
    uv run python ablations/packing_sweep_fit.py \
        packing_sweep_*.json --out results/packing_sweep.json
    uv run --with matplotlib python reports/make_packing_sweep_plots.py
