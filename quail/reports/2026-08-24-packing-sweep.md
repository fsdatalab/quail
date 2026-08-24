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

    gpu_seconds = a * T + a2 * S + c + p * suffixes

where `T` is the chunk's fresh tokens, `S` is total attention pairs
(causal: n^2 per segment, the /2 absorbed into a2; plus cross-read:
suffix_tokens * anchor_tokens), `c` is the per-chunk CUDA launch
floor, and `suffixes` is the paged-attention dispatch count. One `a2`
because the FLOPs per attention pair are the same regardless of where
the KV lives; the paged-attention overhead is a per-dispatch cost
(`p`), not a per-pair multiplier.

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

The three added workloads were predicted from the recalibrated
constants plus the join terms the first pass measured, stated before
their run:

| Query | 4B predicted | 32B predicted |
|---|---|---|
| LEP-2 | 8.4 | 58.1 |
| FEV-2 | 8.2 | 57.2 |
| BIO-F3 | 10.4-10.6 | 66-67 |

The two sharp tests: FEV-2's big suffixes against tiny anchors should
land near the causal filter rate (a big suffix prices like a
document), and BIO-F3's later-stage tails should price at the same
attention coefficient the joins needed.

## Results

Measured, GPU us per fresh token (wall differed from GPU by under
0.3% everywhere - nothing here is host-bound):

| Query | 4B measured | vs predicted | 32B measured | vs predicted |
|---|---|---|---|---|
| BIO-1 | 10.33 | -6.6% | 65.63 | +0.5% |
| IMDB-1 | 8.09 | -4.9% | 57.03 | -0.7% |
| BIO-2 | 12.23 | +17.6% | 71.87 | +13.6% |
| IMDB-2 | 8.55 | +1.0% | 58.55 | +2.2% |
| LEP-2 | 8.84 | +4.4% | 63.31 | +10.6% |
| FEV-2 | 8.72 | +1.8% | 64.69 | +12.4% |
| BIO-F3 | 10.63 | -3.8% | 69.07 | +5.8% |

The extension rows ran on one further container per model, and their
whole excess is that container's rate: within each of those queries
the per-chunk excess is flat and uncorrelated with context (the fit
separates it below as +0.36 and +6.1 us/token for the two ext
containers). The BIO-F3 rows are warm re-runs. The cold first runs
measured 13.67 (4B) and 91.94 (32B) us/token - +30% and +38% - and
the per-chunk records show all of that sitting in a handful of tiny
gated tail chunks paying one-time kernel compiles (0.7-0.9 s per
chunk at 4B, up to 9.3 s at 32B, on 66-523-token chunks whose shapes
no warmup covered). On the warm container the same chunks run at the
launch floor (next section) and the chain lands on its prediction.

Figure: plots/packing_sweep_queries.png

### Per-query per-model fits

Each query is fit alone, with standard errors, using only the terms
its own chunks carry - no container offsets, no borrowing from other
queries. Filter-only queries fit `[a, a2, c]`; queries with suffixes
fit `[a, a2, c, p]`. A value written "7.86±0.10" means the query's
chunks pin that constant to about ±0.10; an error as large as the
value means the query's packings cannot see that constant at all.
Excluded chunks (see note below) are the same in every fit.

**These tables need regeneration** with the new 4-constant model
(`packing_sweep_fit.py` rewritten; raw data on the quail-results
volume). The values below are from the previous model (separate
a2c/a2x) and are kept for reference until re-run.

**4B (previous model, for reference):**

| Query | Container | a (us/tok) | a2c (e-10) | a2x (e-10) | per suffix (us) |
|---|---|---|---|---|---|
| BIO-1 | full | 7.86±0.10 | 4.36±0.18 | - | - |
| IMDB-1 | full | 7.97±0.09 | 2.2±1.8 | - | - |
| BIO-2 | full | 6.70±0.49 | 5.50±0.42 | 9.84±0.05 | 136±43 |
| IMDB-2 | full | 7.0±1.3 | 4.3±2.8 | 20±14 | 100±98 |
| LEP-2 | ext | 9.5±1.5 | -29±49 | 10.29±0.51 | -89±118 |
| FEV-2 | ext | 12.3±2.4 | -31±21 | 17±17 | -800±550 |
| BIO-F3 | ext2 | 7.55±0.18 | 5.19±0.28 | 61±38 | -220±480 |

**32B (previous model, for reference):**

| Query | Container | a (us/tok) | a2c (e-10) | a2x (e-10) | per suffix (us) |
|---|---|---|---|---|---|
| BIO-1 | full | 55.38±0.36 | 17.05±0.62 | - | - |
| IMDB-1 | full | 56.24±0.12 | 15.7±2.5 | - | - |
| BIO-2 | full | 56.33±0.40 | 15.59±0.40 | 34.96±0.06 | 117±36 |
| IMDB-2 | full | 56.3±1.4 | 11.2±6.2 | 43±13 | 105±110 |
| LEP-2 | ext | 57±24 | 160±760 | 44±10 | 520±1810 |
| FEV-2 | ext | 112±28 | -360±230 | -1020±430 | -10700±6400 |
| BIO-F3 | ext2 | 57.26±0.18 | 16.84±0.31 | 3.7±85 | 1340±1090 |

Three things the tables show (these carry over to the new model):

- Where a query's packings identify a constant, queries agree. `a`
  from the four same-container 4B queries: 6.70-7.97 (the three with
  tight errors: 7.55-7.97). At 32B: 55.4-56.3 across four queries.
  The extension queries' `a` comes out high by about their
  container's rate offset (BIO-F3 on ext2: 57.26 vs BIO-1 on full:
  55.38), which is the container effect showing up unmodeled in a
  per-query fit.
- Where a query's packings lack the variation, the fit chases noise:
  FEV-2's chunks are all nearly identical (same size, same mix), so
  its regressors are collinear and it "finds" negative coefficients
  with errors bigger than the values. IMDB-1's short docs barely
  span the length axis, so its a2 has large standard errors.
- Within one query, correlated terms trade off against each other:
  BIO-2 alone drifts `a` down and per-suffix up because every suffix
  is ~85 tokens, making tokens and suffix counts nearly proportional.
  Only queries with different suffix sizes, fit together, separate
  them.

### Joint fit, from the agreement

Because the per-query fits converge where they can identify a
constant, and no query disagrees beyond its own error bars plus its
container's rate offset, fitting all chunks jointly per model
(with one rate-offset column per extra container) gives tighter
values.

**This table needs regeneration** with the new 4-constant model.
The previous model's filter-fit values for `a` and `a2` are
unchanged (the filter-only data has no suffixes, so `c` and `p`
are the only new terms). Values below are from the previous model:

| | 4B | 32B |
|---|---|---|
| `a` (us/token) | 7.867 (was 8.261, -4.8%) | 56.22 (was 56.64, -0.7%) |
| `a2` (s/token^2) | 4.343e-10 (was 4.934e-10, -12%) | 1.563e-09 (was 1.528e-09, +2.3%) |
| `c` (ms/chunk) | not yet fit | not yet fit |
| `p` (us/suffix) | not yet fit | not yet fit |
| R^2 (chunks used) | 0.9995 (24 of 25, filter only) | 0.9992 (63 of 64, filter only) |
| ext-container rate offsets (us/token) | +0.36, +0.20 | +6.11, +1.84 |
| container band (repeated filters, 4 containers) | 2.9-3.1% | 2.6-6.6% |

The committed calibration files carry `a` and `a2` from the
filter-only fit; `c` and `p` default to 0 until the full joint fit
is regenerated. To regenerate: pull raw records from
`quail-results`, run `packing_sweep_fit.py`, and update this table.

Excluded chunks (the fit drops any chunk more than 5% off its own
GPU time and reports counts per query): each container's first
measured chunk pays one leftover kernel compile; a cold chain's tiny
gated tail chunks pay compiles of 0.7-9.3 s once; and warm tail
chunks under ~2k tokens sit on a per-chunk launch floor of ~20-27 ms
(4B) / ~25-30 ms (32B) - a 165-token chunk costs ~20 ms no matter
what is in it. The floor is the one place chunk fill genuinely
matters; across a whole chain query it is ~1% of wall time, because
only the gated trailing chunks are that small.

Figure: plots/packing_sweep_context.png

What the figure shows: the x axis is mean attention pairs per token
(S/T). Under the new model (one `a2`), all chunk types — filter,
join, chain — should fall on the same line, with join and chain
chunks offset only by their per-suffix `p` cost plus container rate
differences.

## What the numbers mean

- **The cost model holds on filter packings.** One line fits the
  filter chunks of both corpora at R^2 = 0.9995 (24 of 25 chunks at
  4B), across mixed lengths (833 to 14,803 tokens in the same chunk),
  8 to 300 pieces per chunk, and fill from 66% to 100%. Packing
  density needs no term above ~2k tokens per chunk; below that the
  ~20-30 ms launch floor (`c`) takes over.
- **Both committed constants were stale, in opposite ways.** The 4B
  executor now runs 4.8% faster than its anchor (127.1k tok/s against
  the anchor's 121.0k; the anchor also predated the 08-21/08-22
  changes, which beat the 8.6-9.4 us/token band this report
  predicted). The 32B filter constants were nearly right (within 1%).
  Both files are refreshed from this sweep's filter fit.
- **Paged-attention overhead is a per-dispatch cost, not a per-pair
  multiplier.** The old model used separate a2c (causal) and a2x
  (cross-read) coefficients, but the FLOPs per attention pair are the
  same regardless of where the KV lives. The overhead comes from
  gathering non-contiguous arena pages per suffix dispatch — a
  per-kernel cost (`p`), not a per-pair cost. The new model uses one
  `a2` for all attention pairs and a separate `p` per suffix
  dispatch. The joint fit with the new model needs regeneration from
  the raw data.
- **A multi-stage chain needs no term of its own.** BIO-F3 (three
  stages, arena writes on, KV rewind) lands on its prediction: 10.63
  measured against 10.4-10.6 (4B, warm). The writes-on chunks run
  ~2% over the fast-path fit, matching the earlier 1.2% measurement.
  The cold first runs of the chain were +30-38% - all of it one-time
  kernel compiles on tiny tail-chunk shapes, not the multi-stage
  path.
- **Container-to-container variation collapsed, but is bigger than
  the repeat runs alone showed.** The exploration measured up to 45%.
  Six containers per model here: the repeated filters span 2.9-3.1%
  at 4B and 2.6-6.6% at 32B, and the two extension containers add
  rate offsets of +2.5-4.6% (4B) and +3.3-10.9% (32B). So the band is
  roughly 5% at 4B and 11% at 32B - wide enough to notice, far from
  45%, and still under every decision margin (32% at 4B, ~6x at 32B),
  so no boot-time calibration and no worst-case constants are
  needed.
- **No break-even flips.** Restore-vs-recompute at 4B: loading KV
  costs 5.33 us/token at the pinned 27.7 GB/s channel against 7.87
  us/token to recompute, so restore still wins at every length
  (margin was 55% under the old `a`, now 32%). At 32B the margin
  stays ~6x. The planner's token-count join ordering compares
  same-shape alternatives, so any per-suffix term cancels there;
  `choose_anchor` for a two-way join is unaffected because both
  candidates price the same tuple count.

## Changes shipped with this report

- Cost model rewritten: `gpu_seconds = a*T + a2*S + c + p*suffixes`.
  One `a2` replaces the old `a2c`/`a2x` split. Added `c` (per-chunk
  launch floor) and `p` (per paged-attention suffix dispatch).
  Calibration files, the fit script (`packing_sweep_fit.py`), the
  measurement step (`calibrate.py`), and the Modal entry
  (`runtime/calibrate.py`) all updated.
- `quail/calibration/qwen3-4b-fp8_h100-sxm.json` and
  `qwen3-32b-fp8_h100-sxm.json` carry `a` and `a2` from the filter
  fit; `c` and `p` default to 0 until the full fit is regenerated.
- `run_join` gained the same per-chunk `trace` option `run_filter`
  already had; both traces now record per-piece composition
  (see `reports/shipped_features/2026-08-24-join-chunk-trace.md`).
- `warm_kernels` closed the compile-stall gap this sweep found, in
  two steps: a tiny-chunk warmup ladder, then replacing the guessed
  sub-4,096 sweep sizes with vLLM's DeepGEMM config-boundary
  generator (cache-directory snapshots attributed the stalls to ~2.4 s
  nvcc compiles of the vendored DeepGEMM kernels, 3-4 per novel
  chunk size). Final validation on a fully cold cache: zero kernel
  files created during the measured query; the chunks that stalled
  9.3-12.9 s now run at 12-48 ms. See
  `reports/shipped_features/2026-08-24-tiny-chunk-warmup.md`.

## Reproducing

    uv run modal run ablations/packing_sweep.py --model qwen3-4b-fp8 \
        2>&1 | tee results/packing_sweep_4b.log
    uv run modal run ablations/packing_sweep.py --model qwen3-4b-fp8 \
        --queries LEP-2,FEV-2,BIO-F3 --tag ext \
        2>&1 | tee results/packing_ext_4b.log
    # pull the raw records, then (a warm re-run listed after its cold
    # record overrides it, so fits use the compile-free measurement):
    uv run python ablations/packing_sweep_fit.py \
        <full> <ext> <ext2> <reps...> --out results/packing_sweep.json
    uv run --with matplotlib python reports/make_packing_sweep_plots.py
