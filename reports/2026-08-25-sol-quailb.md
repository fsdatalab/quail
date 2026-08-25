# Speed of light for all 26 QUAIL-B queries, 4B and 32B

## What this is

A floor on the wall time of every QUAIL-B query at sf=0.1, on one
H100, for Qwen3-4B-fp8 and Qwen3-32B-fp8. The method is
`plans/sol_model.md`; the code is `quail/sol.py`. The floor counts
the dense projection FLOPs, the attention pair FLOPs, the weight
reads and the KV traffic, and counts nothing else, so a run can
approach it and never beat it.

The bound borrows nothing from the engine's own cost model: no
roofline out of `planner/budgets.py`, no calibration constant, no
efficiency factor. `quail/sol.py` imports only `quail.specs`. That
is what lets it judge the engine rather than restate it.

Numbers: `results/sol_quailb_sf0.1.json`. Raw label sets:
`/results/ground_truth/quailb/schema_v1/label_sets` on
`quail-results`. Prebuilt corpora: `/quailb_data/sf0.1` on the same
volume. Measured walls: `/results/sol_check_sf0.1_4b.json`.

## Prediction, before the numbers

Two things were expected and one was not.

Expected: the engine would sit at a similar fraction of the floor
across queries, because the losses it carries (kernel efficiency,
launch gaps, scheduling) are per-token, not per-query. Expected:
32B would cost about 8.6x 4B, the ratio of the two parameter counts.

Not expected: that the 32B ratio would fall to 6.6 on BioDEX. The
attention term does not scale with parameters, and BioDEX documents
are long enough that it dominates.

## Setup

- Inputs are measured, not assumed. Every corpus was tokenized with
  the Qwen3 tokenizer (4B and 32B share it). Survivor counts and
  their token masses come from the QUAIL-B ground truth label sets -
  the 32B model's per-document TRUE/FALSE - counted per document,
  never scaled by a selectivity.
- `chunk_tokens` is 110,376 at 4B and 41,943 at 32B: `(2^31 - 1)`
  over the widest projection, the fused kernels' 32-bit offset
  limit. It is an input to the bound, stated here, not something the
  bound derives.
- Joins price both orientations and keep the cheaper, as the planner
  does. Anchoring a side means holding its KV and streaming the
  other side through it.

## Result

Figure: plots/sol_quailb_per_query.png

| query | shape | tokens | pairs | tuples | 4B SoL | 32B SoL | 32B/4B |
|---|---|---|---|---|---|---|---|
| IMDB-1 | 1F | 1,774,233 | 4.45e8 | - | 6.78 s | 56.90 s | 8.39 |
| IMDB-2 | 1J | 6,124,233 | 1.96e9 | 60,000 | 23.66 s | 197.31 s | 8.34 |
| IMDB-3 | 1F+1J | 5,352,885 | 1.73e9 | 46,476 | 20.69 s | 172.48 s | 8.34 |
| IMDB-4 | 2F+1J | 2,849,949 | 8.97e8 | 11,556 | 11.00 s | 91.78 s | 8.34 |
| IMDB-5 | 3F+1J | 2,590,485 | 8.07e8 | 7,536 | 9.99 s | 83.41 s | 8.35 |
| IMDB-6 | 2F | 1,960,137 | 5.07e8 | - | 7.50 s | 62.89 s | 8.39 |
| IMDB-7 | 3F | 2,010,213 | 5.28e8 | - | 7.70 s | 64.52 s | 8.38 |
| BIO-1 | 1F | 839,599 | 2.38e9 | - | 4.50 s | 31.53 s | 7.00 |
| BIO-2 | 1J | 11,347,799 | 4.65e10 | 122,800 | 69.40 s | 456.48 s | 6.58 |
| BIO-3 | 1F+1J | 7,097,928 | 2.93e10 | 73,066 | 43.52 s | 285.90 s | 6.57 |
| BIO-4 | 2F+1J | 5,157,297 | 2.11e10 | 50,348 | 31.50 s | 207.30 s | 6.58 |
| BIO-5 | 3F+1J | 3,794,195 | 1.46e10 | 34,384 | 22.63 s | 150.59 s | 6.65 |
| FEV-1 | 1F | 6,942 | 2.45e5 | - | 0.026 s | 0.22 s | 8.56 |
| FEV-2 | 1J | 485,820 | 2.02e8 | 5,700 | 1.90 s | 15.75 s | 8.27 |
| FEV-3 | 1F+1J | 302,382 | 1.22e8 | 3,363 | 1.18 s | 9.80 s | 8.28 |
| FEV-4 | 2F+1J | 80,862 | 2.73e7 | 570 | 0.31 s | 2.61 s | 8.33 |
| FEV-5 | 2F+1J two-sided | 209,867 | 8.36e7 | 2,183 | 0.82 s | 6.80 s | 8.28 |
| FEV-6 | 3F+1J two-sided | 67,067 | 2.15e7 | 370 | 0.26 s | 2.16 s | 8.34 |
| LEP-1 | 1F | 58,419 | 1.16e7 | - | 0.22 s | 1.87 s | 8.43 |
| LEP-2 | 1J | 6,086,619 | 1.97e9 | 40,000 | 23.53 s | 196.13 s | 8.34 |
| LEP-3 | 1F+1J | 179,211 | 7.23e7 | 800 | 0.70 s | 5.81 s | 8.28 |
| LEP-4 | 2F+1J | 149,213 | 5.65e7 | 600 | 0.58 s | 4.83 s | 8.30 |
| LEP-5 | 3F+1J | 58,769 | 1.18e7 | 0 | 0.22 s | 1.88 s | 8.43 |
| LEP-6 | 5F+1J | 58,769 | 1.18e7 | 0 | 0.22 s | 1.88 s | 8.43 |
| LEP-7 | 2F+1J two-sided | 175,402 | 5.84e7 | 600 | 0.68 s | 5.66 s | 8.33 |
| LEP-8 | 5F | 58,769 | 1.18e7 | - | 0.22 s | 1.88 s | 8.43 |

Every query on both models is compute bound. Memory never comes
close: the largest `T_memory` in the suite is FEV-1's, at 6.4% of
its `T_compute`, and FEV-1 is the smallest query here - 100 claims
of 11 tokens. Everything else pushes enough tokens per pass that the
weight reads amortize away.

## Is the arithmetic right

Five queries have a measured wall in `sol_check_sf0.1_4b.json`, and
three of them push a token count no model's answers can change -
one filter over a whole corpus, or a join with no filter in front
of it. Those three must match exactly, and do:

| query | engine tokens | bound tokens | ratio |
|---|---|---|---|
| IMDB-1 | 1,774,233 | 1,774,233 | 1.0000 |
| IMDB-2 | 6,124,233 | 6,124,233 | 1.0000 |
| BIO-2 | 11,347,799 | 11,347,799 | 1.0000 |
| IMDB-5 | 2,767,445 | 2,590,485 | 0.9361 |
| FEV-5 | 206,879 | 209,867 | 1.0144 |

IMDB-2 and BIO-2 are joins, on two corpora with opposite shapes -
5,000 short reviews against 12 tiny aspects, 200 long reports
against 614 tiny terms - so matching both to the digit is a real
check of the join arithmetic, not a coincidence. IMDB-5 and FEV-5
depend on which documents survive their filters, and that run was
4B answering while these numbers are the 32B ground truth, so they
should differ by about the amount they do.

FEV-5 only matched after the bound learned to choose an anchor.
Anchoring FEVER on claims, as written, gives 972,497 tokens;
anchoring on evidence gives 178,007, because claims average 11
tokens and evidence 370, and the streamed side is copied once per
tuple. The planner picks the cheaper side and so must the bound.
Getting this wrong is a 4.9x error, the largest single mistake
found while building this.

## What the gap to the wall is

Figure: plots/sol_quailb_measured.png

| query | floor | measured wall | share |
|---|---|---|---|
| IMDB-1 | 6.78 s | 15.06 s | 45% |
| IMDB-2 | 23.66 s | 53.18 s | 44% |
| IMDB-5 | 9.99 s | 23.94 s | 42% |
| BIO-2 | 69.40 s | 140.31 s | 49% |
| FEV-5 | 0.82 s | 1.84 s | 45% |

The engine runs at 0.42 to 0.49 of the floor. That band holds
across three corpora, across filters and joins, and across walls
from 1.8 seconds to 140 seconds. This was the prediction and it
held. The steadiness is the useful part: the loss is per-token, not
per-query, so it is one thing to chase rather than twenty-six.

## Why 32B is not a flat 8.6x

Figure: plots/sol_quailb_attention_share.png

The dense term scales with parameters, 31,206,298,624 over
3,633,511,936, which is 8.59x. The attention term scales with
`4 n_q d_head L`, which is 2,097,152 at 32B against 589,824 at 4B,
or 3.56x. Both attention terms price against the same bf16 peak.

So a query's 32B multiplier sits between 3.56 and 8.59 according to
how much of its compute is attention. IMDB, FEVER and LePaRD are
0.6-6.3% attention and land at 8.3-8.6. BioDEX is 32-40% attention
and lands at 6.6-7.0. The reason is document length: BioDEX reports
average 4,146 tokens against IMDB's 299, and the pair count is
quadratic in length.

This is worth stating plainly because it inverts the usual
intuition. Long documents make the big model relatively cheaper,
not more expensive.

## Things this does not settle

- **The floor is loose.** `max(T_compute, T_memory)` taken once at
  the top is weaker than taking it per kernel and summing. Both are
  lower bounds; the per-kernel one would be larger and tighter.
- **`sol_check_sf0.1_4b.json` carries its own `sol_s`**, 10.195 s
  for IMDB-1 against 6.78 s here. That calculation is not in this
  repository and I could not find what produced it. Until it is
  reconciled, only one of the two numbers should appear in a paper.
- **Only five queries have a measured wall.** The 0.42-0.49 band
  rests on those five. Nothing here is measured on 32B at all: the
  32B column is arithmetic, unchecked against a run.
- **LEP-5, LEP-6 and LEP-8 are the same number** because LEP1 leaves
  4 documents of 200 and LEP3 leaves none, so the later stages and
  the join cost nothing. That is the data at sf=0.1, not a bug, but
  those three queries carry no signal.
