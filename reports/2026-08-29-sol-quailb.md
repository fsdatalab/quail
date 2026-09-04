# Speed of light for the 30 current QuailB queries

## Setup

- The calculation covers 30 of the 32 QuailB queries at scale factor 0.1.
  The two agent trace queries, AGENT-1 and AGENT-2, are recorded as
  skipped: the script does not tokenize the agent trace corpus.
- It runs separately for Qwen3 4B fp8 and Qwen3 32B fp8.
- Each estimate is for one H100! request and one model copy.
- The corpus is `c_1aa2c4f0d0b6c816fd37aa5748c33341`.
- Ground truth collection `gt_77bb8b128743a79aedddaa24c808c3f8`, the
  active selectivity estimate collection, supplies exact survivors for
  every predicate. The file was regenerated on 2026-09-04 against this
  collection; every per query estimate matched the 2026-08-29 file built
  on `gt_363b5ab570635c33894e1a030c21f57e`, so the numbers below are
  unchanged.
- The full result is at `/results/sol/sol_quailb_sf0.1.json` on the
  `quail-results` Modal volume.

The estimate counts modeled fresh tokens, attention pairs, KV reads, and KV
writes. It does not include kernel gaps, host work, or scheduling overhead. No
measured or fitted constant is used.

Filters use fixed selectivity estimates to choose their order. The calculation
then uses saved ground truth to find the exact rows that reach each later
stage. Join queries search every supported eager left deep relation order and
anchor choice. The search supports binary full joins, applies each available
crossing predicate immediately, and does not consider bushy plans. Document
prefix KV has unlimited capacity in this estimate. Work is ideally packed
across the whole query, even across operator barriers. This is separate from
the production planner, which must work without ground truth and with finite
KV.

These assumptions make SoL an optimistic comparison point for the modeled
execution. It is not the exact minimum time for every possible query plan.

Because the survivors come from the ground truth, the SoL models the work of a
query whose model answers are all correct. A measured run does the work the
model's actual answers create. Where the model passes many more documents than
the ground truth does, as on the LePaRD filters, the measured work is a
multiple of the modeled work and the measured time is a multiple of the SoL
that no scheduler can close. `reports/2026-08-31-quailb-kv-regret.md` shows
that effect per query.

## Prediction

The 30-query result was expected to match the old 30-query subset because the
new ground truth collection reuses the same label sets.

## Result

That prediction was wrong. The old SoL file used 200 BioDEX reports and 100
FEVER claims. The active scale factor 0.1 corpus has 500 of each. The measured
engine runs also used 500 of each, so the regenerated estimate is the correct
comparison.

- The 4B estimates total 370.34 seconds. The old file totaled 209.14 seconds.
- The 32B estimates total 2,890.58 seconds.
- IMDB totals 109.23 seconds for 4B.
- BioDEX totals 116.17 seconds for 4B.
- FEVER totals 80.50 seconds for 4B.
- LePaRD totals 64.44 seconds for 4B.
- The five retired BioDEX queries are no longer included.

Figure: plots/sol_quailb_per_query.png

The log scale is used because the query estimates span more than one order of
magnitude.

Figure: plots/sol_quailb_attention_share.png

The second figure shows how attention's share of the required compute changes
with the document context length. Queries with several joins are omitted from
this figure because they can use more than one anchor context.

## Rebuild

Run `reports/make_sol_quailb.py`, then
`reports/make_sol_quailb_plots.py`, from the repository root. Their docstrings
contain the `modal volume get` and `modal volume put` commands.
