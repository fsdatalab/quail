# Speed of light for the 30 current QuailB queries

## Setup

- The calculation covers the 30 default QuailB queries at scale factor 0.1.
- It runs separately for Qwen3 4B fp8 and Qwen3 32B fp8.
- Each estimate is for one H100! request and one model copy.
- The corpus is `c_3bd14ed0758287cba9d88fb68de8b7b8`.
- Ground truth collection `gt_363b5ab570635c33894e1a030c21f57e`
  supplies exact survivors for all 22 predicates.
- The full result is at `/results/sol/sol_quailb_sf0.1.json` on the
  `quail-results` Modal volume.

The estimate counts the model work that the query requires. It includes fresh
tokens, attention pairs, KV reads, and KV writes. It does not include kernel
gaps, host work, or scheduling overhead. No measured or fitted constant is
used.

Filters use fixed selectivity estimates to choose their order. The calculation
then uses saved ground truth to find the exact rows that reach each later
stage. Join queries search every feasible left deep relation order and anchor
choice. Document prefix KV has unlimited capacity in this estimate. This is
separate from the production planner, which must work without ground truth and
with finite KV.

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
