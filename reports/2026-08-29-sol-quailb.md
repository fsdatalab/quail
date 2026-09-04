# Speed of light for the 32 QuailB queries

## Setup

- The calculation covers all 32 QuailB queries at scale factor 0.1. The
  two agent trace queries were added on 2026-09-04; the 30 others were
  regenerated the same day against the active collection and matched
  the 2026-08-29 file exactly.
- It runs separately for Qwen3 4B fp8 and Qwen3 32B fp8.
- Each estimate is for one H100! request and one model copy.
- The corpus is `c_1aa2c4f0d0b6c816fd37aa5748c33341`.
- Ground truth collection `gt_77bb8b128743a79aedddaa24c808c3f8`, the
  active selectivity estimate collection, supplies exact survivors for
  every predicate, including the two agent trace predicates.
- The full result is at `/results/sol/sol_quailb_sf0.1.json` on the
  `quail-results` Modal volume. Each query carries two estimates: `sol_s`,
  where every distinct token prefix in the corpus is computed once, and
  `per_document.sol_s`, where every document is computed once and reused
  only across its own questions. The file also records, per corpus, how
  many tokens are a prefix another document has.

The estimate counts modeled fresh tokens, attention pairs, KV reads, and KV
writes. It does not include kernel gaps, host work, or scheduling overhead. No
measured or fitted constant is used.

Filters use fixed selectivity estimates to choose their order. The calculation
then uses saved ground truth to find the exact rows that reach each later
stage. Join queries search every supported eager left deep relation order and
anchor choice. The search supports binary full joins, applies each available
crossing predicate immediately, and does not consider bushy plans. Document
prefix KV has unlimited capacity in this estimate, and a token prefix that
another document already computed is resident: each document pays only for
the tokens beyond its longest common prefix with the rest of its corpus, the
nodes of the corpus prefix trie. Work is ideally packed across the whole
query, even across operator barriers. This is separate from
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
that effect per query, and shows the agent queries, where the engines' own
prefix reuse falls short of the estimate's.

## Prediction

The 30-query result was expected to match the old 30-query subset because the
new ground truth collection reuses the same label sets.

## Result

That prediction was wrong. The old SoL file used 200 BioDEX reports and 100
FEVER claims. The active scale factor 0.1 corpus has 500 of each. The measured
engine runs also used 500 of each, so the regenerated estimate is the correct
comparison.

- The distinct prefix 4B estimates total 464.74 seconds over 32 queries;
  the per document ones total 632.66 seconds. On the 30 non agent queries
  the two are 369.40 and 370.34 seconds: the review, report, claim, and
  passage corpora share under 1% of their tokens as prefixes.
- The distinct prefix 32B estimates total 3,425.75 seconds; the per
  document ones total 4,466.82 seconds.
- The two agent queries total 95.34 seconds for 4B and 543.16 seconds for
  32B. Each is one filter over 1,772 traces of 9,736 mean tokens. Trace
  rows sampled from the same trajectory are prefixes of each other, and
  68.9% of the corpus tokens are a prefix some other row already contains.
  The per document estimate for the two queries is 262.32 seconds for 4B;
  the distinct prefix estimate is 95.34.
- IMDB totals 108.40 seconds for 4B.
- BioDEX totals 116.14 seconds for 4B.
- FEVER totals 80.47 seconds for 4B.
- LePaRD totals 64.40 seconds for 4B.
- Agent totals 95.34 seconds for 4B.
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
