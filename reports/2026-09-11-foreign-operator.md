# A user function between two GPU operators: FEV-10 paired by apply()

RESULTS_SUMMARY

[![Query time, recomputed KV tokens, and fresh tokens for the three pairings](plots/foreign_operator.png)](plots/foreign_operator.png)

Figure: plots/foreign_operator.png

## Setup

- One Modal H100 running `Qwen/Qwen3-4B-FP8` with bf16 KV, sf=0.1,
  lf=1, three variants of FEV-10 in one process on one GPU, one
  unmeasured warmup run per variant after a FEV-1 run that absorbs the
  cold boot. Query time excludes model startup and result collection.
- The three variants ask SUPPORT of a claim and its own Wikipedia
  page after F11 on claims and F13 on evidence:
  - equality: FEV-10 as in QUAIL-B, `join(evidence, on=col("c.evidence_wiki_url")
    == col("e.id"))`. The session builds the pair table and the join
    streams each anchor's pairs ([the pair join report](2026-09-11-pair-join.md)).
  - per_batch: `join(evidence).apply(same_page, columns=[...])`. The
    same pairing done by a Python function (an Arrow hash join of the
    claim's page column against the evidence id). The planner keeps the
    anchor's chain streaming; the join calls the function on each batch
    of survivors the chain hands over, with the KV still pinned.
  - barrier: `join(evidence).apply_table(same_page, columns=[...])`.
    The same function, run once over every survivor. The planner does
    not pin the anchor's chain: it finishes first, its survivors go
    through the retention pool, and the join starts after the function.
- The anchor is evidence (the longer side, 287 rows of about 440
  tokens; 500 claims of about 12 tokens are the partners). Every
  variant plans the same filter order and anchor; only the pairing and
  the streaming edge differ. The planner prices the equality variant by
  its pair fraction (500 of 143,500 pairs) and the apply variants as a
  cross join, since it does not run the function at plan time.
- Cell: `experiments/cells/foreign_pairs.py`. Modal function call
  `FC_ID`. Results and answer tables are on `quail-results` at
  `/results/ablations/foreign-pairs-RUN_STAMP/` (one directory per
  variant with `summary.json`, `rows.parquet`, and the answer tables).

## Prediction

- Stated before the run: per_batch gives the same 185 pairs, the same
  answer table and 145 rows as the equality run, zero recomputed KV
  tokens, the same 187,567 fresh tokens, and a query time within noise
  of the equality run's 1.66 seconds, because the function runs on
  batches of at most a few hundred ids and each call is an Arrow hash
  join well under a millisecond.
- barrier: the anchor's chain (evidence, 171 survivors) finishes
  before the function runs, so the join cannot overlap it. At sf=0.1
  those survivors fit the retention pool (about 75,000 tokens against
  a 141,488-token cap), so no KV is recomputed and the cost is the lost
  overlap only: about 0.1 to 0.4 seconds over the equality run, with the
  same answers and rows.

## Results

RESULTS_TABLES

## What the numbers mean

RESULTS_MEANING
