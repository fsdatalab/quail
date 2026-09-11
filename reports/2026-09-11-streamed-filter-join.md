# Filter survivors streamed into the join

- The first join group's anchor filter now streams each passing document
  into the join with its KV pinned, instead of finishing over the whole
  corpus first and leaving the join to find survivors in the retention
  pool or recompute them. This removes every recomputed anchor prefix on
  the queries that had them.
- RESULTS_PLACEHOLDER

## Setup

- One Modal H100 running `Qwen/Qwen3-4B-FP8` with bf16 KV, sf=0.1,
  lf=1. Both implementations ran in the same container on the same GPU,
  each in its own process, with one unmeasured warmup run per query
  before the measured run.
- The baseline is main at commit `8338d92`: operator-at-a-time
  execution, where a filter chain finishes for all documents before its
  join starts and survivors reach the join through the retention pool
  (the arena minus two chunk budgets, 8,843 pages) or are recomputed.
  The comparison is this branch with the streamed edge. Nothing else
  differs between the two checkouts.
- Chunk budget 110,376 tokens and KV budget 362,250 tokens for both, as
  the planner derives them. There is no baseline-only setting: the change
  is in when the join reads a survivor's KV, not in any budget.
- Queries, chosen because they are the five with nonzero recomputed KV in
  the saved suite run ([saved results](2026-09-05-quailb-saved-results.md)):
  - IMDB-3: one filter on reviews, then a join with 12 aspects. 5,000
    reviews of about 299 tokens; the join needs about 150 anchors to fill
    a chunk, so the join runs about two chunks per filter chunk.
  - IMDB-4 and IMDB-5: two and three filters on reviews, then the same
    join.
  - IMDB-10: one filter on reviews, then a three-join chain whose first
    group anchors on the filtered reviews.
  - BIO-3: one filter, then a join with longer documents.
- Cell: `experiments/cells/streamed_filter_join.py`. Modal function call
  `fc-01M27GHD7K71XA6RNX9XEY6XP8`. Results, answer tables, and the prediction as stated
  before the run are on `quail-results` at `/results/ablations/DIR_PLACEHOLDER/`.

## Prediction

- Stated before the run: recomputed KV tokens go from 1,217,732 (IMDB-3),
  361,440 (IMDB-4), 188,587 (IMDB-5), 1,217,732 (IMDB-10), and 919,409
  (BIO-3) to 0, and fresh tokens fall by exactly those counts, because a
  streamed anchor packs the frame and partner suffixes only, the same as
  a resident one.
- At the saved runs' 8.4 to 11.9 microseconds per fresh token that is
  about 10.4, 3.1, 1.6, 10.7, and 11.0 seconds: IMDB-3 near 22 seconds
  from 32.35, IMDB-4 near 17 from 19.99, IMDB-5 near 16.2 from 17.76,
  IMDB-10 near 49 from 59.71, BIO-3 near 79 from 89.92.
- Answer tables, evaluated pairs, and returned rows should be identical
  between the two implementations: the same prefixes, frames, and
  suffixes are computed, only their chunk placement moves. The baseline
  should reproduce the saved suite numbers within run-to-run noise.

## Results

RESULTS_TABLES_PLACEHOLDER

## What the numbers mean

MEANING_PLACEHOLDER
