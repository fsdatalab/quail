# Three filter-chain queries with no join

Date: 2026-08-25. No engine change and no run: this adds three
queries to the QUAIL-B suite in `quail/bench/quailb.py`.

## What changed

Three new queries, taking the suite from 23 to 26:

| Query | Shape | Same filters as |
|---|---|---|
| IMDB-6 | F1 -> F4, 2 filters, no join | IMDB-4 |
| IMDB-7 | F1 -> F4 -> F5, 3 filters, no join | IMDB-5 |
| LEP-8 | LEP1..LEP5, 5 filters, no join | LEP-6 |

No new predicates. Each query reuses the same prompts, in the same
order, as the query it is the filter prefix of.

## Why

Before this, every query with more than one filter ended in a join.
The only join-free queries were the four single-filter ones (IMDB-1,
BIO-1, FEV-1, LEP-1). So the suite could not measure what a filter
chain costs on its own.

That matters because a filter chain's later stages do not rebuild a
document's KV. Stage 0 builds it and keeps the document rows plus the
token prefix its question shares with the later stages' questions
(`_shared_preamble_tokens` in `executor/loop.py`). Every later stage
then computes only the tokens of its own question past that shared
prefix, reading the KV that is already there.

The saving is large. For IMDB-6 over the 5,000 IMDB reviews at
sf=0.1: F1 and F4 share an 18-token prefix, so stage 1 computes 30
tokens per surviving document rather than its full 48, against a
context of `2 + review + 18`. Rebuilding instead of reusing would
raise the token count from 1,898,163 to 3,215,318, a factor of 1.69.
Until now that saving was always measured together with join work,
which is a much bigger and differently shaped cost.

Because each new query is an exact filter prefix of an existing one,
subtracting the two isolates the join:

    IMDB-4 - IMDB-6 = the cost of J1 after 2 filters
    IMDB-5 - IMDB-7 = the cost of J1 after 3 filters
    LEP-6  - LEP-8  = the cost of the self-join after 5 filters

LEP-8 is the deepest filter chain in the suite: five stages of KV
reuse with nothing else mixed in.

## Numbers

None measured. These queries have not been run on a GPU. The 1.69x
figure above is computed from the committed IMDB-1 token count, the
tokenized corpus, and the engine's own shared-preamble rule, not from
a run of IMDB-6.

The predicates carry no selectivity hints, matching every other query
in the suite, so the planner uses `as_written` ordering.

## Checks

- `uv run pytest tests/ --ignore=tests/gpu` - 185 passed.
- `test_all_queries_compile_and_plan` now covers all 26; each new
  query plans as `DocScan -> FilterChain -> Sink` with zero join
  stages.
- Verified each new query's filter stages match its counterpart's
  exactly (written positions [0,1], [0,1,2], [0,1,2,3,4]).
