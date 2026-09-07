# Two speed of light estimates and two KV regrets

The QUAIL-B speed of light (SoL) estimate and the KV regret metric now
come in two forms each.

- Per document: every document is computed once and reused only across
  its own questions. This is the execution model Quail and per document
  vLLM implement, and the regret the runs already recorded.
- Distinct prefix: every distinct token prefix in the corpus is computed
  once. The regret adds the shared prefix tokens an engine recomputed
  and subtracts the cached tokens its requests received from other
  documents' requests.

`quail.runtime.tokens.shared_prefix_lengths` computes the credit: sort
the token sequences and take each one's longest common prefix with its
predecessor, which sums to the size of the corpus prefix trie.
`reports/make_sol_quailb.py` produces both estimates in one run and
records the corpus prefix statistics. The request backends now split
cached tokens per request into the part the document's own earlier
request could explain, whose shortfall is per document regret, the
part beyond it that falls inside the document's own tokens,
`cross_row_cached_tokens`, and the rest (`cached_other_tokens`: the
preamble, label and question tokens, and the KV block that straddles
the end of a join prefix), which counts for neither. The first cut of
this split, on 2026-09-04, credited every cached token beyond the own
prefix as a cross row hit. On join queries that credited the
straddling block of every pair, about 7 tokens per pair, and made the
request backends' distinct prefix regret negative (BIO-2: -4,004,562
tokens over 563,500 pairs). `split_cached_tokens` in
`quail.backends.request_scheduling` clips the credit to the document
since 2026-09-05. Per document regret on the
request backends now includes filter stages, which used to count zero.
`quail.bench.evaluate.add_prefix_metrics` writes `shared_prefix_tokens`,
`cross_row_cached_tokens`, and `regret_distinct_tokens` into every
benchmark row.

Since 2026-09-05 a column scanned under several aliases counts as one
prefix trie in both places. The SoL's distinct prefix estimate lets an
anchor row pay only the frame when another alias of its column already
holds the prefix (FEV-7's second evidence alias, IMDB-9's second reviews
alias), and `scanned_shared_prefix_tokens` charges every copy beyond the
first in full. SGLang's radix cache served exactly those prefixes on
FEV-7, FEV-8, and FEV-9 (125,851 tokens, the whole evidence corpus), and
before this change that credit had no matching charge and its distinct
prefix regret came out negative.

On the QUAIL-B corpora the two SoL estimates differ by 0.3% except on the
agent traces, where 68.9% of tokens are a prefix another row already has
and the distinct prefix estimate is 2.8 times lower. On AGENT-1 Quail's
distinct prefix regret is 11,882,610 tokens and pipelined vLLM's is
20,386, which is the whole reason vLLM is faster there. See
the `quail-results` volume at
`/results/benchmarks/quailb/family-runs/20260905T021527Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/manifest.json`
and `/results/sol/sol_quailb_sf0.1.json`. Those files describe FEV-9 before
the additional claim and evidence filters were added.
