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
request could explain, whose shortfall is per document regret, and the
part beyond it, `cross_row_cached_tokens`. Per document regret on the
request backends now includes filter stages, which used to count zero.
`quail.bench.evaluate.add_prefix_metrics` writes `shared_prefix_tokens`,
`cross_row_cached_tokens`, and `regret_distinct_tokens` into every
benchmark row.

On the QUAIL-B corpora the two SoL estimates differ by 0.3% except on the
agent traces, where 68.9% of tokens are a prefix another row already has
and the distinct prefix estimate is 2.8 times lower. On AGENT-1 Quail's
distinct prefix regret is 11,882,610 tokens and pipelined vLLM's is
20,386, which is the whole reason vLLM is faster there. See
`reports/2026-08-31-quailb-kv-regret.md` and
`reports/2026-08-29-sol-quailb.md`.
