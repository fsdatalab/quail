# Slide descriptions for join experiment

Each section describes one method/baseline: what it is, how it is
configured, what it measures.

---

## Stock vLLM (grouped)

Standard vLLM 0.26, prefix caching on, one synchronous request per
(report, term) pair. Pairs are submitted grouped by report so the
engine's prefix cache can match the shared report prefix. Output is
constrained to YES/NO with max_tokens = 1 (zero decode — no
autoregressive generation step). KV cache is bf16 (the fairest
engine setting). Admission is controlled by max_num_seqs = 249,
derived from the same token budget formula as the packed runs.
max_num_batched_tokens = 25,305. This is the best-case stock
configuration: grouped submission order, admission tuned, zero
decode, eviction-free at this sample size (100 report prefixes fit
the pool). Measured: **495 s** for 256,000 pairs. About 200 s of
that is GPU computation; the rest is host-side overhead from
processing 256,000 individual request objects.

---

## Packed forward pass, B = 25,305

All 256,000 pairs processed in a single loop of 400 packed chunks,
no vLLM engine. Each chunk is a contiguous token sequence of up to
25,305 tokens containing one or more report prefixes followed by
their term suffixes, packed to the boundary with no padding (brim
packing). Attention is split into two FlashAttention-3 calls per
layer: (1) causal self-attention within each segment (prefix over
itself, each suffix over itself), and (2) non-causal
cross-attention from all suffix tokens to the shared prefix KV.
The two calls are merged by softmax state (the exact convex form).
Suffix tokens never write to the KV cache — they attend to the
prefix KV but produce no persistent state, so there is nothing to
erase between pairs. YES/NO answers are read from the prefill
logits at each suffix's last token position. At this B, each
report's 2,560 terms span about 4 chunks (m = 4), so each report
prefix is computed 4 times. 25,305 is the largest point from the
measured throughput sweep, included as a reference to separate rate
changes at large B from kernel issues. Measured: **105.8 s** at
88,000 tokens/s.

---

## Packed forward pass, B = B* = 421,752

Same mechanism as above, but chunks are 421,752 tokens — the
derived budget B* from the formula act x B + reservation <= M_free,
with a slack of 2. At this B, each report's full partner list fits
one chunk (m = 1), so each prefix is computed exactly once. 256,000
pairs pack into 100 chunks. The rate is the same 88,000 tokens/s
(flat in B), but fewer prefix recomputations mean fewer total
tokens (8.42M vs 9.31M), saving 10 s. This run confirms the budget
formula and the rate plateau. Measured: **95.6 s**, **5.2x faster
than stock**.

---

## 3-way staged join (planted)

Three 100-document collections (A, B, C) from IMDB reviews. Each B
document is ~4,000 tokens and carries two planted keys (X and Y). A
documents carry X, C documents carry Y. The join finds triples
(a, b, c) where a.X = b.X and b.Y = c.Y — the true output is
known by construction. Executed as two packed forward passes with
Python staging between them:

- **Stage 1**: B anchors, A streams as suffixes. Record all YES/NO
  answers. Gate: skip B documents with zero YES answers. Dedup:
  each surviving B document enters stage 2 once, regardless of how
  many A documents matched it.
- **Stage 2**: surviving B anchors, C streams as suffixes. B's
  prefix KV is kept from stage 1 (stored as plain tensors, not in
  any engine cache) and reused — no recomputation.
- **Assembly**: the recorded answer matrices from both stages are
  crossed in Python to produce the output triples. No model calls.

Two correctness checks: (1) the triple set matches a nested-loop
replay of the recorded answers identically (74,600 triples), so
any staging or bookkeeping bug would show as a difference. (2)
Stage-2 pair count = survivors x 100, confirming dedup reduced the
work. Measured: **48.7 s** (stage 1) + **29.7 s** (stage 2) =
**78.4 s** total.
