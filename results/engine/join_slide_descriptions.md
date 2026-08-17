# Slide descriptions for join experiment

Each section describes one method/baseline: what it is, how it is
configured, what it measured.

---

## Stock vLLM (grouped)

Standard vLLM 0.26, prefix caching on, one synchronous request per
(report, term) pair. Pairs are submitted grouped by report so the
engine's prefix cache can match the shared report prefix. Output is
constrained to YES/NO with max_tokens = 1 (zero decode — no
autoregressive generation step). KV cache is bf16 (the fairest
engine setting). Admission is controlled by max_num_seqs = 249,
derived from the same token budget formula as the packed run.
max_num_batched_tokens = 25,305. This is the best-case stock
configuration: grouped submission order, admission tuned, zero
decode, eviction-free at this sample size (100 report prefixes fit
the pool). Measured: **433 s** for 256,000 pairs. About 199 s of
that is GPU computation; the rest is host-side overhead from
processing 256,000 individual request objects.

---

## Packed forward pass, B = B* = 421,752

All 256,000 pairs processed in a single loop of 100 packed chunks,
no vLLM engine. The chunk budget B* comes from the memory
constraint act x B + reservation <= M_free (suffix tokens never
persist anything, so activations are the only per-token memory),
over a declared slack of 2. At this budget each chunk is one report
prefix followed by all 2,560 of its term suffixes — the whole
partner list fits beside its prefix (m = 1), so each prefix is
computed exactly once, nothing is kept, and nothing is recomputed.
Attention is split into two FlashAttention-3 calls per layer:
(1) causal self-attention within each segment (prefix over itself,
each suffix over itself), and (2) non-causal cross-attention from
all suffix tokens to the shared prefix KV. The two calls are merged
by softmax state (the exact convex form). Suffix tokens never write
KV — they attend to the prefix and produce an answer, so nothing
persists between pairs. YES/NO answers are read from the prefill
logits at each suffix's last token position. Wall time includes
chunk packing (building GPU tensors from token lists). Measured:
**98.7 s** at 85,300 tokens/s, **4.4x faster than stock**.

---

## 3-way staged join (planted)

Three 100-document collections (A, B, C) from IMDB reviews. Each B
document is ~4,000 tokens and carries two planted keys (X and Y). A
documents carry X, C documents carry Y. The join finds triples
(a, b, c) where a.X = b.X and b.Y = c.Y — the true output is
known by construction. Executed as two packed forward passes with
Python staging between them:

- **Stage 1**: B anchors, A streams as suffixes. Record all YES/NO
  answers. Drop B documents with zero YES answers. Each surviving B
  document enters stage 2 once, regardless of how many A documents
  matched it.
- **Stage 2**: surviving B anchors, C streams as suffixes. B's
  prefix KV is kept from stage 1 (stored as plain tensors, not in
  any engine cache) and reused — no recomputation.
- **Assembly**: the recorded answer matrices from both stages are
  crossed in Python to produce the output triples. No model calls.

Two correctness checks: (1) the triple set matches a nested-loop
replay of the recorded answers identically (74,600 triples), so
any staging or bookkeeping bug would show as a difference. (2)
Stage-2 pair count = survivors x 100, confirming deduplication
reduced the work. Measured: **48.7 s** (stage 1) + **29.7 s**
(stage 2) = **78.4 s** total.
