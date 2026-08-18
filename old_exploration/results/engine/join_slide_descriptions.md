# Slide descriptions for join experiment

Each section describes one method/baseline: what it is, how it is
configured, what it measured.

---

## Stock vLLM (grouped)

Standard vLLM 0.26, prefix caching on, one synchronous request per
(report, term) pair. Pairs are submitted grouped by report so the
engine's prefix cache can match the shared report prefix. Output is
constrained to YES/NO with max_tokens = 1 (zero decode). KV cache
is bf16 (the fairest engine setting). Admission is max_num_seqs =
249, derived from the same token budget formula as the packed run;
max_num_batched_tokens = 25,305. This is the best-case stock
configuration: grouped order, admission tuned, zero decode,
eviction-free at this sample size. Measured: **429 s** for 256,000
pairs (four walls across two containers: 449, 418, 442, 415).
About 199 s is GPU computation; the rest is host-side overhead
from processing 256,000 individual request objects. Stock is not
bit-stable across its own reps (12 of 256,000 answers differ).

---

## Packed forward pass (one executor for every join)

All 256,000 pairs in 77 brim-packed chunks, no vLLM engine. The
chunk budget is min of two ceilings: memory (act x B <= M_free
over slack 2 -> 421,752) and kernel indexing (chunk rows x the
widest projection width < 2^31 -> 110,375; derived from the loaded
weights) — the kernel one binds, so chunks are up to 110,376
tokens. Chunks pack to the brim across reports; every prefix is
computed exactly once: when the budget cuts a report's term stream
at a chunk boundary (76 of 100 reports), the prefix KV is written
to an explicit per-anchor cache — plain tensors, no paged pool, no
eviction — and the continuation's cross-attention reads it.
Attention per layer is two FlashAttention-3 calls merged exactly
by softmax state: causal self-attention within each segment, and
non-causal cross-attention from suffix tokens to their prefix KV
(fresh in-chunk or cached). Suffix KV is never cached. The CPU
builds the next chunk while the GPU runs the current one; YES/NO
bits cross as event-synced pinned copies. Wall is end to end.
Measured: **103.6 s** at 81,300 tokens/s, **4.1x faster than
stock**, answers matching stock on 213,976 of 256,000 pairs
(stock's own rep-to-rep noise is 12 pairs).

---

## 3-way staged join (planted)

Three 100-document collections from IMDB reviews. Each B document
is ~3,700 tokens (12 reviews) ending with two planted keys
`[KEYS] X=.. Y=..`; A documents carry an X key (50 values, 2 A's
each), C documents a Y key (25 values, 4 C's each). The query:
triples (a, b, c) with a.X = b.X and b.Y = c.Y. 80 of 100 B
documents have matchable X; 20 are planted unmatchable and should
die at the gate. True output by construction: 640 triples.

Executed by the same executor, two stages with a gate per anchor:

- **Stage 1**: [B prefix | all 100 A suffixes] per B, one chunk;
  the prefix KV is written to the cache. A B with zero YES is
  finished (the gate); survivors enter stage 2 once each (dedup).
- **Stage 2**: [all 100 C suffixes] against the cached prefix KV —
  no prefix recomputation. The loop is software-pipelined: stage 1
  of the next B is launched before the current gate resolves, so
  the GPU never idles.
- **Assembly**: the recorded answer matrices are crossed in
  Python. No model calls.

Measured: **48.4 s + 45.5 s of stage GPU time, 95.6 s wall**. The
stages are near-equal because per-pair suffix work dominates; the
cached KV removes only the once-per-document prefix term (~10% of
stage 2). Checks: the triple set equals a nested-loop replay of
the recorded answers exactly (991,911 triples), and stage-2 pairs
= survivors x 100. The judge answers ~99% YES, so all 100 B
survived (planted: 80) — the instrument is saturated (see
findings); execution correctness rests on the replay and
pair-count checks plus the probe's judge-independent gates.

---

## Estimated grouped-stock baseline for the 3-way

Never run — arithmetic from committed measurements
(`join_nway3_vs_stock.png`). Same submission strategy as the 2-way
stock arm: request per pair, grouped by B, prefix caching on (all
100 B prefixes fit the pool, so stock also computes each prefix
once). Components: 7.025M fresh tokens at stock's measured 70.8k
tokens/s fresh rate (99 s) + paged reads of the cached prefix
(20,000 x 3,684 x 125 ns = 9 s) + host ingestion (80.33M prompt
tokens at the measured 234 s / 770M = 24 s) + per-request fixed
(1 s) = **~134 s**, 1.4x the measured 95.6 s. The gap is smaller
than the 2-way's 4.1x because 310-355-token suffixes amortize
stock's per-request costs ~10x better than 32-token suffixes.
