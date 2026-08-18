# Join prototype findings

Qwen3 4B fp8, one H100. BioDEX 2-way (100 reports x 2,560 terms =
256,000 pairs) and a planted 3-way chain (100 x 100 x 100). All
numbers from `join2way.json`, `join_nway3.json`, `join_probe.json`.

One executor (`run_join`) runs every join: brim-packed chunks,
each anchor's prefix computed exactly once (a stream cut at a
chunk boundary writes the prefix KV to an explicit per-anchor
cache and the continuation reads it), the next chunk built on the
CPU while the GPU runs, answers crossing as event-synced byte
copies. The 2-way is its one-stage case; the 3-way its two-stage,
gate-per-anchor case. Wall times are end to end: chunk building,
forward passes, answer readout.

---

## 2-way results

| Method | Wall (s) | Tokens/s | Fresh tokens | Chunks |
|---|---|---|---|---|
| Stock vLLM (grouped) | 429 (mean of 442, 415) | 17,200 | 7.36M | — |
| Packed, B = 110,376 (kernel cap) | 103.6 (2 reps, spread 0.1 s) | 81,300 | 8.42M | 77 |

The packed pass is **4.1x faster** than stock vLLM.

The packed budget is min(B*, kernel cap): the memory bound B* =
421,752 is not executable (finding 2), so chunks run at 110,376.
Each report's prefix is computed exactly once — 76 of 100 reports
are cut at a chunk boundary and their continuations read the
cached prefix KV. Fresh tokens are exactly the computed-once
count, 8,417,425.

Against the pre-rewrite executor's 98.7 s (one report per chunk,
~84k tokens, no cache path): +5%. About 2 points are the
generalized attention's gather cost (the probe's rate gate moved
83.6k -> 82.1k tokens/s); the rest is unattributed — candidates
are the 76 cache-read continuations and allocator pressure at the
48.3 GiB peak. That peak was itself a bug the run exposed: the
executor freed cut anchors' cached KV only at stage end, holding
~76 x 0.54 GB at once. Fixed after the run (each anchor's KV is
freed at its last chunk); expected peak ~12-16 GiB, unmeasured.

Answers: yes = 177,831 (was 177,830), agreement with stock 213,976
of 256,000 (was 213,989) — knife-edge flips from prefixes now
computed at different chunk shapes, within the predicted "tens."
Stock itself is not bit-stable: its two reps disagree with each
other on 12 pairs (yes = 154,731 vs 154,719), so packed-vs-stock
agreement sits on top of that noise floor.

Stock's effective rate is ~17,200 tokens/s. Its 7.36M fresh tokens
take only ~199 s at GPU speed; the remaining ~230 s is host-side
work ingesting 256,000 request objects (finding 1). Four stock
walls measured across two containers: 449, 418, 442, 415 s.

---

## 3-way results

| Stage | GPU (s) | Pairs | Survivors |
|---|---|---|---|
| Stage 1 (B x A, prefix KV written) | 48.4 | 10,000 | 100 (all) |
| Stage 2 (B x C, reads cached KV) | 45.5 | 10,000 | — |
| **Total wall** | **95.6** | — | 991,911 triples |

Stage times are CUDA-event sums per stage; the loop is
software-pipelined, so the wall (95.6 s) is slightly more than
their sum. The stages are near-equal because both are dominated by
per-pair suffix work — C suffixes are even slightly longer than A
(355 vs 310 tokens). The cached KV removes only the once-per-
document prefix term (~10% of stage 2); every suffix still pays
its cross-attention read over B's 3,684 positions.

**These numbers replace the earlier 78.4 s / 74,600-triple result,
which was invalid** — see finding 4. Checks: the triple set equals
a nested-loop replay of the recorded answers exactly; stage-2
pairs = survivors x 100 = 10,000. All 100 B documents survived
(planted design expected 80) because of finding 5.

An estimated grouped-stock baseline for this workload — same
submission strategy as the 2-way stock arm, arithmetic from
measured constants, never run — is ~134 s
(`join_nway3_vs_stock.png` shows the four components). The gap
(1.4x) is smaller than the 2-way's 4.1x because 310-355-token
suffixes amortize stock's per-request costs ~10x better than
32-token suffixes.

---

## Correctness gates (probe)

- Shared-prefix attention math: max diff 0.0068 against an fp32
  reference
- Shared vs unshared answers: 0 of 64 disagree
- Cached-KV replay vs in-chunk: 0 of 64
- Several fresh prefixes in one chunk vs each alone: 0 of 64
- Fresh + cached prefix mixed in one chunk: 0 of 64
- Rate at the executor's chunk geometry: 82.1k tokens/s

---

## Findings

**1. The cost model is missing a host ingestion term for stock
vLLM.** Predicted stock wall was 199 s (GPU-side computation
only). Measured is ~429 s. The difference is host-side work:
ingesting 770M prompt tokens across 256,000 request objects. The
packed side submits 77 pre-built chunks and has no analog. Add the
term to cost.py before any full-scale stock prediction.

**2. The chunk budget has two ceilings, and the kernel one binds.**
The memory formula gives B* = 421,752, but the fused kernels
compute element offsets in 32-bit ints, so a chunk needs rows x
widest-projection-width < 2^31 — at most 110,375 tokens here
(gate_up is 19,456 wide). A 421,750-token chunk dies with an
illegal memory address. The cap is derived from the loaded weights
at runtime; rate is flat in chunk size, so it costs nothing. No
earlier run hit this because one-report chunks (84k tokens) sat
24% under the bound by accident.

**3. The cross-attention call is roughly 30% of the packed wall at
these shapes.** The effective rate (81.3k tokens/s) is far below
the pure-packed filter rate (121k) because BioDEX prefixes are
3,000 tokens — every suffix attends to all of them. At the
filter's 270-token prefixes, cross-attention was 12% of the wall.
The cost scales with the suffix-to-prefix pair count, not the
batch size.

**4. The original kept-KV path never read the cached KV — and an
answers-only gate missed it.** The old chunk builder set
`prefix_rows = 0` for cached-KV chunks, which triggered the
no-sharing early-return in attention: 3-way stage 2 ran without
the anchor document visible at all. The old probe's kept gate
passed anyway, because it compared answers only, on 64 short term
suffixes where this near-always-YES judge answers the same with or
without the report. The rewritten probe cross-checks cached
against fresh inside one chunk (structurally independent of the
judge) and caught the fix. Lesson: with a degenerate judge,
validation must never lean on answer agreement alone.

**5. The 4B checkpoint is a near-unusable judge for both
predicates.** 2-way: precision 0.33% (YES on ~70% of pairs).
3-way: ~99% YES on both stages, so all 100 B documents survive
(planted: 80) and the model's 991,911 triples dwarf the planted
640. This saturates the planted instrument but breaks no execution
claim — execution correctness rests on the replay check, the
pair-count identity, and the judge-independent probe gates.
Follow-up: a single-flag lookup predicate this model can read, or
a stronger judge.
