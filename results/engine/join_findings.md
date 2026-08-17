# Join prototype findings

Qwen3 4B fp8, one H100. BioDEX 2-way (100 reports x 2,560 terms =
256,000 pairs) and a planted 3-way chain (100 x 100 x 100). All
numbers from `join2way.json`, `join_nway3.json`, `join_probe.json`.

---

## 2-way results

| Method | Wall (s) | Tokens/s | Fresh tokens | Chunks |
|---|---|---|---|---|
| Stock vLLM (grouped) | 495 (mean of 504, 487) | 14,900 | 7.36M | — |
| Packed, B = 25,305 | 105.8 (3 reps, spread 0.04 s) | 88,000 | 9.31M | 400 |
| Packed, B = B* = 421,752 | 95.6 (mean of 95.7, 95.5) | 88,000 | 8.42M | 100 |

The packed pass is **5.2x faster** than stock vLLM at the derived B*.

Packed processes more fresh tokens (9.31M vs 7.36M at B = 25,305)
because each anchor prefix is recomputed at the head of each chunk
group. At B*, each report fits one chunk (m = 1), so tokens drop to
8.42M. The 10.6 s gap between the two packed runs is entirely from
different token counts, not a rate difference — the rate is 88,000
tokens/s at both batch sizes.

Stock's effective rate is 14,900 tokens/s. Its 7.36M tokens take
only ~199 s at GPU speed. The remaining ~296 s is host-side work:
ingesting, hashing, and scheduling 256,000 request objects. This is
the missing cost-model term (finding 1 below).

Predictions landed within 4% for both packed runs. The stock
prediction (199 s) captured only the GPU terms; the full measured
wall is 2.5x higher because of the host ingestion overhead.

---

## 3-way results

| Stage | Wall (s) | Pairs | Survivors |
|---|---|---|---|
| Stage 1 (A-B) | 48.7 | 10,000 | 100 (all) |
| Stage 2 (B-C) | 29.7 | 10,000 | — |
| **Total** | **78.4** | — | 74,600 triples |

Stage-2 pair count is exactly survivors x 100 = 10,000, confirming
dedup (each surviving B document runs once against all C documents,
not once per matching A document). The triple set from staged
execution matches the nested-loop replay of the recorded answers
identically (74,600 triples).

All 100 B documents survived because the 4B checkpoint answers YES
to nearly everything (9,763 of 10,000 stage-1 answers wrong against
the planted keys). The gate executed zero skips on GPU. Gating and
dedup correctness is covered by unit tests and the replay check.

---

## Correctness gates (probe)

- Shared-prefix attention math: max diff 0.0068 against an fp32
  reference
- Shared vs unshared disagreements: 0 of 64
- Kept-KV vs shared disagreements: 0 of 64
- Rates: 83.6k tokens/s (B*), 84.2k tokens/s (25,305) — flat

---

## Findings

**1. The cost model is missing a host ingestion term for stock
vLLM.** Predicted stock wall was 199 s (GPU-side computation only).
Measured was 495 s. The difference (~296 s) is host-side work:
ingesting 770M prompt tokens across 256,000 request objects. The
packed side submits 100-400 chunks and has no analog. This term
must be added to cost.py before any full-scale stock prediction.

**2. Throughput is flat in B from 25,305 to 421,752.** Rate was
88,000 tokens/s at both batch sizes. The 10 s wall difference is
entirely from token count (B* has fewer chunks, therefore fewer
prefix recomputations). This confirms the budget formula's
prediction that larger B reduces chunks without changing per-token
cost, at least out to B* on this hardware.

**3. The merge call works but cross-attention is 27% of the packed
wall at these shapes.** The effective rate (88k tokens/s) is lower
than the pure-packed filter rate (121k tokens/s) because BioDEX
prefixes are 3,000 tokens — every suffix attends to all of them
via the cross-attention call. At the filter's shape (270-token
prefixes), cross-attention was 12% of the wall. The cost comes
from the pair count in the suffix-to-prefix attention, not the
batch size.

**4. The 4B checkpoint is a near-unusable judge for both
predicates.** 2-way: precision 0.33% (YES on 60% of pairs). 3-way:
9,763 of 10,000 wrong. This breaks the planted instrument (no
gating exercised) but no execution claim — gating and dedup are
validated by the replay check and unit tests. Follow-up: use a
single-flag lookup predicate the model can handle, or a stronger
model.
