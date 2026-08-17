# Join prototype findings

Qwen3 4B fp8, one H100. BioDEX 2-way (100 reports x 2,560 terms =
256,000 pairs) and a planted 3-way chain (100 x 100 x 100). All
numbers from `join2way.json`, `join_nway3.json`, `join_probe.json`.

Wall times include end-to-end cost: chunk packing (building GPU
tensors from raw token lists) plus forward passes plus answer
readout.

---

## 2-way results

| Method | Wall (s) | Tokens/s | Fresh tokens | Chunks |
|---|---|---|---|---|
| Stock vLLM (grouped) | 433 (mean of 449, 418) | 17,000 | 7.36M | — |
| Packed, B = B* = 421,752 | 98.7 (mean of 98.4, 99.0) | 85,300 | 8.42M | 100 |

The packed pass is **4.4x faster** than stock vLLM.

At B*, each report's full term list fits beside its prefix in one
chunk (m = 1), so each prefix is computed exactly once: 100 chunks,
8.42M fresh tokens, nothing kept and nothing recomputed. The packed
prediction was 92 s; measured 98.7 s (+7%).

Stock's effective rate is 17,000 tokens/s. Its 7.36M tokens take
only ~199 s at GPU speed. The remaining ~234 s is host-side work:
ingesting, hashing, and scheduling 256,000 request objects. This is
the missing cost-model term (finding 1 below).

A second packed configuration at B = 25,305 ran as a rate control
and was dropped from the reported results: its prefix recomputation
changes the prefix/suffix token mix (12.9% prefix tokens vs 3.6% at
B*), and prefix tokens cost about half a suffix token's attention
work, so equal rates would not isolate chunk size. A clean control
would compare two budgets with the same m — for example 44,000 and
84,000, both m = 2, identical tokens and mix. Not run.

---

## 3-way results

| Stage | Wall (s) | Pairs | Survivors |
|---|---|---|---|
| Stage 1 (A-B) | 48.7 | 10,000 | 100 (all) |
| Stage 2 (B-C) | 29.7 | 10,000 | — |
| **Total** | **78.4** | — | 74,600 triples |

Stage-2 pair count is exactly survivors x 100 = 10,000, confirming
that each surviving B document runs once against all C documents,
not once per matching A document. The triple set from staged
execution matches the nested-loop replay of the recorded answers
identically (74,600 triples).

All 100 B documents survived because the 4B checkpoint answers YES
to nearly everything (9,763 of 10,000 stage-1 answers wrong against
the planted keys). The filter executed zero skips on GPU. The
filtering and deduplication logic is covered by unit tests and the
replay check.

---

## Correctness gates (probe)

- Shared-prefix attention math: max diff 0.0068 against an fp32
  reference
- Shared vs unshared disagreements: 0 of 64
- Kept-KV vs shared disagreements: 0 of 64
- Rate at the B* geometry: 83.6k tokens/s

---

## Findings

**1. The cost model is missing a host ingestion term for stock
vLLM.** Predicted stock wall was 199 s (GPU-side computation only).
Measured was 433 s. The difference (~234 s) is host-side work:
ingesting 770M prompt tokens across 256,000 request objects. The
packed side submits 100 pre-built chunks and has no analog. This
term must be added to cost.py before any full-scale stock
prediction.

**2. The cross-attention call is roughly 30% of the packed wall at
these shapes.** The effective rate (85.3k tokens/s) is lower than
the pure-packed filter rate (121k tokens/s) because BioDEX prefixes
are 3,000 tokens — every suffix attends to all of them via the
cross-attention call. At the filter's shape (270-token prefixes),
cross-attention was 12% of the wall. The cost comes from the pair
count in the suffix-to-prefix attention, not the batch size.

**3. The 4B checkpoint is a near-unusable judge for both
predicates.** 2-way: precision 0.33% (YES on 60% of pairs). 3-way:
9,763 of 10,000 wrong. This breaks the planted instrument (no
filtering exercised) but no execution claim — filtering and
deduplication are validated by the replay check and unit tests.
Follow-up: use a single-flag lookup predicate the model can handle,
or a stronger model.
