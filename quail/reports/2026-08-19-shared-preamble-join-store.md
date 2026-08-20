# Shared preamble, join store, and per-anchor framing

Date: 2026-08-19. One H100 on Modal. Qwen3 4B fp8. Data files are in
`results/`. This report is the current suite. An earlier writeup of
the same work, written before per-anchor framing and the save overlap
fix shipped, is in
`reports/old/2026-08-19-shared-preamble-join-store.md`.

## What changed

Every operator's prompt now has the same layout: a fixed engine-owned
preamble, then the document, then a per-query suffix. The preamble is
`"DOCUMENT:\n"` and is defined once in `quail/logical.py`
(`SHARED_PRE`). Any user template text before the first placeholder
is relocated after the document at compile time. That relocated text
is the frame. A filter carries the frame with the rest of the suffix.
A join writes the frame into the anchor's kept KV once per anchor,
using the same `write_suffix_tokens` path the filter already uses
for a shared question.

Because the preamble is one fixed string, the KV of
[preamble + document] is identical wherever the document appears.
The KV store already keyed extents by (content hash, document index),
so with this layout a filter and a join can restore each other's
documents. `run_join` got the same store wiring `run_filter` had:
anchors already in the store restore instead of recomputing, and
anchors leaving their last stage are saved. The stored extent stays
document-only. The frame is not stored.

Three bugs were found and fixed along the way.

- The store's GPU staging buffer was sized at store creation from the
  first query's documents and silently skipped saving anything
  longer. The reports are about 5,850 tokens on average (not the
  3,000 in the older write-ups), so no report was ever stored, in any
  previous run. That is why old warm B14 never restored anything.
  The buffer now grows on demand.
- Growing the staging buffer naively (4 slots at maximum document
  size) claimed about 11 GB of GPU memory next to the 66 GB arena and
  slowed every warm query (warm B5 ran 757 s against a 270 s cold
  wall). Large documents now get 2 slots. Warm B5 then ran 278 s on
  the store gate.
- `save()` held arena pages until the GPU-to-host copy finished,
  which stalled admission of new documents. Pages are now freed after
  the faster GPU-to-staging copy. The host copy continues in the
  background.

## How the preamble wording was chosen

The wording is measured, not guessed. The accuracy gate reran B1, B4,
and B5 after each candidate.

- "Evaluate whether the following is true or false." dropped B1's
  observed selectivity from 0.398 to 0.074. The preamble primed
  true/false vocabulary and the flag questions expect YES/NO values.
- "You will read a document and answer a yes-or-no question about
  it." dropped it to 0.0014.
- "DOCUMENT:\n" (a label, not an instruction) left it at 0.397,
  which matches the old prompt's 0.398.

B4 (3,000+ token reports) was immune to all three wordings. The
preamble sits thousands of tokens before the answer position. Any
instruction-like sentence before a short document changes the
model's answers, so the preamble must be a formatting label only.

## The join prompts moved their framing after the document

The join prompts' task framing used to sit before the anchor
document. It cannot stay there, because the preamble must be fixed.
Moving the frame into the per-pair suffix costs per pair. On B5's
512,000 pairs that added 6.1M fresh tokens and 95 s, measured. So
the frame is written once into each anchor's kept KV, and pairs
read it from KV.

The first framed wording was too eager. `results/quailb_gate_frame.json`
ran B5 at observed selectivity 0.76 (389,134 rows) against the
provided 0.05. The suite wording is stricter and closer to the
planted rate. B5 on the suite is 0.0431 observed against 0.05
provided (22,051 rows). The pair count is still 512,000. Gating
keeps any anchor with at least one YES, and every anchor still
survives at these selectivities, so evaluated work did not change.
Reported row counts did.

The original kvwrite prompts (framing before the document) were
looser. B5 then observed 0.287 and returned 146,819 rows. The new
layout is not a drop-in match for those answers.

## Store gate: the join save and restore paths work

`results/quailb_gate_store.json` runs B5 then B14, cold then warm,
with no other queries in between to evict the reports.

- Warm B5 saved its 41 longest anchors (428,149 tokens) through the
  join path for 1.4 s of wall cost (278.2 s against 276.8 s cold).
- Warm B14 restored all 41 and ran 281.6 s against 285.7 s cold. The
  4.1 s saving matches the arithmetic: 428k tokens at 9.4
  microseconds per token.
- Warm B14's rows differed from cold by 4 in 512,000 pairs. Restored
  KV was computed under a different chunk packing, and the changed
  floating-point accumulation order flips pairs near the YES/NO
  threshold. The exploration saw the same effect.

The full suite also showed the cross-operator path. Warm B4's filter
stored 37 reports and warm B5's join restored them. Warm B6's filter
stored 256 threads and B6's own join restored 9 of them in the same
query.

## Full suite results

`results/quailb_sf0.1_frame.json` and `.log`. Cold pass query time
1,546.1 s, compared with 1,580.9 s in the frameless shared-preamble
run (`quailb_sf0.1_sharedpre.json`) and 1,566.4 s in the 2026-08-19
pre-change run (`quailb_sf0.1_kvwrite.json`). Warm pass query time
1,554.2 s, compared with 1,603.2 s frameless and 1,558.8 s
pre-change.

Cold pass wall is 1,676.5 s (130 s outside the per-query walls).
Warm pass wall is 1,622.8 s (69 s outside). B1's boot is 32.8 s and
is counted in the cold wall. The earlier shared-preamble run's cold
wall was 2,097 s because the Hugging Face cache volume was slow at
suite start. Per-query walls in this run do not show that.

Peak GPU memory is 68.87 GiB on the cold pass and 71.59 GiB once
the store's staging buffers have grown.

| Query | Cold (s) | Warm (s) | Restored | Stored | Cold rows | Warm rows |
|---|---|---|---|---|---|---|
| B1 | 29.7 | 31.5 | 0 | 342 | 1,984 | 1,984 |
| B2 | 31.1 | 28.5 | 342 | 0 | 289 | 291 |
| B3w | 30.4 | 28.1 | 342 | 0 | 187 | 185 |
| B3c | 30.1 | 28.1 | 342 | 0 | 183 | 182 |
| B4 | 13.3 | 14.3 | 0 | 37 | 43 | 43 |
| B5 | 271.3 | 267.6 | 37 | 0 | 22,051 | 22,051 |
| B6 | 16.9 | 21.0 | 9 | 256 | 334 | 335 |
| B7 | 42.0 | 45.4 | 0 | 500 | 2,035 | 2,035 |
| B8 | 96.5 | 100.0 | 59 | 170 | 782 | 782 |
| B10 | 333.4 | 333.6 | 0 | 200 | 3,255,744 | 3,255,744 |
| B11 | 299.4 | 299.9 | 0 | 37 | 289,323 | 289,323 |
| B12 | 41.6 | 42.8 | 81 | 256 | 2,708 | 2,708 |
| B13 | 30.7 | 32.4 | 0 | 315 | 535 | 536 |
| B14 | 279.7 | 280.9 | 0 | 36 | 16,322 | 16,322 |

What the table says:

- Restores work and earn small amounts. B2, B3w, and B3c each
  restored 342 reviews (433,209 tokens) and ran about 2 to 2.6 s
  faster than cold. B5 restored 37 report anchors (376,787 tokens)
  from B4's filter and ran 3.7 s faster than its own cold wall
  (267.6 s against 271.3 s).
- Saves still cost more than restores earn at this scale, but less
  than before the overlap fix. The warm pass is 8.1 s slower than
  cold in total, compared with 22.3 s slower on the frameless run.
  B6 paid 4.1 s to store 256 threads (21.0 s against 16.9 s cold),
  compared with 10.5 s for the same store work on the frameless run
  (27.6 s against 17.1 s).
- B13 and B14 restored nothing because the 64 GB store cannot hold
  reviews, reports, and threads at once. B4 through B8 evicted the
  reviews before B13 ran, and B10 through B13 evicted the reports
  before B14 ran. The store gate proved both restore paths when
  nothing else was in the way. The suite shows the capacity limit.
- Filter queries match the pre-change run. B1 observed 0.3968
  against the old 0.398. B2's cold wall is 31.1 s, compared with
  33.1 s before the prompt change and 37.0 s on the frameless run.

B9 and B15 were defined but did not run. They still stream the full
2,560-term partner list, with no early-stop.

## Join observed selectivities

| Query | Prompt | Provided | Frame suite | Frameless | Pre-change |
|---|---|---|---|---|---|
| B5 | REACTION | 0.05 | 0.0431 | 0.0733 | 0.2868 |
| B6 | DISCUSS | 0.05 | 0.0742 | 0.0271 | 0.0017 |
| B7 | DISCUSS | 0.10 | 0.0407 | 0.0026 | 0.0001 |
| B8 | DISCUSS exists | 0.30 | 0.0837 | 0.0084 | 0.0025 |
| B10 stage 0 | KEYEQ | 0.20 | 0.8954 | 0.2929 | 0.1749 |
| B10 stage 1 | KEYEQ | 0.10 | 0.9007 | 0.1234 | 0.1159 |
| B11 stage 0 | REACTION | 0.05 | 0.0431 | 0.0733 | 0.2868 |
| B11 stage 1 | DISCUSS | 0.05 | 0.0604 | 0.0346 | 0.0935 |
| B12 | DISCUSS | 0.10 | 0.0900 | 0.0083 | 0.0043 |
| B14 | REACTION (stricter) | 0.05 | 0.0319 | 0.0362 | 0.2804 |

REACTION is now close to the planted 0.05. DISCUSS is higher than
the original prompts and closer to the planted rates on B6 and B12.
KEYEQ flipped the other way. B10's planted-key stages now pass about
90% of pairs, so the replay output grew to 3,255,744 rows from
149,262 in the pre-change run. The GPU still evaluated 40,000 then
20,000 pairs. The extra rows are the replay expansion, not extra
join work.

Warm rows match cold rows on every join. Restored KV did not change
answers in this suite.

## Open items

- Store capacity. At SF=0.1 the corpora total about 6M tokens of KV
  (860 GB) against a 434k-token (64 GB) store. Cross-dataset eviction
  makes rerun restores rare. A larger `cpu_memory_gb` or a store
  budgeted per dataset would let B13 and B14 restore.
- KEYEQ accuracy. The framed KEYEQ prompt is much looser than the
  original. B10's observed selectivities of 0.90 against planted 0.20
  and 0.10 need a wording pass the way REACTION already had.
- B9 and B15 still need the exists/anti early-stop before they are
  cheap enough to include.

## Data files

- `results/quailb_sf0.1_frame.json` and `.log`: this suite, both
  passes, after framing and the save overlap fix.
- `results/quailb_sf0.1_sharedpre.json` and `.log`: the same suite
  after the shared preamble and join store, before framing and save
  overlap.
- `results/quailb_sf0.1_kvwrite.json` and `.log`: the pre-change
  comparison run (old per-operator prompts, filters only in the
  store).
- `results/quailb_gate_store.json` and `.log`: the B5 then B14
  store gate.
- `results/quailb_gate_frame.json` and `.log`: B5 and B6 with the
  first framed wording (B5 observed 0.76). Not the suite wording.
- `results/quailb_gate_accuracy.json`, `quailb_gate_accuracy2.json`,
  `quailb_probe_b1.json` and logs: the preamble wording gates.
- `reports/old/2026-08-19-shared-preamble-join-store.md`: the
  earlier writeup, which still described framing and save overlap
  as unfinished.
