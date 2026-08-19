# Shared preamble and the join KV store

Date: 2026-08-19. One H100 on Modal. Qwen3 4B fp8. Data files are in
`results/`.

## What changed

Every operator's prompt now has the same layout: a fixed engine-owned
preamble, then the document, then a per-query suffix. The preamble is
`"DOCUMENT:\n"` and is defined once in `quail/logical.py`
(`SHARED_PRE`). Any user template text before the first placeholder
is relocated after the document at compile time.

Because the preamble is one fixed string, the KV of
[preamble + document] is identical wherever the document appears.
The KV store already keyed extents by (content hash, document index),
so with this layout a filter and a join can restore each other's
documents. `run_join` got the same store wiring `run_filter` had:
anchors already in the store restore instead of recomputing, and
anchors leaving their last stage are saved.

Two bugs were found and fixed along the way:

- The store's GPU staging buffer was sized at store creation from the
  first query's documents and silently skipped saving anything
  longer. The reports are about 5,850 tokens on average (not the
  3,000 in the older write-ups), so no report was ever stored, in any
  previous run. That is why old warm B14 never restored anything.
  The buffer now grows on demand.
- Growing the staging buffer naively (4 slots at maximum document
  size) claimed about 11 GB of GPU memory next to the 66 GB arena and
  slowed every warm query (warm B5 ran 757 s against a 270 s cold
  wall). Large documents now get 2 slots. Warm B5 then ran 278 s.

## How the preamble wording was chosen

The wording is measured, not guessed. The accuracy gate reran B1, B4,
and B5 after each candidate:

- "Evaluate whether the following is true or false." dropped B1's
  observed selectivity from 0.398 to 0.074. The preamble primed
  true/false vocabulary and the flag questions expect YES/NO values.
- "You will read a document and answer a yes-or-no question about
  it." dropped it to 0.0014.
- "DOCUMENT:\n" (a label, not an instruction) left it at 0.397,
  which matches the old prompt's 0.398.

B4 (3,000+ token reports) was immune to all three wordings: the
preamble sits thousands of tokens before the answer position. The
lesson is that any instruction-like sentence before a short document
changes the model's answers, so the preamble must be a formatting
label only.

## The join prompts changed, and their answers moved

The join prompts' task framing ("You will be shown a patient report
and one candidate medical reaction term...") used to sit before the
anchor document. It cannot stay there (the preamble must be fixed),
and moving it into the per-pair suffix costs per pair: on B5's
512,000 pairs it added 6.1M fresh tokens and 95 s, measured. So the
framing was dropped and the suffixes kept the old tail wording.

Without framing the model is stricter. B5's observed selectivity fell
from 0.287 to 0.073, and B10's planted-key stage rose (more rows
matched). Pair counts and evaluated work are unchanged, because
gating keeps any anchor with at least one YES and every anchor still
survives at these selectivities. Only reported row counts moved.

The right mechanism for framing, not built yet: write a per-query
framing segment into the anchor's kept KV once per anchor, the way
the filter already keeps its shared question preamble
(`write_suffix_tokens`). The framing would then cost tokens per
anchor instead of per pair, and the stored document prefix would stay
query-independent.

## Store gate: the join save and restore paths work

`results/quailb_gate_store.json` runs B5 then B14, cold then warm.

- Warm B5 saved its 41 longest anchors (428,149 tokens) through the
  join path for 1.4 s of wall cost (278.2 s against 276.8 s cold).
- Warm B14 restored all 41 and ran 281.6 s against 285.7 s cold. The
  4.1 s saving matches the arithmetic: 428k tokens at 9.4
  microseconds per token.
- Warm B14's rows differed from cold by 4 in 512,000 pairs. Restored
  KV was computed under a different chunk packing, and the changed
  floating-point accumulation order flips knife-edge pairs. The
  exploration saw the same effect.

The full suite also showed the cross-operator path: warm B4's filter
stored 37 reports and warm B5's join restored them. Warm B6's filter
stored 256 threads and B6's own join restored 9 of them in the same
query.

## Full suite results

`results/quailb_sf0.1_sharedpre.json` and `.log`. Cold pass query
time 1,581 s, compared with 1,566 s in the 2026-08-19 pre-change run
(`quailb_sf0.1_kvwrite.json`). Warm pass query time 1,603 s.

| Query | Cold (s) | Warm (s) | Restored | Stored |
|---|---|---|---|---|
| B1 | 30.0 | 32.7 | 0 | 342 |
| B2 | 37.0 | 35.0 | 342 | 0 |
| B3w | 31.1 | 29.0 | 342 | 0 |
| B3c | 30.5 | 28.6 | 342 | 0 |
| B4 | 13.5 | 14.7 | 0 | 37 |
| B5 | 275.1 | 273.6 | 37 | 0 |
| B6 | 17.1 | 27.6 | 9 | 256 |
| B7 | 42.5 | 45.3 | 0 | 500 |
| B8 | 97.4 | 103.0 | 59 | 170 |
| B10 | 339.3 | 340.4 | 0 | 200 |
| B11 | 306.1 | 306.3 | 0 | 37 |
| B12 | 42.3 | 44.0 | 81 | 256 |
| B13 | 32.0 | 35.5 | 0 | 317 |
| B14 | 286.9 | 287.6 | 0 | 37 |

What the table says:

- Restores work and earn small amounts: B2, B3w, B3c each restored
  342 reviews and ran about 2 s faster than cold. B5 restored 37
  report anchors and beat its own cold wall.
- Saves cost more than restores earn at this scale. The warm pass was
  22 s slower than cold in total. B6 alone paid 10.5 s to store 256
  threads. The save path defers page frees until the host copy lands,
  which stalls admission when documents are large. This is the main
  optimization left on the store.
- B13 and B14 restored nothing because the 64 GB store cannot hold
  reviews, reports, and threads at once. B4 through B8 evicted the
  reviews before B13 ran, and B10 through B13 evicted the reports
  before B14 ran. The gate proved both restore paths; the suite shows
  the capacity churn honestly. A larger `cpu_memory_gb` or smarter
  cross-dataset budgeting would let the reruns restore.
- The filter queries match the pre-change run within noise once the
  selectivity shifts are accounted for (B2's five flags each pass
  slightly more documents now, so its cold wall grew 4 s with more
  survivors).

## Open items

- Save stalls: overlap the device-to-host copy with admission instead
  of holding pages until the copy lands (stage device-to-device
  first, or free pages per-layer as they are staged).
- Framing in kept KV: per-anchor task framing written after the
  document rows, per the mechanism above, to recover join accuracy
  without per-pair cost.
- Store capacity: at SF=0.1 the corpora total about 6M tokens of KV
  (860 GB) against a 434k-token (64 GB) store. Cross-dataset eviction
  makes rerun restores rare. Decide whether reruns matter enough to
  budget the store per dataset.
- The suite's cold pass wall (2,097 s) carried about 430 s of
  overhead outside the per-query walls that the warm pass (78 s
  overhead) did not. The Modal logs show the HF cache volume was
  degraded at suite start (48 s per safetensors shard against the
  usual 2 s), and B1's boot was also misattributed (reported 0.0
  while the logs show an 88 s boot). Per-query walls are unaffected;
  worth a look if it recurs.

## Data files

- `results/quailb_sf0.1_sharedpre.json` and `.log`: the full suite.
- `results/quailb_gate_store.json` and `.log`: the B5/B14 store gate.
- `results/quailb_gate_accuracy.json`, `quailb_gate_accuracy2.json`,
  `quailb_probe_b1.json` and logs: the preamble wording gates.
- `results/quailb_sf0.1_kvwrite.json`: the pre-change comparison run.
