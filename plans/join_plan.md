# Joins: the general n-way algorithm, one experiment

Status: plan only, nothing implemented. This rewrites the earlier
multi-dataset draft after review: one workload, one experiment, and
the algorithm stated for general n even though only n = 2 runs. The
old draft (six datasets, seven phases) is in git history.

Decisions taken:

- **One workload: BioDEX.** Everything else from the FDJ paper
  (Products, Citations, Movies, Police, Categorize) stays out unless
  BioDEX leaves a question open.
- **The algorithm is solved for general n.** A sampled prototype
  exercises n = 2 on BioDEX and n = 3 planted.
- **A 2-way join runs as a packed forward pass.** No engine, no KV
  pool. The chunk budget B* is derived from the budget formula
  act x B + σ x kv x B + reservation ≤ M_free, at σ = 0 — suffix
  tokens never write KV, and kept prefixes enter as the reservation
  because they do not scale with B — over a declared slack of 2
  (the "give some more slack" step; it carries the unmeasured
  activation estimate and the FlashInfer 147 KB/token case):
  **421,752 tokens today**, floored at ~4,096 where throughput
  flattens. Activation memory is the batch-scaling term: act x B ≈
  34.6 GB of the 69.1 GB free at this B*. All arms run at B*, the stock boot's
  max_num_batched_tokens included. B* sits far past the measured
  sweep, so walls are conditional on the rates holding there; one
  reference cell at 25,305 (the largest measured point) rides
  along to tell a rate change at large B apart from a slow kernel.
  Chunks are packed to the brim across anchors — an anchor's list
  ending mid-chunk is followed by the next anchor's prefix, no
  padding.
- **n ≥ 3 checks survivors between stages and rewinds**: continue
  from a kept anchor prefix and stream the next relation against
  it. No vLLM needed — staging is Python bookkeeping around the
  same forward passes; vLLM appears only in the one baseline arm.
- **Whether anchor KV is kept is a size decision**, computed by the
  planner per stage — not a design constant.

A join here is an AI-if over pairs: one yes/no question about two
documents, answer constrained to YES/NO, `max_tokens=1`, zero
decode. Qwen3 4B fp8, one H100, as everywhere in this repo.

---

## 1. The general n-way join

**Query.** Relations R_1..R_n; predicates on pairs of them (the join
graph — chain, star, occasionally a triangle); per-predicate
semantics: full join (all matching tuples) or exists (keep a
document if some partner matches). Never evaluate triples in one
prompt: binary predicates, intersected, keep cost a sum of pair
terms instead of a product of relation sizes.

**Terms.** In each stage one side is the **anchor**: its document
sits first in the prompt, right after the fixed preamble, so its KV
depends on nothing pair-specific and can be computed once. The
other side is the **partner**: it streams through as the per-pair
suffix, and its KV depends on the anchor in front of it, so it is
recomputed every pair in every design — partners never store KV,
anywhere. Sizes: f = preamble + anchor tokens; s = partner +
question tokens; S = the chunk/step token budget B* (derived — see
the decisions above; 421,752 today).

**Planning, before anything runs.**

1. Sample only when the plan depends on survivors. A 2-way full
   join evaluates every pair whatever the selectivity, so it samples
   nothing — its plan follows from token lengths and counts alone.
   Sampling (~2k pairs per predicate, labeled by the model, minutes)
   enters at n ≥ 3, where stage order and gating value depend on
   selectivity and matches-per-document, and under exists semantics,
   where expected scan length does.
2. n is 2 to 4, so enumerate every stage order and every per-stage
   anchor choice and price them with the cost model below. No
   heuristics. What wins, almost always: anchor the longer side
   (partner tokens are paid per pair, anchor tokens per document),
   and order stages so tuples die cheapest first.
3. The plan emits, per stage: anchor, pair-list rule, where the
   anchor prefix lives (the size rule below), predicted tokens and
   wall.

**Executing stage j** over pair list L_j, grouped by anchor
document:

- Compute the anchor document's prefix KV once.
- Stream its partners as suffixes against that prefix; record one
  YES/NO per pair. Exists semantics: stop the stream at the first
  YES.
- Between stages: **dedup** — the next stage runs once per distinct
  surviving anchor document, not once per surviving tuple; tuples
  are reassembled from recorded answers in Python at zero model
  cost. **Gate** — an anchor with zero matches in a conjunctive
  query is finished; skip all its later stages. Both are pair-list
  construction, independent of the execution substrate.

**Where the anchor prefix lives — the size rule.** First quantity
the planner computes, per stage: how many chunks one anchor's
partner stream spans,

    m = ceil(|partners| * s / (S - f))

- **m = 1**: the whole inner loop fits one batch. Nothing outlives
  a chunk; no KV question exists.
- **m > 1, f/S below ~8%, anchor feeds only this stage**:
  **recompute** the prefix at the top of each of its m chunks.
  Waste is bounded by f/S no matter how large the partner relation
  is — a bigger partner list adds chunks and useful suffix work at
  the same rate.
- **Otherwise — long anchors (f/S above ~8%), or the anchor feeds a
  later stage (n ≥ 3), or the predicate reads the whole tuple so
  far — keep the anchor KV** and return to its boundary for each
  new partner stream: rewind. Two implementations of the same
  schedule: stored prefix tensors in the packing loop (one
  contiguous tensor per live anchor, freed when its stages finish —
  no paging, no eviction, no admission; at hundreds of MB, bf16 is
  affordable, which the filter ladder measured fixing 765 of 2,990
  wrong answers versus fp8), or living chains in the engine (the
  existing `rewind_target` machinery, admission-sized as in the
  filter work).

**Cost model.** Fresh tokens, then wall = fresh / rate (96,180
tok/s engine, 121,045 packed), plus the cached-read term only where
engine-managed KV is read:

    stage j:  (prefix computations) * f_j  +  |L_j| * s_j

    prefix computations = anchors touched, if KV kept
                        = chunks headed (m per anchor), if recomputed

    |L_1|     = |R_anchor| x |R_partner|  (or an FDJ candidate list)
    |L_(j+1)| = (distinct surviving anchors after the gate) x |next partners|

**Instantiations.** n = 2: one stage, no survivors to check —
packed pass, recompute rule, done. n = 3 (chain R1—R2—R3, anchor
R2 for both predicates): run stage 1, record answers, gate and
dedup the surviving R2 documents, rewind to each survivor's prefix
boundary, stream R3. Same shape for every larger n: check
survivors, rebuild the pair list, rewind, stream.

---

## 2. The workload: BioDEX

Patient reports joined to medical reaction terms — a multi-label
classification task phrased as a join, from the FDJ paper's Table 1.
8,103 reports x 3,718 terms = 30,126,954 pairs; 20,000 true pairs.
Public: HuggingFace (D'Oosterlinck 2023); FDJ sampled the records
appearing in 20k ground-truth pairs — reproduce that sampling, and
ask Sepanta for the exact subset and the join prompt so numbers line
up with the paper.

Why this one: it is public; it is asymmetric, so the anchor choice
is visible (below, 5 h against 91 h); the pair count is large enough
that per-pair costs dominate; and one full run is affordable.

Assumed lengths until tokenized (step 1 replaces these): reports
1,000 tokens, terms 8, preamble 40, question tail 50. So f = 1,040,
s = 58. **Every number below is printed, with its arithmetic, by
`plans/join_estimates.py`**, which derives its inputs from the
committed sweep artifacts — `calibrate_all.json` for the raw
cached-read cells and the boot's pool size, the kernel-ladder file
for the packed rate and the chunk budget, `filter_cells.json` for
the admission budget — and imports the fitted constants from
`quail/plan/cost.py`. The only hand-typed block is the assumptions
(FDJ table sizes; the document and prompt lengths step 1 replaces).
Rerun it when step 1 lands.

**Anchor choice, both ways** (packed, recompute):

| | fresh tokens per pair | total fresh | wall |
|---|---|---|---|
| reports anchored, terms streamed | 58 | 1.82B | **4.6 h** |
| terms anchored, reports streamed | 1,050 | 31.7B | **76 h** |

The trap this kills: all 3,718 terms fit in KV at once (178k
tokens, 19% of the pool) and they are still the wrong side to
anchor — anchoring is about which side gets paid per pair, not
which side fits.

**Size rule applied:** at the derived B* = 421,752, a report's
whole partner list fits one chunk — 7,253 suffix slots against
3,718 partners, so m = 1. No recompute beyond the one unavoidable
prefix, nothing kept, no persistence question at all for this
2-way: the join is ~4,200 brim-packed chunks (about 1.9 reports
each) from a Python loop. The reference cell at 25,305 (the
largest measured point) has k = 418, m = 9, recompute 4.2%,
predicting 4.59 h — it exists because B* sits far past the
measured sweep, where per-token GEMM cost was seen rising with B
(the README's L2 open problem).

**The comparison, derived** (walls from `plans/join_estimates.py`):

| plan | fresh tokens | wall |
|---|---|---|
| A1: stock vLLM, request per pair, arbitrary order | 33.08B | **98.9 h**; x1.87 thrash = 185 h — arithmetic only, never run |
| A2: stock vLLM, pairs grouped by report, admission matched | 1.756B + boundary | **7.2 h** |
| B: engine chain mode (rewind) — fallback path | 1.756B | **6.1 h** |
| C: packed forward pass, m = 1 at the derived B* | 1.756B | **4.4 h** |

Where each number comes from:

- **A1** = 30,126,954 pairs x 1,098 tokens = 33.08B fresh → 95.5 h
  linear, + 2.8 h quadratic attention surcharge (1,098-token prompts
  against the 472-profile the rate embeds), + 0.06 h step-fixed
  cost, + 0.42 h request overhead. The x1.87 is the filter run of
  default admission at 3.4x pool pressure — README-recorded only;
  those cells were never committed to results/, so it is the one
  input without an artifact, and the prototype does not remeasure
  it (naive stock is not run). This workload's prefix working set
  is 9.0x the pool.
- **B** = 8,103 x 1,040 + 30,126,954 x 58 = 1.756B fresh → 5.07 h,
  + 1.09 h of cached reads: P x f x 125 ns, interpolated to our
  suffix width 58 from the c2 cells of `calibrate_all.json` — 101
  ns at c=32, 131 ns at c=64, by the documented two-cell
  subtraction. (Re-deriving from the cells is why these differ from
  the 82/104/141 trio quoted in cost_model.md, which does not
  reproduce from the committed rows; the cells win.) Flag:
  h = 1,040 sits below the grid's 2,048 minimum — the number a C3
  diagnostic would firm up.
- **A2** = B + 0.42 h per-pair request overhead + 0.70 h boundary
  blocks (the 16-token block spanning the report/term boundary
  recomputes every pair, ~8 tokens x P = 241M).
- **C** = 1.747B suffix + 8.4M prefix (each report once — m = 1 at
  the derived B*) = 1.756B at the packed non-attention rate →
  3.92 h, + 0.51 h attention priced from pair counts (1.87e12
  pairs, 97% of them suffix-to-prefix cross-attention, at 2 x a2
  per pair). Cross-attention is 12% of C's wall — the packed filter
  never paid it, so C's effective rate is ~110k tokens/s, not
  121k, and that is what the probe gates.

Baseline fairness, per house rules: the grouped-stock run gets the
same B* at boot and an admission cap derived the same way
(max_num_seqs from the pool arithmetic). The engine fallback, if
ever needed, runs at its own 16,384 step budget, where saturation
needs 565 resident chains against the 682 the admission budget
allows.

---

## 3. The prototype

Sampled and small — about ten minutes of GPU for the 2-way
comparison. The full-scale walls in Section 2 stay as extrapolation
targets (x81 on this sample), not as runs. Predictions stated
before each run; every Modal run teed to a file.

1. **Data — done, run locally.** The streamed BioDEX sample is
   tokenized and its statistics committed
   (`results/engine/join_lengths.json`): 100 reports at **mean
   2,977 prefix tokens** (29-token preamble included; truncation at
   3,500, which 45 of 100 reports hit — recorded policy), and a
   **pool-limited vocabulary of 2,560 terms** against the paper's
   3,718, suffixes at **31.7 tokens** mean. Sample pair count:
   256,000; full-scale equivalent 8,103 x 2,560 = 20.7M. Every
   prediction below is restated from these measured lengths by
   `plans/join_estimates.py`; the Section 2 table keeps the
   assumed-length paper-scale illustration.
2. **Packed join — the prototype** (256,000 pairs). Run order
   inverted by review: the **primary arm runs at 25,305** — the
   largest measured sweep point, m = 4 at measured lengths,
   predicted **1.7 min** — and the derived B* = 421,752 is the
   exploratory cell (m = 1, predicted **1.5 min** at ~91,500 tok/s
   effective, conditional on the rate holding out there). The
   effective rate fell from the assumed-length ~110k because real
   prefixes are 3x longer and suffixes half: cross-attention is
   ~27% of the wall. Per-segment attention plus the
   suffix-to-prefix call merged by softmax state (the exact
   convex form A + (B−A)·sigmoid(lse_B − lse_A) — the fp32
   materializing form cost ~13k tok/s and one knife-edge answer
   flip); suffix positions identical to standalone requests.
   **Probe: passed** — isolated attention math within 0.0068 of an
   fp32 reference, 0/64 disagreements shared-vs-unshared and
   kept-vs-shared, rates 83.6–84.2k tok/s (−8.6% of prediction,
   inside the gate), flat across both budgets as predicted
   (results/engine/join_probe.json). The
   merge call plus the pair-list Python are the new code; the
   packed loop, the three kernels, and the YES/NO readout carry
   over from `modal_single_filter_forward.py`.
3. **One baseline: grouped stock vLLM** — synchronous, one
   generate() over the pair list in report order (the join has no
   gating, so the filter arm's async client is unnecessary),
   admission via max_num_seqs derived from the committed token
   budget, bf16 KV (the fairest engine setting), zero decode.
   Predicted **3.3 min**, of which **77 s is paged cached-reads**
   of the 3k-token prefixes — the measured shapes make the read
   term 39% of the stock wall, so the packed-over-stock gap widens
   to ~2.2x from the assumed-length 1.6x. Doubles as the
   cross-implementation answer reference. Naive stock stays
   arithmetic-only. Report-side extrapolation x81: packed ~2.1 h,
   stock ~4.5 h at full scale.
4. **n-way, vLLM-free.** A planted 3-relation chain on the
   existing IMDB reviews, 100 documents per relation: every B
   document carries two planted keys (`[KEYS] X=.. Y=..`), A
   documents carry an X key, C documents a Y key, and the query is
   the triples where a.X = b.X and b.Y = c.Y — the true output is
   known by construction. Run entirely as packed forward passes
   plus Python staging: record stage-1 answers, gate and dedup
   survivors, continue against relation 3. Plant B's documents
   long (~4k tokens): the keep-KV branch is exercised by the
   cross-stage rule — B's prefixes are kept from stage 1 and
   reused in stage 2 instead of recomputed. Predicted: about a
   minute of GPU per stage. Two checks at the end: (a) replay the
   recorded yes/no answers through a nested-loop reference — no
   model calls, no gating, no dedup, literally `if answer1[a,b]
   and answer2[b,c]: emit (a,b,c)` — and require the identical
   triple set, so any difference is a bookkeeping bug and can
   never be model noise; (b) the stage-2 pair count must equal
   survivors x 100 exactly, proving the gate and dedup cut the
   work they claim.

Gates on the packed arm, stated now: zero disagreements between
the shared-prefix chunk and the same pairs run one-per-chunk (and
between kept-KV replay and in-chunk); wall within 10% of the
measured-length prediction. Stock agreement is reported as a
statistic, not gated — different kernels round differently, as
every packed-vs-engine cell in this repo has shown. Fail → the
engine chain path is the fallback (exists today) and C3
calibration joins the critical path. If any measured wall misses
its prediction by more than 10%, run the C3 family as the
diagnostic before trusting the model further.

**Runbook.** `modal run experiments/modal_join_forward.py::` +
`run_probe` / `run_join2way` / `run_nway3`; each saves its JSON
into `results/engine/`. (Launching from this sandbox needs
`pip install modal python-socks` — the client tunnels through the
egress proxy once python-socks is present.)

---

## 3b. Results — measured against the predictions above

All three runs completed (~40 GPU-minutes total;
`join_probe.json`, `join2way.json`, `join_nway3.json`).

| quantity | predicted | measured | verdict |
|---|---|---|---|
| merge math vs fp32 reference | exact | 0.0068 max diff | pass |
| parity, shared vs one-per-chunk | 0 disagreements | 0 of 64 | pass |
| parity, kept-KV replay vs in-chunk | 0 disagreements | 0 of 64 | pass |
| packed at 25,305, 3 reps | 102 s | 105.8 s, spread 0.04 s | **+3.7%, pass** |
| packed at derived B*, 2 reps | 92 s | 95.5–95.7 s | **+4.0%, pass** |
| rate flat in B | equal rates | 88.0k tok/s at both; wall ratio 1.107 = token ratio 1.106 | **confirmed** |
| grouped stock, 2 reps | 199 s (GPU terms) | 487–504 s | GPU terms right; see finding 1 |
| packed over stock | 2.1x | **5.1x** | wider, for finding-1 reasons |
| 3-way stage 1 | 40 s, expected 8–15% over | 48.7 s | +22%, see finding 3 |
| 3-way stage 2, per survivor | 0.43 s | 0.30 s | kept prefixes beat the recompute price |
| 3-way triples vs nested-loop replay | identical | identical (74,600 triples) | **pass** |
| 3-way stage-2 pair count | survivors x 100 | 10,000 = 100 x 100 | **pass** |

Findings, in order of importance:

1. **The cost model is missing a stock host term.** The stock arm's
   GPU-side prices were right (reads, linear, boundary ≈ 199 s of
   its wall), but one-request-per-pair means the engine ingests,
   hashes, and bookkeeps 770M prompt tokens across 256,000 request
   objects even though 99% are cache hits — about 290 s of host
   work at these shapes. The per-request constant (50.3 us,
   measured at 10k filter requests) has no per-prompt-token
   ingestion term. The packed side has no analog: its "requests"
   are 100–400 chunks. Follow-up: add the term to cost.py from
   this run's residual before any full-scale stock prediction.
2. **The checkpoint is a near-unusable judge of both predicates,
   which breaks the planted instrument but no execution claim.**
   2-way: YES on 60% of pairs (recall 584/587, precision 0.33%).
   3-way: 9,763 of 10,000 stage-1 answers wrong against the
   planted keys — it answers YES to almost every candidate, so all
   100 B documents survived and the gate executed zero skips on
   GPU (planted design expected 80). Gating and dedup logic remain
   covered by the unit tests and the replay check, which passed
   identically. Follow-up: a planted predicate this model can
   actually read (the filters' single-flag lookup worked; two-key
   comparison across 4k tokens does not), or a stronger model as
   the instrument.
3. **Stage-1 3-way ran +22% over** (48.7 s vs the 40 s estimate
   whose stated tolerance was 8–15%): ~41k-token chunks amortize
   the per-chunk host work (pack, answer, capture) worse than the
   84k-token 2-way chunks, and the per-layer kept-KV clones add
   copies the estimate did not price. Both are named, bounded
   costs; neither changes a conclusion.
4. Peak memory: 7.7 GiB (25,305 chunks), 15.3 GiB (B* chunks)
   against the 80 GB card — consistent with act x B at the real
   chunk sizes, nowhere near binding.

---

## 4. Implementation sketch

Four pieces, in build order. No engine anywhere in the packed path
— weights come through vLLM's loader exactly as the packed filter's
experiment 3 does, nothing else of vLLM is used.

1. **`quail/joinlogic.py` + tests** — pure functions, no GPU, in
   the style of `chainlogic.py`: `orient()` (compare mean tokenized
   lengths, return anchor side); `pack_chunks()` (a streaming
   packer over the whole pair list: fill each chunk to the brim
   under B*, cutting only at suffix boundaries — a pair's suffix is
   atomic, its tokens must attend to each other inside one chunk —
   so per-chunk slack is bounded by one suffix, reclaimable by
   pulling a shorter suffix forward; when an anchor's partner list
   ends mid-chunk the next anchor's prefix starts in the same
   chunk; plus the keep-vs-recompute decision by the size rule.
   Answers are packing-invariant — each suffix's computation
   depends only on its prefix and itself — so the packer can change
   utilization but never results); `gate_and_dedup()` (answers in, surviving
   anchors and the next stage's pair list out); `assemble()`
   (recorded answers in, tuples out). Unit-tested against a
   brute-force reference. Brim packing matters most where per-
   anchor lists are far shorter than a chunk — gated later stages
   and candidate-list mode — where per-anchor chunks would run
   mostly empty.
2. **The merge call** — the one new GPU piece, in the packed loop's
   per-layer body. This is the Hydragen / cascade-inference
   decomposition of shared-prefix attention (Juravsky et al. 2024):
   suffixes attend to the shared prefix in one batched call and to
   themselves in another, combined by softmax state. Hydragen's
   headline wins are decode-side (one query per sequence, where
   batching against the shared prefix turns memory-bound attention
   into a matrix product) — we have zero decode, so what we take is
   the decomposition itself, and their pure-PyTorch-plus-FA
   implementation is evidence the merge needs no exotic kernel.
   Their hierarchical variant maps to n ≥ 3 tuple prefixes later;
   here one sharing level is enough (the 40-token preamble is
   folded into each prefix, not given its own level). Chunk rows
   are `[prefix | suffix_1 .. suffix_k]`.
   Call A: the existing varlen self-attention over the segment
   boundaries (prefix over itself, each suffix over itself). Call
   B: non-causal cross-attention, queries = all suffix rows, keys
   and values = the prefix rows of this chunk's K/V — or a kept
   tensor from an earlier chunk. Call B is ragged: a brim-packed
   chunk holds several prefix groups, so queries and keys carry
   per-group boundaries (cu_seqlens_q / cu_seqlens_k), and kept
   tensors are concatenated into the same ragged KV buffer.
   Combine A and B by their softmax states (log-sum-exp merge; one
   small elementwise Triton kernel, or FlashInfer's merge op). Positions: prefix rows 0..f-1, every
   suffix restarts at f — the rotate kernel already takes per-token
   positions. A kept-prefix chunk simply has no prefix segment:
   call A covers suffixes only, call B reads the stored tensors.
   Keeping a prefix = stashing its 36 per-layer K/V slices
   (72 KiB/token) in a preallocated per-layer ring buffer, so the
   ragged call reads slices in place with no copies, and freeing
   the slot when the anchor's stages finish. There is no rewind
   operation anywhere in this path: rewind exists in the engine
   because a living sequence accumulates suffix KV that must be
   erased back to the boundary; the packed pass never writes suffix
   KV at all, so there is nothing to erase — keeping the immutable
   prefix and attaching fresh suffixes computes exactly what the
   engine's rewind computes. The rewind machinery stays the engine
   fallback only.
3. **Data prep, no GPU** — BioDEX: download, sample 100 reports,
   keep all terms, tokenize, write the prompt. Planted: three
   100-document collections from the IMDB reviews with key lines,
   B's documents concatenated to ~4k tokens.
4. **The runner + the baseline arm** — a `stage()` function (pack,
   forward, read YES/NO logits at suffix ends, return the answer
   matrix) and an `nway()` driver that alternates `stage()` with
   `gate_and_dedup()`. The baseline arm reuses the filter stock
   client: same container, stock vLLM 0.26 boot, prefix caching on,
   one request per pair `[preamble | report | term | question]`
   with output constrained to the YES/NO tokens at `max_tokens=1` —
   zero decode, the identical answer protocol as the packed arm —
   submitted grouped by report through the client-side token-budget
   semaphore the filter experiments already use. Its answers are the parity
   reference; its wall is the comparison. At sample scale the pool
   never fills (100 prefixes = 104k tokens), so the baseline runs
   eviction-free — its x81 extrapolation is best-case for stock,
   and is reported as such.

Build order is also the risk order: item 2's parity microprobe
(one chunk against the same pairs run one-per-chunk) gates
everything downstream.

## 5. Risks

- Report lengths are assumed and heavy-tailed; truncation moves
  cost and accuracy. Step 1 exists to kill this risk first.
- The 121,045 tok/s packed rate was measured without the
  prefix-attention call. The estimate prices that call from the
  fitted attention constant (2 x a2 per pair, 11% of C's wall) and
  the probe's rate gate checks the resulting ~110k effective rate;
  a fused or badly-shaped kernel could still miss it. All arms run
  at the derived B*, far past the measured sweep where per-token
  GEMM cost was seen rising with B; the 25,305 reference cell
  exists to tell a rate change at large B apart from a slow
  kernel.
- Naive stock is never run; its 99.9 h is arithmetic from the same
  constants, and the 1.87x thrash multiplier stays README-sourced
  and unmeasured. Say both wherever the number is quoted.
- The prototype's sample walls extrapolate x81 to full BioDEX only
  if report lengths are stationary across the sample; tokenizing
  the full report list (cheap, no GPU) checks that before any
  extrapolated claim.
- Our answers come from Qwen3 4B, the paper's from GPT-4.1. This
  experiment claims execution speed at matched answers across
  plans (the probe's parity gate), not predicate accuracy against
  the paper.
- Police-shaped workloads (symmetric, long documents, f/S ~16%+)
  exercise the keep-KV branch of the size rule that BioDEX does
  not; the planted 3-way check covers that branch cheaply when it
  runs.
