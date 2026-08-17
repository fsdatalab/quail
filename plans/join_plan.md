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
  pool. The chunk budget B* is a parameter with a derived cap, not
  a constant: B* ≤ (memory x 0.92 − weights − kept prefixes x
  kappa) / activation bytes per token ≈ 843k tokens, floored at
  ~4,096 where throughput flattens. Default is 25,305, the largest
  measured sweep point; bigger values cut prefix recomputes but
  sit past the measured range, so they are probe cells. Chunks are
  packed to the brim across anchors — an anchor's list ending
  mid-chunk is followed by the next anchor's prefix, no padding.
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
question tokens; S = the chunk/step token budget B* (a derived
parameter — see the decisions above; 25,305 by default).

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

**Size rule applied:** k = (25,305 − 1,040)/58 = 418 partners per
chunk; m = ceil(3,718/418) = 9 chunks per report; f/S = 4.1%. So:
recompute, keep nothing, the KV pool stays empty. The whole join is
9 x 8,103 = 72,927 chunks of prefix groups + suffixes, packed to
the brim, built from a Python loop. At B* = 85,000 the same rule
gives k = 1,447, m = 3, recompute 1.4%, and 4.47 h — conditional
on the packed rate holding at that B, which the sweep does not
cover (per-token GEMM cost rose with B there; the README's L2 open
problem): a probe cell, not an assumption.

**The comparison, derived** (walls from `plans/join_estimates.py`):

| plan | fresh tokens | wall |
|---|---|---|
| A1: stock vLLM, request per pair, arbitrary order | 33.08B | **99.9 h**; x1.87 thrash = 186 h — arithmetic only, never run |
| A2: stock vLLM, pairs grouped by report, admission matched | 1.756B + boundary | **7.2 h** |
| B: engine chain mode (rewind) — fallback path | 1.756B | **6.1 h** |
| C: packed forward pass, recompute | 1.823B | **4.6 h** |

Where each number comes from:

- **A1** = 30,126,954 pairs x 1,098 tokens = 33.08B fresh → 95.5 h
  linear, + 2.8 h quadratic attention surcharge (1,098-token prompts
  against the 472-profile the rate embeds), + 1.07 h step-fixed
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
- **C** = 1.747B suffix + 75.8M prefix recompute = 1.823B at the
  packed non-attention rate → 4.07 h, + 0.52 h attention priced
  from pair counts (1.91e12 pairs, 95% of them suffix-to-prefix
  cross-attention, at 2 x a2 per pair). Cross-attention is 11% of
  C's wall — the packed filter never paid it, so C's effective rate
  is ~110k tokens/s, not 121k, and that is what the probe gates.

Baseline fairness, per house rules: the grouped-stock run gets the
same memory budget and an admission cap derived the same way
(max_num_seqs from the pool arithmetic; the engine rows want the
step budget dropped toward 16k because the admission budget caps
resident chains at 682 against the 873 that two step budgets of
58-token suffixes would need).

---

## 3. The prototype

Sampled and small — about ten minutes of GPU for the 2-way
comparison. The full-scale walls in Section 2 stay as extrapolation
targets (x81 on this sample), not as runs. Predictions stated
before each run; every Modal run teed to a file.

1. **Data.** Fetch BioDEX, sample **100 reports and keep all 3,718
   terms**, write the join prompt, tokenize. Sample only the anchor
   side: with all terms kept, each report still spans 9 chunks, so
   prefix persistence is actually exercised — a 100 x 100 sample
   would fit each report's partners in one chunk and test nothing.
   Tokenizing replaces every assumed length; reports have heavy
   tails, so pick and record a truncation policy.
2. **Packed join — the prototype** (371,800 pairs, predicted
   **3.4 min** at the derived ~110k tok/s effective rate). Chunk =
   brim-packed prefix groups + suffixes (~418 per prefix at the
   default B*); per-segment attention plus the suffix-to-prefix
   call merged by softmax state; suffix positions identical to
   standalone requests. One extra cell at B* = 85,000 — bigger
   chunks cut the recompute to 1.4% and predict 4.47 h at full
   scale, conditional on the rate holding past the measured sweep. Existing code carries most of
   it — the packed loop, the three kernels, the chunk packer, and
   the YES/NO readout from `modal_single_filter_forward.py`; the
   new work is that merge call plus ~100 lines of pair-list Python.
3. **One baseline: grouped stock vLLM, admission on the client**
   (the filters' semaphore pattern), same sample, predicted
   **5.4 min**. It doubles as the answer-parity reference for the
   packed arm. Naive stock is cut: its 99.9 h is arithmetic, not an
   experiment worth buying.
4. **n-way, vLLM-free.** A planted 3-relation chain on the
   existing IMDB reviews (100 documents per relation, two group
   keys), run entirely as packed forward passes plus Python
   staging: record stage-1 answers, gate and dedup survivors,
   continue against relation 3. Plant one relation long (~4k
   tokens) so the keep-KV branch of the size rule runs, not just
   recompute. Predicted: about a minute of GPU per stage.

Gates on the packed arm, stated now: answers identical to the
stock arm's, wall within 10% of prediction. Fail → the engine
chain path is the fallback (exists today) and C3 calibration
(long-suffix cached reads) joins the critical path to price it.
If any measured wall misses its prediction by more than 10%, run
the C3 family as the diagnostic before trusting the model further.

---

## 4. Implementation sketch

Four pieces, in build order. No engine anywhere in the packed path
— weights come through vLLM's loader exactly as the packed filter's
experiment 3 does, nothing else of vLLM is used.

1. **`quail/joinlogic.py` + tests** — pure functions, no GPU, in
   the style of `chainlogic.py`: `orient()` (compare mean tokenized
   lengths, return anchor side); `pack_chunks()` (a streaming
   packer over the whole pair list: fill each chunk to the brim
   under B*, cut wherever the budget lands, and when an anchor's
   partner list ends mid-chunk start the next anchor's prefix in
   the same chunk — no padding; plus the keep-vs-recompute decision
   by the size rule); `gate_and_dedup()` (answers in, surviving
   anchors and the next stage's pair list out); `assemble()`
   (recorded answers in, tuples out). Unit-tested against a
   brute-force reference. Brim packing matters most where per-
   anchor lists are far shorter than a chunk — gated later stages
   and candidate-list mode — where per-anchor chunks would run
   mostly empty.
2. **The merge call** — the one new GPU piece, in the packed loop's
   per-layer body. Chunk rows are `[prefix | suffix_1 .. suffix_k]`.
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
   (72 KiB/token) and freeing them when its stages finish; stage 2
   against the same anchor is the same call B with a new suffix
   stream — that is the rewind, as tensors.
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
  a fused or badly-shaped kernel could still miss it. The
  B* = 85,000 cell additionally extends the rate past the measured
  sweep, where per-token GEMM cost was seen rising with B.
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
