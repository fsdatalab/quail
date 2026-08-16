# Joins: the general n-way algorithm, one experiment

Status: plan only, nothing implemented. This rewrites the earlier
multi-dataset draft after review: one workload, one experiment, and
the algorithm stated for general n even though only n = 2 runs. The
old draft (six datasets, seven phases) is in git history.

Decisions taken:

- **One workload: BioDEX.** Everything else from the FDJ paper
  (Products, Citations, Movies, Police, Categorize) stays out unless
  BioDEX leaves a question open.
- **The algorithm is solved for general n.** The experiment
  exercises n = 2.
- **A 2-way join runs as a packed forward pass.** No engine, no KV
  pool.
- **n ≥ 3 checks survivors between stages and rewinds**: continue
  from a kept anchor prefix and stream the next relation against it.
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
question tokens; S = the 25,305-token chunk/step budget.

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
9 x 8,103 = 72,927 chunks of one prefix + ~418 suffixes each, built
from a Python loop.

**The comparison, derived** (walls from `plans/join_estimates.py`):

| plan | fresh tokens | wall |
|---|---|---|
| A1: stock vLLM, request per pair, arbitrary order | 33.08B | **99.9 h**; x1.87 thrash = 186 h — priced by 5% sample, never run full |
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
  input without an artifact, and the 5% sample remeasures it. This
  workload's prefix working set is 9.0x the pool.
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

## 3. The runs

Three steps, each with its prediction stated before it runs; every
Modal run teed to a file.

1. **Data and prompts.** Fetch BioDEX, sample per the paper, write
   the join prompt, tokenize. Replaces every assumed length; restate
   the predictions with measured f, s, and length tails (reports
   have heavy tails — pick and record a truncation policy).
2. **Packed-join probe** (~one container-hour). One chunk shape:
   prefix + k suffixes, per-segment attention plus a
   prefix-attention call merged by softmax state; suffix positions
   identical to standalone requests. Gates, stated now: (a) answers
   identical to the same pairs run as per-pair engine requests;
   (b) chunk throughput within 10% of the derived effective rate,
   ~110k tok/s — the 121,045 filter rate minus the priced
   cross-attention. Fail (a) or (b) → the experiment runs on engine chain
   mode instead, which exists today, and C3 calibration (long-suffix
   cached reads) joins the critical path to price it.
3. **The experiment.** In order, one container: packed on a 5%
   pair sample as the confirming cell (predicted 14 min) — proceed
   only if within 10% of prediction; packed full (predicted 4.6 h);
   grouped-stock full (predicted 7.2 h); arbitrary-order stock on a
   5% pair sample, extrapolated (predicted 99.9 h full before
   thrash, 186 h at the README-recorded 1.87x). Sample
   pairs, not reports, so the baseline's prefix working set keeps
   its real 8.9x pool pressure and the thrash multiplier is
   honestly measured. Report all four with predictions alongside.

If the estimator misses any measured wall by more than 10%, run the
C3 calibration family as the diagnostic before trusting the model
further — joins are the workload that finally makes those cells
identifiable.

**n ≥ 3 check, later.** Not scheduled until the 2-way lands: a
planted 3-way on the existing IMDB reviews (two group keys per
document, chain graph), small, to validate the survivor gate, the
dedup bookkeeping, and the rewind path end to end. Execution
measurement only, like the flag filters.

---

## 4. Risks

- Report lengths are assumed and heavy-tailed; truncation moves
  cost and accuracy. Step 1 exists to kill this risk first.
- The 121,045 tok/s packed rate was measured without the
  prefix-attention call. The estimate prices that call from the
  fitted attention constant (2 x a2 per pair, 11% of C's wall) and
  the probe's rate gate checks the resulting ~110k effective rate;
  a fused or badly-shaped kernel could still miss it.
- The arbitrary-order stock number is an extrapolation from a 5%
  sample by design; say so wherever it is reported.
- Our answers come from Qwen3 4B, the paper's from GPT-4.1. This
  experiment claims execution speed at matched answers across
  plans (the probe's parity gate), not predicate accuracy against
  the paper.
- Police-shaped workloads (symmetric, long documents, f/S ~16%+)
  exercise the keep-KV branch of the size rule that BioDEX does
  not; the planted 3-way check covers that branch cheaply when it
  runs.
