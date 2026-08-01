# Review of `paper.md` (Scheduling n-Stage AI Filters Under KV-Cache Constraints)

Process: five independent review lenses (math, hardware facts, scheduling
semantics, algorithms/complexity, reproducibility) swept the full source; 28
raw findings were deduplicated to 9, and each survivor was adversarially
verified by an agent instructed to refute it. Verdicts: 4 confirmed, 3
confirmed in weakened form, 2 refuted. The refuted ones are listed too —
they mark places where the paper is right and a careless reader (me
included) gets it wrong. Everything below was additionally cross-checked
against the working solver in this repo, which implements the model
end-to-end.

**Bottom line: the framework is sound and implementable — the solvers in
this repo reproduce its propositions on real data. But two technical errors
need fixing before any reported number is checkpoint-faithful, one proof and
one well-foundedness argument need repair, and two accounting definitions
need to be pinned before independent implementations can agree.**

## Confirmed errors

### E1 (critical). Attention FLOPs use the wrong width: `4·L·h` should be `4·L·n_q·d_h` — an exact 1.6× undercount for both committed models
Eq. (20) (`eq:attention-flops`), propagating into H(B) (eq. 25), τ₀, and the
attention term of LB_res (eq. 55). The paper's own derivation — one QK dot
product plus one AV accumulation per allowed pair — yields `4·n_q·d_h` FLOPs
per pair per layer (GQA shares KV *storage*, not query-side compute). The
substitution `n_q·d_h = h` is false for both committed checkpoints because
Qwen3 fixes `d_h = 128` independently of hidden width: 4B has `n_q·d_h =
4,096` vs `h = 2,560`; 32B has `8,192` vs `5,120` — 1.6× in both cases, an
internal inconsistency with the paper's own Table 3. The neighboring κ
formula (eq. 18) correctly uses `heads·d_h` (72/128 KiB verified), isolating
the bug to the `4Lh` factor. Materiality: at `L_ctx = 40,960` on 4B, correct
attention FLOPs (4.95e14) exceed dense FLOPs (2.95e14) — the 60% error
dominates exactly where the length-scaling plots live, and since A_T, A_P,
A_S differ per policy it can flip break-evens rather than rescale them.
**Fix:** define `h_attn = n_q·d_h`, use `F_A = 4·L·h_attn·A(B)` in eqs. (20),
(25), (55); add `n_q`/`h_attn` to Tables 2–3 and the recorded-configuration
list (line 984, which currently omits `n_q` entirely). The solver already
does this (convention C1).

### E2 (major). The strong NP-hardness proof does not cover the model's optimum: the 3-PARTITION reduction collapses under chunking
Prop. 7.2 asserts hardness of "the offline optimum", but the offline optimum
(what the δ=1 exact DP computes) ranges over Σ_chunk, and the paper itself
proves Σ_atomic ⊊ Σ_chunk (Prop. 5.4). On the constructed instances chunking
fills every batch to exactly B tokens by splitting the straddling document,
so OPT over Σ_chunk equals m·c whether or not a 3-partition exists — the
reduction proves hardness only for the atomic-restricted class. Also, the
proof's "positive constant cost per nonempty batch" and "zero attention
cost" are not primitives of τ₀; they hold only in the (unstated)
weight-read-bound regime. **Fix options:** (a) restate the proposition for
atomic prefill (δ ≥ max d_i) with the parameter regime stated; or (b) repair
the reduction so chunking cannot help — e.g. drive capacity through the
peak-memory constraint with fused branch evaluation: a document's *entire*
KV must be resident in the batch where its branches run regardless of how
prefill was chunked, so branch placement induces the partition. (b) restores
the stronger claim.

### E3 (critical). The online recurrence is not well-founded as stated
The paper fixes eviction/recomputation cycles for the *offline* search
(Dijkstra, line 873) but then says the online table is obtained by
"enumerating the same states and evaluating this recurrence" (eq. 54, line
889). The online state graph has the same positive-cost cycles (prefill →
evict → same state), so eq. (54) is a fixed-point equation, not a recursion;
no evaluation order, uniqueness, or convergence argument is given, and the
optimality proofs assume well-founded backward induction. **Fix:** state
that eq. (54) is a stochastic shortest-path problem: positive batch costs +
existence of a proper policy give a unique fixed point, computable by value
iteration (or a label-correcting SSP method). The solver implements exactly
this (convention C4; Gauss–Seidel VI over the reachable graph).

### E4 (major). LB_res uses schedule-dependent totals as if they were instance constants, and B_min has no derivation rule
Eq. (55) introduces U_tot, A_tot, B_KV,tot "for a fixed policy instance",
but under the paper's own rules these vary across schedules (recompute after
eviction, adaptive branch sets, chunking-dependent re-reads), and B_min is
"any valid lower bound" with no rule for obtaining one. As written, LB_res
is ill-defined and the equality certificate could compare a schedule against
a bound computed from a *different* schedule's totals. **Fix:** define each
total as the schedule-independent minimum over the policy class — the
recomputation-free, maximal-sharing ledger for the realized X (the realized
analogues of eqs. 32–37/43–45) — and give B_min a stated derivation, e.g.
from the peak-memory constraint: every new document token's KV occupies HBM
during its batch, so B_min = ⌈doc_tokens/((M−W_mem−S)/κ)⌉, plus any
configured cap. This is what `docengine/lb.py` implements (C8), and with it
the N=10k certificates are legitimate.

## Confirmed in weakened form

### P1 (major). K_W / K_R accounting is genuinely ambiguous for fused batches
Line 426's consumer-based K_W ("written … for use by a later chunk, branch,
or batch") never says whether "later branch" includes *same-batch* branches,
and the notation table gives a conflicting second definition ("new KV token
positions materialized"). K_R's "resident" is likewise unpinned for
in-batch-produced blocks. For a fully fused batch (doc + branches, nothing
persists), the ledger admits 0, κ·d, or 2κ·d per document — 147.5 MB per
2,000-token doc on 4B, so B_KV and the H(B) crossover move materially.
**Fix:** pick one convention and state it. The solver's C3: K_W = new doc
and prompt-block tokens (their KV has consumers beyond their producing
operation); branch tokens never; K_R = resident-at-batch-start blocks only,
deduplicated per physical block. The manifest then needs the op→physical-
block mapping so a validator can tell fused from reload (the repo's manifest
records per-op `read_blocks` for this reason — eq. 57 as written cannot).

### P2 (major, mostly resolved by careful reading). The eviction-timing "contradiction" between eqs. (30) and (54) is a readability defect, not two different optima
Line 514's joint (batch, eviction) action is explicitly the *offline*
action; the online protocol is defined sequentially in the very next
sentence, and eq. (54) encodes it. But eq. (30)'s single `min_a` invites the
precommitted-eviction misreading (min_E E[V] ≥ E[min_E V], strict under
memory pressure), and three of five reviewers initially misread it. **Fix:**
one sentence at eq. (30) defining the online action as the batch alone with
eviction chosen in the post-outcome state, or write eq. (30) in the two-stage
form of eq. (54) from the start.

### P3 (minor). The speed-of-light model is computable, but several run parameters still need pinning
The W_run "not computable" claim was refuted: line 386 defines it
operationally (recorded bytes of the repeated transformer blocks of the
pinned checkpoint — a static safetensors inspection, ≈P bytes within ~1% at
FP8). What genuinely remains unpinned and must be recorded per run: primary
R_A (2× swing on the attention term if calibrated ≈ R_D/2), primary q_KV
(2× on κ and capacity), chunk quantum δ for the 10k runs, the selectivity
grid and Monte Carlo R, exact prompts (p_j content, not just ≈50), the IMDb
pool ("the source dataset" is ambiguous among 25k/50k/100k; file IDs are
unique only per split/class), and tokenizer revision. The repo pins all of
these (C5–C11; workload builder records pool = 50k labeled, doc_id =
split/row, tokenizer sha `aeb13307…`, both FP8 repos ship byte-identical
tokenizers).

## Refuted on verification (the paper is right; worth keeping as reader traps)

- **R1. Offline oracle gate semantics.** Suspected ambiguity: may the
  clairvoyant scheduler co-batch F_j(i) and F_{j+1}(i)? Refuted — Sec 3.4
  pins gates at the execution-model level, before any information model:
  outcomes become visible only at batch boundaries, and a non-speculative
  batch cannot include F_{j+1}(i) while F_j(i) is unresolved *at the start of
  that batch*. This binds the oracle too; clairvoyance affects only packing,
  retention, and eviction. The solver enforces exactly this (C2).
- **R2. Speculative-block atomicity.** Suspected unrepresentable state for
  blocks spanning batches. Refuted — since outcomes are revealed at every
  batch boundary, a "block" split across batches is definitionally just
  sequential (pipeline) execution; speculation is intra-batch by
  construction, so S_S = (z, r, K) suffices.

## Minor notes

- Eq. (2) LaTeX bug: `X_{ik}quad` — missing backslash on `\quad`; the
  compiled equation renders spurious math-italic "quad" inside the survival
  definition.
- Line 93 says decisions are read "from the logits at the final prompt
  position", but under the task-first template `[F_j][D_i]` the final
  position is a document token. Reword to "final position of the serialized
  sequence".
- FP8 KV (q_KV = 1) silently drops quantization-scale storage; depending on
  granularity this is 0–3% of κ. Worth one sentence (ideal model excludes
  scales; calibrated layer measures them).
- T_init (eq. 28) has no cost model and unstated interaction with K_1 and
  LB_res. The solver sets T_init = 0 and prefills prompt blocks inside
  ordinary batches (C11), which the text should either adopt or price.

## What the implementation already shows about the fixed model

With E1/E3/E4 fixed as above (the repo's C1/C4/C8) the model is not just
consistent but *solvable*: exact DP optima on small instances reproduce
every proposition (chunking dominance with strictness, forced-speculation
outcome independence, the VoI inequality over all outcome scenarios), the
N=10k feasible schedules meet their lower bounds to <0.07% (ledger-dominated
regime), and the small-N exact optima exhibit staggered pipelining — outcome
gates hidden behind held-back document prefill — which is the scheduling
phenomenon the paper's framework exists to capture. See `notes/RESULTS.md`.
