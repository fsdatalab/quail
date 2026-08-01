# Solved schedules: what the solvers actually show

All numbers are **analytical-model results under τ₀** with conventions C1–C12
(`notes/PLAN.md`): nominal ceilings, R_A=R_D, S=2 GB, fp8 KV, p_j=50, the
seeded 10k-doc IMDb workload (Σd = 2.966M tokens, mean 297). Raw grids:
`results/n10k_two_stage.csv`, manifests in `results/manifests/`, small-N
study in `experiments/run_smallN.py`. Every schedule cited was replayed by
the independent validator.

The results split into three regimes, and being explicit about which regime a
number comes from is the whole story:

## 1. N = 10,000: the τ₀ problem is ledger-dominated (and that is the finding)

Feasible schedules for all four model×device configs, s₁ ∈ {0.1…0.9}, land
within **0.07% of LB_res** (most at equality ⇒ optimality certificates,
Prop. 9.1). Full table in the CSV; representative numbers (seconds, τ₀):

| config | policy | s₁=0.10 | 0.50 | 0.90 |
|---|---|---|---|---|
| 4B/H100 | task | **12.12** | 16.56 | 20.98 |
| | pipe | 13.05 | **13.80** | **14.54** |
| | fullspec | 14.73 | 14.73 | 14.73 |
| 32B/L40S | task | **280.8** | 383.5 | 485.1 |
| | pipe | 301.8 | **319.3** | **336.3** |
| | fullspec | 340.5 | 340.5 | 340.5 |

Read this correctly: the near-zero gaps do **not** mean the constructors are
clever — they mean that at this scale, under τ₀, batches hold 10⁵–10⁶ tokens,
dense compute binds in every batch, and *any* capacity-filling packing
achieves the resource bound. The schedule-construction layer moves < 0.1%.
Consequently:

- The certified optimum equals the token ledger (eqs. 32–34) evaluated on the
  realized d. Policy ranking at N=10k is decided by token totals alone.
- Task↔pipeline break-even: s₁* = N·p₁/(Σd − N·p₂) = **0.203** on every
  config (grid-bracketed in (0.10, 0.25); the batch layer shifts it < 0.01).
- Full speculation never wins at this scale: its extra (1−s₁)·N·p₂ prompt
  tokens are never repaid because outcome barriers are hidden behind other
  work (see §2) and retention never forces recompute even at 3.3% KV
  residency (32B/L40S) — outcome-conditioned retention with a one-batch
  retention span is enough.
- Attention ≤ 2.5% of makespan at these document lengths (grows as Σd² for
  length-scaled runs); weight reads ≤ 1.2 s even at 70 batches.

## 2. Small N: the exact DP earns its keep (real 4B/H100 numbers)

Exact offline DP (Dijkstra, atomic prefill, real doc lengths, s=0.5 coupled
outcomes) against the same constructors, τ₀ in ms:

| N | task (exact) | pipe (exact) | fullspec (exact) | pipe (constructor) |
|---|---|---|---|---|
| 2 | 2.44 | 2.17 | **2.09** | 2.80 |
| 3 | 3.92 | **3.12** | 3.30 | — |
| 4 | 5.98 | **4.52** | 4.70 | 5.05 |

Two certified findings:

- **Staggered pipelining hides outcome barriers.** The optimal N=3 pipeline
  is: batch 1 = docs {0,1} prefill + their F₁ branches; batch 2 = doc 2
  prefill + doc 2's F₁ + the F₂ branches of batch-1 survivors. The F₂ gate
  costs nothing because it overlaps a held-back document's prefill, so
  pipeline gets speculation's batch count without its wasted prompts. This
  kills the naive prediction that speculation wins whenever N·p₂ is below
  the weight-read floor (U* = R_D·W_run/(2P·BW) ≈ 295 tokens on H100):
  speculation wins only when there is no other work to hide behind — here
  N=2, and generally the tail of a query.
- **The constructors are ledger-optimal, not schedule-optimal.** They do not
  stagger, and are 12–24% above exact at N ≤ 8 (2.80 vs 2.17 at N=2; 5.05 vs
  4.52 at N=4), converging by N ≈ 16 where staggering stops mattering. At
  N=10k this gap is invisible (§1); any small-N or tail-of-query claims must
  use the exact DP or a stagger-aware constructor.

Exact-DP practical boundary (recorded per sec. 10.5 step 3): with atomic
prefill and no memory pressure, N=4 solves in seconds at real lengths; at
δ=1 token granularity with evictions, the toy study reached N=4 with
d≈(7,6,6,4) in ~146 s offline, N=3 online (859 states). Beyond that the
state space explodes — consistent with strong NP-hardness; large-N claims
rest on feasible schedules + matching lower bounds instead.

## 3. n = 4 lookahead (single coupled scenario, validated)

| s/stage | config | task | k=1 | k=2 | k=4 |
|---|---|---|---|---|---|
| 0.50 | 32B/L40S | 473.8 | **334.8** | 361.6 | 426.5 |
| 0.95 | 32B/L40S | 946.8 | **414.4** | 418.4 | 426.5 |
| 0.95 | 4B/H100 | 41.0 | **17.9** | 18.1 | 18.5 |

Strict pipelining dominates for n=4 at N=10k too (same §1 logic); task-first
degrades sharply with n (document re-prefilled at every reached stage).
Speculation's real currency is batch count: 138 (k=1) vs 53 (k=4) batches at
s=0.95 on 32B/L40S. Under a **calibrated** cost with per-batch overhead β₀,
the k=4 block wins once β₀ ≳ (426.5−414.4)/(138−53) ≈ 140 ms per batch —
that, plus the small-N/tail regime of §2, is where speculation lives, and it
is the first thing the calibrated layer should measure.

## Caveats

- τ₀ omits per-batch software overhead by design; every "X never wins" above
  is a claim about the ideal model at the stated scale, per the paper's
  three-layer discipline (sec. 5.5).
- W_mem/W_run are nominal placeholders (C5), not yet measured from
  safetensors; they enter through the ≤1.2 s weight-read term and U*.
- Small-N exact numbers are per-realization (one coupled X per N), not Monte
  Carlo averages; the N=2 vs N=3 flip point moves with the realization.
- Attention uses the corrected width n_q·d_h (C1) — with the paper's
  eq. (20) as written, all attention terms would be 1.6× smaller.
