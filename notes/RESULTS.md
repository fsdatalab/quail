# Solved schedules: first results (analytical model, no GPU runs)

All numbers are **model-optimum territory under τ₀** with conventions C1–C12
(`notes/PLAN.md`): nominal ceilings, R_A = R_D, S = 2 GB, fp8 KV, δ = 256 at
N=10k (δ=1 for exact runs), p_j = 50, the seeded 10k-doc IMDb workload
(Σd = 2.966M tokens). Full grid in `results/n10k_two_stage.csv`; canonical
manifests in `results/manifests/` (validated by the independent checker).

## 1. N = 10,000, n = 2: feasible schedules vs. resource lower bounds

Every constructed schedule lands within **0.07%** of its LB_res (most ≤0.01%,
many at exact equality ⇒ optimality certificates per Prop. 9.1). Because the
gaps are ~0, the numbers below are effectively the model optima, and the
policy comparison is decided by the ledger totals (eqs. 32–34).

Makespan in seconds, mean over 2 coupled scenarios:

| config | policy | s₁=0.10 | 0.25 | 0.50 | 0.75 | 0.90 |
|---|---|---|---|---|---|---|
| 4B / H100 | task | **12.12** | 13.80 | 16.56 | 19.17 | 20.98 |
| | pipe | 13.05 | **13.33** | **13.80** | **14.25** | **14.54** |
| | fullspec | 14.73 | 14.73 | 14.73 | 14.73 | 14.73 |
| 4B / L40S | task | **32.78** | 37.32 | 44.57 | 52.20 | 56.65 |
| | pipe | 35.22 | **35.99** | **37.23** | **38.51** | **39.29** |
| | fullspec | 39.77 | 39.77 | 39.77 | 39.77 | 39.77 |
| 32B / H100 | task | **104.2** | 118.2 | 141.8 | 165.4 | 179.3 |
| | pipe | 111.8 | **114.2** | **118.2** | **122.1** | **124.5** |
| | fullspec | 126.1 | 126.1 | 126.1 | 126.1 | 126.1 |
| 32B / L40S | task | **280.8** | 318.4 | 383.5 | 445.7 | 485.1 |
| | pipe | 301.8 | **308.2** | **319.3** | **329.8** | **336.3** |
| | fullspec | 340.5 | 340.5 | 340.5 | 340.5 | 340.5 |

Findings (this workload, this cost model):

- **Task-first ↔ pipeline break-even is s₁ ≈ 0.20** on every config (ledger
  crossover s* = N·p₁ / (Σd − N·p₂) = 0.203; the batch-level effects move it
  by <0.01). Below it, re-prefilling the few survivors is cheaper than paying
  N·p₁ prompt branches; above it, pipeline wins.
- **Full speculation never strictly wins under τ₀** — as s₁→1 pipeline
  converges to fullspec from below. Outcome barriers are nearly free when
  batches hold 10⁵–10⁶ tokens: the extra (1−s₁)·N·p₂ speculative prompt work
  is never repaid. The same holds even at 3.3% KV residency (32B/L40S):
  outcome-conditioned retention keeps recompute at ~0.
- Attention ≤2.5% of makespan at these document lengths; everything is dense-
  compute-bound, weight reads ≤1.2 s even at 70 batches.

## 2. n = 4, per-stage s, lookahead k (32B/L40S tight, 4B/H100 loose)

τ₀ seconds (single coupled scenario, validated):

| s/stage | config | task | k=1 | k=2 | k=4 | batches k=1→k=4 |
|---|---|---|---|---|---|---|
| 0.50 | 32B/L40S | 473.8 | **334.8** | 361.6 | 426.5 | 70 → 53 |
| 0.80 | 32B/L40S | 753.3 | **381.7** | 395.7 | 426.5 | 110 → 53 |
| 0.95 | 32B/L40S | 946.8 | **414.4** | 418.4 | 426.5 | 138 → 53 |
| 0.95 | 4B/H100 | 41.0 | **17.9** | 18.1 | 18.5 | 16 → 5 |

- Strict pipelining (k=1) dominates for n=4 too; k=2 closes to within 1% at
  s=0.95 but does not cross. Task-first degrades sharply with n (document
  re-prefill at every reached stage: 2.3× pipeline at s=0.95).
- The batch-count column shows speculation's real currency: k=4 needs 53
  batches where k=1 needs 138. Under a **calibrated** cost with per-batch
  overhead β₀, speculation crosses over when β₀·ΔB exceeds the wasted-prompt
  dense time — e.g. at s=0.95/L40S, β₀ ≳ (426.5−414.4)/(138−53) ≈ 140 ms per
  batch. That is the experiment the calibrated layer should target.

## 3. Exact-DP enumeration boundary (sec. 10.5 step 3)

Toy-scale model, n=2, δ=1, binary evictions: offline Dijkstra solves N=4 with
d≈(7,6,6,4) in ~146 s (N=3 in <1 s); online SSP value iteration reaches N=3,
d≈4 (859 reachable states, ~1 s). Beyond that, exact enumeration is
impractical without action-space restriction — consistent with the strong
NP-hardness result; the N=10k results above instead rest on feasible
schedules + matching lower bounds.

## Caveats

- τ₀ omits per-batch software overhead by design; all "speculation never
  wins" statements are claims about the ideal model, labeled as such (the
  paper's three-layer discipline, sec. 5.5).
- W_mem/W_run are nominal (C5 placeholders), not yet measured from
  safetensors; LB and τ shift proportionally with W_run only through the
  ≤1.2 s weight-read term.
- Break-evens use the Δ-interval discipline of sec. 10.8: the task↔pipe sign
  change is bracketed by grid cells (0.10, 0.25); refine with a finer s grid
  if a tighter interval is needed.
