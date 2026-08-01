# Solver plan: schedules for the n-stage AI-filter model

Goal: **solve the schedulers** of `paper.md` analytically — no GPU runs. Concretely:
(1) an exact DP (offline Dijkstra + online stochastic-shortest-path) that is provably
optimal on small instances; (2) strong feasible schedules plus resource lower bounds
for the real N=10,000 IMDb workload on all four model×device configs; (3) the
validator, manifests, and break-even sweeps the paper specifies. GPU measurement
(H100/L40S) is a later, separate phase.

## 0. Workload grounding (done)

`workloads/documents.parquet` (built by `scripts/build_workload.py`, seed 20260731):
10,000 reviews sampled without replacement from the 50k labeled stanfordnlp/imdb
pool, tokenized with the Qwen3 tokenizer (byte-identical for 4B-FP8 and 32B-FP8,
sha256 `aeb13307…`).

| quantity | pool (50k) | sample (10k) |
|---|---|---|
| mean / p50 / p90 / p99 / max tokens | 295 / 220 / 580 / 1153 / 3112 | 297 / 223 / 588 / 1142 / 2924 |
| Σd_i | 14.75M | **2.966M** |
| tokens per whitespace word | 1.276 | — |

Residency capacity (M − W_mem − 2 GB reserve, fp8 KV): 4B/H100 ≈ 1.00M tokens
(**34%** of corpus), 4B/L40S ≈ 570k (19%), 32B/H100 ≈ 343k (12%), 32B/L40S ≈ 99k
(**3.3%**). Retention is therefore genuinely capacity-bound at N=10k — the paper's
central tension is present in the real data.

Ideal-ledger rooflines (eqs 32–37, R_A=R_D, corrected attention width, expected
values, p1=p2=50): task-first beats pipeline only for s ≲ 0.2; full speculation is
within 7–15% of pipeline everywhere; attention ≤ 2.5% of dense time at these
lengths (e.g. 4B/H100 s=0.5: dense 14.4 s, attention 0.30 s, weight-reads ≤ 0.01 s).
So on this workload the schedulers are decided by **batch-level effects** — outcome
barriers, weight-read counts, and capacity-forced recomputation — which is exactly
what the DP and the N=10k constructions must capture. (For length-scaled
sensitivity runs the Σd² attention term grows quadratically and the balance shifts.)

## 1. Conventions (resolving the review's ambiguities)

Adopted defaults, all recorded in configs and revisitable (see `notes/REVIEW.md`
for why each is needed):

- **C1 — attention width.** F_A(B) = 4·L·(n_q·d_h)·A(B), not 4·L·h·A(B): Qwen3
  decouples attention width from hidden width (4B: 32×128 = 4096 vs h = 2560;
  32B: 64×128 = 8192 vs h = 5120). The paper's eq. (20) undercounts by 1.6×.
- **C2 — offline oracle keeps structural gates.** For non-speculative policies the
  offline optimum may not co-batch F_j(i) with work that requires F_j(i)'s result;
  clairvoyance affects only packing, retention, and eviction. This isolates the
  value of information from barrier removal (otherwise offline task-first morphs
  into zero-waste speculation and the VoI estimand conflates two effects). The
  gate-free oracle is a cheap sensitivity run later.
- **C3 — KV write/temp accounting.** K_W(B) = new positions whose KV survives their
  producing operation: (a) new document tokens that persist into K_{t+1}, and
  (b) document tokens consumed by a *different* operation in the same batch.
  Positions consumed only inside their own fused operation, and final prompt-branch
  positions with no future consumer, are not written. K_tmp(B) = all new document
  tokens + all prompt-branch tokens live during the batch (conservative peak).
- **C4 — algorithms.** Offline: on-demand Dijkstra (positive batch costs; handles
  evict/recompute cycles). Online: eq. (54) has cycles too (prefill → evict →
  re-prefill revisits a state), so it is solved as a stochastic shortest path by
  value iteration over the reachable state graph, not by naive recursion.
- **C5 — weights.** W_run = bytes of the L repeated transformer blocks in the FP8
  checkpoint (weights + FP8 scale tensors, read from the safetensors index);
  W_mem = full loaded checkpoint bytes. Nominal placeholders until measured:
  4B ≈ 3.6/4.5 GB, 32B ≈ 31.2/33.5 GB.
- **C6 — reserve.** Speed-of-light runs: S = 2 GB nominal, stated (S = 0 as a
  labeled variant). Calibrated S comes from engine measurement later.
- **C7 — rates.** Primary R_A = R_D (labeled optimistic); sensitivity R_A = R_D/2.
- **C8 — B_min.** Under C3 every new document token's KV exists in HBM during its
  batch, so per-batch document tokens ≤ free-KV capacity C and
  B_min = ceil(doc_tokens_total / C); if a new-token cap is configured, take the
  max with ceil(U_tot / cap). Without either, B_min = 1 (weak, reported as such).
- **C9 — chunk quantum.** δ = 1 (token-exact) for exact DP instances; δ = 256 for
  N=10k constructions, always reported (OPT_1 ≤ OPT_δ, eq. 50).
- **C10 — workload.** Pool = 50k labeled reviews, doc_id = split/row; 418 duplicate
  texts kept as distinct rows; no cross-document KV sharing for identical texts in
  v1. p_j = 50 exactly, all j. Primary KV dtype fp8 (q_KV=1); bf16 sensitivity.
- **C11 — T_init = 0.** Task prompts are prefilled inside the first batch that
  needs them; never double-counted.
- **C12 — online estimand.** OPT_on is the expected makespan of the optimal online
  policy computed from s; offline Monte Carlo uses coupled X matrices reused
  across policies (paper sec. 10.3).

## 2. Repository architecture

```
docengine/
  configs/models.py      # Qwen3-4B/32B: P, L, h, n_q, n_kv, d_h, L_ctx, κ(q_KV), W_mem, W_run
  configs/devices.py     # H100 SXM, L40S: M, BW, R_D, R_A
  workload.py            # load documents.parquet; length-scaling transforms
  outcomes.py            # coupled latent X matrices, seeds, selectivity grids
  costmodel.py           # a(c,q), U/A/K_R/K_W/K_tmp/M_peak per batch; D(B), H(B), τ0, τθ
  lb.py                  # eq. 55 resource LB with C8 B_min rules, per policy instance
  manifest.py            # eq. 57 schema, JSONL IO
  exact/
    state.py             # canonical state (z, r, K-residency, H) per policy family
    actions.py           # feasible-batch generator: task-first / pipeline / spec-k, δ, evictions
    offline.py           # on-demand Dijkstra + predecessor reconstruction → manifest
    online.py            # reachable-graph SSP value iteration (eq. 54)
  sched/
    taskfirst.py         # N=10k constructive scheduler + local search
    pipeline.py          # cohorted retention scheduler (knapsack eviction scoring)
    speculate.py         # fused-tree scheduler, per-stage lookahead k choice
    improve.py           # merge/split/move local search on manifests
  validator/check.py     # independent replay: feasibility, information, U/A/B_KV/M_peak/τ recompute
  experiments/           # runners → results/{manifests,bounds}; sweep drivers
  tests/                 # hand-enumerated instances, invariants, property tests
scripts/build_workload.py
workloads/documents.parquet
```

The validator shares **no** cost-model code with the solver (App. B contract): it
re-implements a(c,q), τ, memory, and information checks from the paper text alone,
and returns batch-indexed errors.

## 3. Exact DP (small instances)

**State.** Per document: (z_i, prefix kind ∈ {task-j, doc}, r_i, complete-flag);
plus pinned prompt set and revealed outcomes. Resident KV is exactly determined by
these per-doc residencies + pins, so K needs no separate arbitrary-set state —
this matches the paper's (z, r, K, H) without blowing up the encoding.

**Actions.** Per batch: per-doc increments Δ_i (multiples of δ), branch ops
(pipeline: F_{z_i} when r_i = d_i; speculation: contiguous k_i block), then
post-batch eviction keep-sets. Pruning that provably preserves optimality only:
lazy eviction (evict at a boundary only what the next chosen batch needs freed —
delaying eviction is WLOG only in this per-transition form), and never evict pinned
prompt blocks under task-first. Everything else is enumerated.

**Search.** Offline: Dijkstra, states materialized on demand, predecessor map →
schedule manifest. Online: BFS-enumerate the reachable graph, then Gauss–Seidel
value iteration to convergence (positive costs + a proper policy ⇒ unique fixed
point); the policy at each state is (argmin batch, then per-outcome argmin evict).

**Verification battery.**
- N=1..2, d ∈ {2..4}, tiny synthetic device (small M, W_run) hand-enumerated =
  DP output, all three policies, δ=1.
- Invariants: chunk attention identity (eq. 16); chunking dominance (eq. 31) as
  OPT_chunk ≤ OPT_atomic on every instance; speculation outcome-independence
  (eq. 41): identical value for all X; VoI inequality E[OPT_off] ≤ OPT_on (eq. 42)
  on every solvable instance; validator accepts every DP manifest.
- Scaling study: grow N and d until state count explodes; record the frontier
  (paper sec. 10.5 step 3) — expected practical limit ≈ N ≤ 6–8 with d_i ≤ ~12δ.
- Small-instance science: barrier cost (task-first vs pipeline vs spec as s and
  W_run vary), forced-eviction regimes (shrink M), lookahead k transitions for
  n = 3, 4.

## 4. N = 10,000 feasible schedules + bounds

Exact DP cannot reach N=10k (strongly NP-hard; sec. 7.6). Per the paper, report
constructive feasible schedules against LB_res, with the gap.

- **Task-first:** stage waves; within a wave, greedy length-aware packing of doc
  chunks into batches sized near the device crossover U* = R_D/(2·BW) new tokens
  (≈ 295 on H100, 424 on L40S — model-independent at FP8 since both terms scale
  with P) or larger; last partial chunks filled with δ-chunks of the next docs.
- **Pipeline:** cohorts sized to residency capacity: prefill cohort docs + F_1
  branches; retain survivors' doc KV (offline: exactly the X_i=1 docs; online:
  eviction scored by s·recompute_cost/bytes), run F_2 branches next batch, evict,
  next cohort. Piggyback next-cohort prefill chunks into branch-heavy batches to
  keep U near target.
- **Speculation:** per-doc fused trees (doc chunks + all/k prompt branches in one
  batch); no retention pressure; pure packing. Lookahead k per stage chosen by
  marginal break-even π_{j+k}·(recompute-or-barrier saving) vs (1−π_{j+k})·p
  waste, then validated by sweep.
- **Local search** on all three: merge/split batches, move chunks, retention swaps,
  accept on τ0 improvement; stop at local optimum. Report LB gap per config.
- **Sweeps:** s ∈ {0.05, 0.1, …, 0.95} (n=2), length scales ×{1, 2, 4, 8} (context
  rule enforced), n ∈ {2, 3, 4} with per-stage s grids for lookahead study,
  4 model×device configs, KV dtype {fp8, bf16}, R = 32 coupled outcome replications
  (means ± CI via eq. 59–60).

## 5. Deliverables & order

1. `costmodel` + `lb` + `manifest` + `validator` + tests.  (foundation)
2. Exact DP offline → verify battery → exact DP online.  (paper sec. 10.5 steps 1–3)
3. Small-instance result grid + plots (selectivity × policy × device).
4. N=10k constructors + local search + LB gaps; break-even maps.  (step 4)
5. Report tables/plots per paper sec. 10.7; every point labeled
   {LB, exact, feasible+gap}; manifests in `results/manifests/`.

Not in scope here: engine calibration (τθ fitting), vLLM/engine profiling, any
H100/L40S execution — the manifests are designed so those plug in later.
