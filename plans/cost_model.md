# The measured cost model

This document walks through the step cost model of the FPS draft
(`plans/fps_draft.tex`; Section 4, Eq. 3 and Eq. 6), component by
component: how each term was measured, what the fitted constant is,
and how accurate the fit is. Every number comes from committed result
files in `results/engine/`, produced this week on one H100 SXM
(Qwen3 4B, fp8 weights and fp8 KV, vllm 0.26.0, CUDA 13 image). The
draft's Remark 1 says its constants are analytical placeholders until
calibrated values replace them; these are the calibrated values.

Scope, fixed on purpose: filter queries only, one GPU, contexts to
16,384 tokens, calibration family C3 skipped (Section 5 below says
what that forbids). The draft's sharding-invariance claims are
untested here; nothing below contradicts them, but nothing measures
them either.

## 0. Measurement discipline

- Every engine cell submits exactly one step of work and verifies it:
  the step trace must show one step of the requested shape, and every
  request must report exactly the designed cached-token count. Rows
  that fail are marked invalid; rows whose five measured repeats spread
  past 10 percent are retried once and flagged.
- The fits read one file, `calibrate_all.json`: 100 cells plus the
  transfer probes, all from a single container. Rows from different
  containers are never mixed, because host speed varies up to 45
  percent across containers (Section 10 measures this directly).
- The landing run came back with zero invalid and zero unstable cells,
  and its drift pair (the same reference cell run first and last)
  agreed within 1.05 percent.
- Two engine-boot defaults had to be overridden before any measurement
  was honest, and both are recorded in every run's boot row:
  `async_scheduling=False` (the vllm 0.26 default moves the GPU wait
  outside the window the step trace times; the first run "measured" a
  16K prefill at 19.5 ms, five times below the hardware floor) and
  `enforce_eager=True` (the boot's single 8,192-token CUDA graph
  padded every smaller step up to 8,192 tokens, so h=512, 1,024, and
  2,048 all cost the same 84 ms). The consequence of the eager boot for
  the fixed step cost is stated in Section 3.

The workload behind the corpus-level numbers: 10,000 IMDB reviews
(3,203,917 tokens; mean 320, max 2,947), five yes/no filters with
planted answers, selectivities (0.9, 0.9, 0.9, 0.8, 0.8). One cached
token costs kappa = 73,728 bytes at fp8 (the draft's m = 72 KiB).

![Document lengths](../results/plots/doc_lengths_histogram.png)

## 1. Prefill: T_pre(h) = a1·h + a2·h²  (draft Eq. 6)

**How measured.** The alpha family: one request of h fresh tokens per
step, h swept 512 to 16,384 (11 cells), plus the drift pair at 4,096.
Nonnegative least squares on the medians fits intercept, a1, a2.

**Constants.**

| constant | value | prediction before the run | verdict |
|---|---|---|---|
| a1 | 9.20 µs/token | 10.4, accept 8.5–12.5 | in range |
| a2 | 4.93e-10 s/token² | ~4.2e-10 | +17 percent |
| bend a1/a2 | 18,640 tokens | — | — |
| T_pre(16,384) | 290.8 ms | ~286 ms (173 if linear) | in range |

**Accuracy.** Quadratic fit error 4.9 percent over the alpha cells;
forcing a linear-only model gives 18.1 percent. The h² term is real
and is attention: Section 8 confirms it with an independent
instrument.

![Prefill vs length](../results/plots/calib_alpha.png)

## 2. Step composition: f_B(B) and beta_N  (draft Eq. 3)

**How measured.** The C1 family: N requests of c fresh tokens each,
c in {64, 256, 512}, N doubling until B = N·c reaches 32,768
(25 cells, all valid). f_B is monotone piecewise-linear over B-knots;
beta_N is the per-request slope.

**Constants.** beta_N = 50.3 µs per request. Above B = 4,096 the
series is linear at ~10.7 µs/token; below B = 2,048 every cell sits
on a flat floor (18.6–19.5 ms in the landing container) that the
next section explains.

**Gate.** The B=64 and B=8,192 cells separate by 4.7x. Equal times
were the CUDA-graph padding signature; the eager boot removed it.

![Step time vs B](../results/plots/calib_fb.png)

## 3. The fixed step cost b0, and which engine it describes

b0 fitted 16.79 ms. The band predicted before the run was 1.5–3.5 ms,
so this is a miss, and the mechanism is known exactly: the prediction
described the production boot, where CUDA graphs replay the forward
pass as one pre-recorded program, while calibration must run eager
(Section 0), so the host launches each of the ~600 kernels itself,
every step, at a cost of ~17–19 ms regardless of step size. The floor
is host-bound, not GPU-bound: a slower container floored at 26–29 ms
while its GPU-bound large cells matched this container within 2
percent.

Consequence for use: the size-dependent terms (a1, a2, f_B, f_P,
beta_N) transfer to the production engine; the intercept does not.
The production fixed step cost remains the separately measured
STEP_FIXED_S = 2.94 ms (batch-size sweep, graphs on), with the knee
near 283 tokens.

![Batch sweep and the production step model](../results/plots/batchsweep_regression.png)

## 4. Cached reads: f_P(P), t_read, and epsilon

**How measured.** The C2 family: N requests, each a c-token suffix
over an h-token cached document, c in {16, 32, 64}, h to 16,384,
N to 64 (54 cells, all valid — every request reported exactly h
cached tokens). f_P is monotone piecewise-linear over attention
pairs P. t_read is the fitted f_P slope between the two reference
cells (c=32, N=32, h=8,192 vs 16,384), times c.

**Constants and the draft's largest correction.**

| quantity | measured | draft assumed | ratio |
|---|---|---|---|
| t_read at c=32 | 56.8 ns/token fitted; 101 ns raw slope | ~22 ns (m / HBM bandwidth) | 2.6–4.6x |
| epsilon = t_read/a1 | 6.2e-3 | ~1.5e-3 (Prop. 2 discussion) | 4x |
| effective read bandwidth | ~0.7 TB/s at c=32 | 3.35 TB/s | 21 percent of peak |

The read price is not a per-byte constant: the raw h-slope is 82, 104,
and 141 ns per cached token at c = 16, 32, 64. The cost tracks
query-KV pairs, not bytes — paged attention reads 16-token pages
against narrow query tiles and cannot amortize the fetches. This is
also why the fitted single number (56.8 ns, a c=32 reference on the
shared f_P curve) sits below the raw c=32 slope (101 ns): the fit's
own cross-check ratio is 0.56, outside its 20-percent trust band, and
the constant must be read as a reference point, not a universal rate.

What survives in the draft: Proposition 2's structure. Reading a
cached token is still ~160x cheaper than recomputing it (1/epsilon).
What must change: every number derived from epsilon = 1.5e-3. At
z = 5 filters, the cohorting slack 1 + z·epsilon is 1.031, not 1.008,
and the worked example "one cached pass over a 4K document costs
0.12 ms" becomes ~0.41 ms.

**Accuracy.** The reference cell c2_c32_h16384_n32 measured 67.2 ms
against a 30 ms prediction; the miss decomposes exactly into the
eager floor (Section 3) plus the real read price above. The scaling
is internally consistent three ways: linear in N (86–106 ns/token
increments), linear in h, and reproduced across two containers
(104 vs 101 ns).

![Cached evaluation vs P](../results/plots/calib_fp.png)

## 5. Identifiability: the missing R term  (draft Prop. 4)

The draft proves that with a shared suffix length c, the pair count
obeys P = (c/m)·R + c(1−c)/2·N exactly, so a resident-bytes term
theta_R cannot be identified from the designs this workload produces.
The fitting code enforces the proposition: asked for an R term
without C3 cells, it refuses with that equation in the error message,
and a detector recognizes genuine C3 pairs (matched P within 2
percent, suffix lengths 4x apart, long side ≥ 128 tokens) — the plain
C2 grid never qualifies. C3 was skipped on purpose; the model
therefore carries the single attention term f_P, and Section 4's
c-dependence is the visible cost of that compression.

## 6. Host time: the draft's form fails, one term fixes it

**How measured.** Scheduler plus bookkeeping time (`sched_ms +
update_ms`) is recorded separately from GPU time for every cell, so
this model's failure cannot contaminate any other constant.

The draft's form T_host = h0 + hN·N + hA·A fits at **528 percent**
mean relative error. The failure is structural: at fixed N=32 and
fixed A=64 fresh blocks, host time still rises 5.4 → 38.4 ms as
cached length grows 2,048 → 16,384. The scheduler rebuilds per-step
block tables spanning each request's whole context, and neither N nor
A can see that. Adding a resident-blocks term R (the whole context in
16-token blocks):

| constant | value |
|---|---|
| h0 | 0 |
| hN | 24.3 µs/request |
| hA | 0.49 µs/allocated block |
| hR | 1.12 µs/resident block |
| fit error | 14.9 percent (was 528) |

The draft's T_host equation should gain the hR·R term. Note this R is
identifiable for the *host* model from the existing C2 grid (h varies
at fixed N and A); the GPU-side R of Section 5 is the one that needs
C3. The two statements are consistent.

## 7. Additivity and imbalance: C4 and C5  (draft Remark 1)

The draft's near-optimality proof assumes step cost is additive
across requests. C4 measures it directly: mixed steps (fresh 512-token
prefills plus cached c=32 suffixes over 8K documents), entirely held
out of the fit, are predicted with **7.1 percent** error — inside the
10 percent the design allows. C5 holds B, P, N fixed and varies only
the per-request cached-length spread (uniform / mild / extreme /
one-hot over 131,072 total tokens): the four cells agree within
**1.3 percent**, against a 5 percent gate. No systematic residual;
no correction terms enter; the envelope of Proposition 2 stands.

## 8. Whole-model validation

Fit on 74 cells; validate on 22 held-out cells (all of C4 plus every
fifth C1/C2 cell by name). Held-out error **5.9 percent**; pairwise
ranking accuracy — the property the scheduler actually consumes —
**96.5 percent**. By regime: pure prefill 6.4, cached 5.1, mixed 7.1
percent.

![Predicted vs measured, every valid cell](../results/plots/calib_validation.png)

An independent instrument confirms the model's attention story: a
torch profiler sweep over single-document prefills classifies GPU
kernel time by class, and the measured attention and GEMM shares land
on the curves the fitted a1/a2 predict, crossing near **9,800
tokens** — inside the 16K ceiling. The profiler's own per-kernel
slope for the FlashAttention forward is 4.4e-10 s/token² against the
engine-fitted 4.93e-10. (This sweep also exposed a classification
bug: on Hopper the FlashAttention mainloop is a
`cutlass::device_kernel`, so every earlier profile had filed it under
GEMMs; both instruments now match attention patterns first, and the
corrected mix at B = 25,305 is GEMMs 50.9, quantize 18.3, norm 12.2,
attention 11.8, elementwise 6.7 percent of kernel time, GPU 99.5
percent busy.)

![Attention vs GEMM share by length](../results/plots/attnshare_profile.png)

![Corrected per-token budget and ncu verdict](../results/plots/phi_budget.png)

Where the speed-of-light goes, per component: dense per-token work
runs at 40 percent of the 275k tokens/s ceiling (the GEMM kernels
themselves at 92–93 percent of peak while resident — the gap is the
mix, not the kernels); attention compute at ~30 percent; cached
reads at 16–27 percent of memory bandwidth; pinned PCIe at 87
percent of spec.

![Roofline](../results/plots/roofline_4b.png)

## 9. Transfers and the offload crossover  (draft Eq. 8, Prop. 5)

**How measured.** The C6 probes, no engine: 4 GiB GPU-host copies
both directions, pinned and unpinned; 16 GiB container-disk write and
read with the page cache dropped; 4 GiB on the network volume.

| tier | rate | per-token m/beta | crossover h* |
|---|---|---|---|
| PCIe pinned (h2d / d2h) | 55.5 / 55.3 GB/s | 1.33 µs | 0 — transfer dominates every length |
| PCIe unpinned | 10.9 / 11.9 GB/s | 6.8 µs | 0 — still under a1 |
| container disk read | 3.88 GB/s | 19.0 µs | 19,900 tokens — past the 16K ceiling, recompute wins in scope |
| volume read | 3.24 GB/s | 22.7 µs | 27,500 tokens |

This is Proposition 5 operating on measured constants: a1 = 9.2 µs
exceeds m/beta for both PCIe tiers, so its first branch (offload
weakly dominates at every length) holds on real numbers; the disk
tiers fall to the second branch with h* beyond the measured envelope.
Pinned rates reproduced a months-old probe within 0.1 percent;
unpinned within 12.7 percent. Disk write rate varied 2.6–4.9 GB/s
across containers; only the read side prices a restore.

![Persist threshold](../results/plots/persist_threshold.png)

![Restore vs recompute](../results/plots/restore_vs_recompute.png)

## 10. The per-query residue, measured host-controlled

The draft's T_ser and the estimator's c0 absorb per-query software
cost. The shipped value was 3.2 s, derived by subtracting predicted
token work from measured walls across containers — and walls spread
up to 45 percent across containers, so that derivation booked host
variance as overhead. The anchor protocol removes the confound: in
one container, first measure that container's own serving rate with a
full stage-1 pass through the same submission machinery (96,804
tokens/s; the fleet anchor is 97,000), then run the query, then
subtract.

| arm | c0 reps | median |
|---|---|---|
| planned (rewind) | 0.244, 0.026, 0.013 s | **0.026 s** |
| stock pipelining | 0.659, 0.354, 0.694 s | 0.659 s |

The planned executor's query-level residue is 26 milliseconds. The
constant is corrected accordingly.

## 11. End to end: predicted against measured walls

The estimator prices the full query (T_in + T_quest + T_reread + c0)
from the constants above; the measured walls come from two containers
(the anchor run and the banked comparison). Stock's read multiplier
is an input (the estimator does not predict prefix-cache eviction);
rewind's token count is the model's own.

| walls from | arm | measured mean | predicted (fleet constants) | error |
|---|---|---|---|---|
| anchor container | rewind | 39.71 s | 41.87 s | +5.4 percent |
| anchor container | stock | 41.22 s | 40.62 s | −1.5 percent |
| banked container | rewind | 39.84 s | 41.87 s | +5.1 percent |
| banked container | stock | 42.92 s | 40.62 s | −5.4 percent |

At the anchor container's own rate and measured c0, the rewind
prediction is 41.93 s (+5.6 percent) — so the rewind error is
structural, not host speed: the model charges 852,889 question tokens
(46-token stage-1 question, 13-token survival-thinned tails over a
33-token preamble) while the step trace measured 631,171. The
executor keeps more of the question resident than the preamble
accounting assumes. That one over-charge is the entire rewind error;
the banked-stock −5.4 percent is the known cross-container host
spread.

For reference, the measured comparison itself:

![Rewind vs stock](../results/plots/rewind_vs_stock_4b.png)

![Filter timeline](../results/plots/filter_timeline_detail.png)

![Rewind mechanism](../results/plots/rewind_schematic.png)

## 12. What the measurements change in the draft

1. **epsilon**: 1.5e-3 → **6.2e-3**. Every derived number moves:
   the z=5 cohorting slack is 3.1 percent, not 0.75; the 4K cached
   pass costs ~0.41 ms, not 0.12. The qualitative claim (reads are
   orders cheaper than prefill) survives at 160x.
2. **t_read is shape-dependent** (82–141 ns/token across c=16–64),
   so epsilon is a c=32 reference, not a constant of the hardware.
   The draft should state the reference.
3. **T_host needs the resident-blocks term** (Section 6): the stated
   h0 + hN·N + hA·A form fits at 528 percent error; with hR·R it
   fits at 14.9.
4. **b0 is boot-dependent**: 2.94 ms with CUDA graphs (production),
   16.8 ms eager (calibration). The draft should name which engine
   its per-step overhead describes.
5. **Prop. 5's PCIe branch is the measured case**: a1 > m/beta for
   pinned and unpinned PCIe, so offload dominates recomputation at
   every length on those tiers; disk crossovers land beyond the 16K
   envelope.
6. **The additivity and imbalance assumptions hold** within 7.1 and
   1.3 percent (C4, C5) — Remark 1's escape hatch is not needed at
   this scale.
7. **Per-query residue**: 26 ms for the planned executor, measured
   host-controlled; the earlier 3.2 s was container variance.
8. **Known model conservatism**: the estimator over-predicts the
   planned arm's wall by ~5.5 percent via question-token accounting
   (Section 11); fixing it needs either a measured keep-resident
   fraction or a longer effective preamble.

## Reproduction

| artifact | command |
|---|---|
| calibration rows | `modal run experiments/modal_calibrate.py --families all` |
| fits | `python -m quail.plan.fit --rows results/engine/calibrate_all.json --out results/engine/cost_model_fit.json` |
| figures | `python plots/make_figures.py calib attnshare_profile phi_budget` |
| attention share sweep | `modal run experiments/modal_profiling.py::attnshare_main` |
| c0 anchor | `modal run experiments/modal_filters.py::c0_anchor` |
| end-to-end check | `python -m quail.plan.validate --anchor results/engine/c0_anchor.json --banked results/engine/filter_cells.json --out results/engine/makespan_check.json` |

Constants live in `quail/plan/cost.py`; every result file's provenance
is in `results/engine/README.md`.
