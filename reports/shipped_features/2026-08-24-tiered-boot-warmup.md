# Boot warmup split into a compile pass (once ever) and a touch pass

## What changed

`warm_kernels` (quail/executor/loop.py) is now a policy wrapper over
two passes:

- `compile_kernels`: the DeepGEMM sweep over vLLM's config-boundary
  generator up to the full chunk budget, plus every attention-path
  shape as real forward passes (budget chunk + tiny-chunk ladder
  under both attention modes, one join chunk, the fast path). Runs
  once per (software stack, GPU, model, budget); a marker file next
  to the kernel caches on the `quail-kernel-cache` volume records
  that it ran.
- `touch_kernels`: the forward passes only. Runs on every container
  whose marker matches, so cached kernel binaries load into the
  process at boot (milliseconds each) instead of inside the first
  measured query.

Warmup inputs are synthetic token ids: no kernel keys on token
values, so nothing in warmup depends on the query any more. The old
signature (corpus documents + questions) is gone from all callers
(worker, calibration, GPU cells, ablations).

## Why

The audit (report 2026-08-24-boot-tiered-warmup.md, removed 2026-08-29 - git history) showed the
per-container sweep only does useful work the first time a
configuration is ever seen - compiled kernels persist on the volume.
Splitting the passes lets the compile side afford provable coverage
(the generator's full list to the budget, closing the +0.46 s
residual above 4,096 tokens) while the per-container side shrinks to
the touch cost.

## Before/after

Measured 2026-08-24 (4B, H100 SXM; details in
reports/2026-08-24-boot-tiered-warmup.md, removed 2026-08-29 - git history):

- Per-container warmup: 3.62 s touch pass, against 3.7-4.9 s for
  the swept warmup it replaces - with compile coverage extended
  from stepped guesses to the generator's complete list up to the
  budget.
- One-time compile pass: 103.8 s, once per (stack, GPU, model,
  budget), persisted on the kernel-cache volume with a marker.
- Query unchanged: best fast-path wall 28.42 s vs 28.30 s
  committed, 0 wrong answers; stock vLLM same day 34.18 s
  (separate requests per document), so 20% faster than stock.
