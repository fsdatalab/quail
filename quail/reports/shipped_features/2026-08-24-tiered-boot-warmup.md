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

The audit (report 2026-08-24-boot-tiered-warmup.md) showed the
per-container sweep only does useful work the first time a
configuration is ever seen - compiled kernels persist on the volume.
Splitting the passes lets the compile side afford provable coverage
(the generator's full list to the budget, closing the +0.46 s
residual above 4,096 tokens) while the per-container side shrinks to
the touch cost.

## Before/after

TO FILL AFTER RUN: warm_kernels_s per cold container, boot_s, query
wall vs committed and vs stock.
