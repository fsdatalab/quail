# Warmup: tiny-chunk ladder, then provable coverage below 4,096 tokens

## What changed

Two changes to `warm_kernels` (quail/executor/loop.py), shipped the
same day:

1. A tiny-chunk ladder (`TINY_WARM_TOKENS`: 64-2048): real forward
   passes at trailing-chunk sizes, under both attention modes, with
   arena writes on.
2. The bare-matmul sweep's sizes below 4,096 tokens now come from
   vLLM's own DeepGEMM warmup generator
   (`_generate_optimal_warmup_m_values`), which mirrors the
   library's C++ config heuristic and emits every token count at
   which the chosen kernel configuration can change. The old guessed
   256-token steps remain only as a fallback if the import ever
   breaks. Above 4,096 the stepped grid (1,024 then 2,048 steps) is
   unchanged.

The diagnosis and validation cell is `ablations/dg_buckets.py`.

## Why

A gated multi-stage run's trailing chunks are 66-523 tokens, sized
by which documents pass the gates - unknowable at boot. On a kernel
cache that had not seen those sizes, each one stalled the measured
run: cache-directory snapshots attribute the stalls to nvcc compiles
of the vendored DeepGEMM matmul kernels (vLLM 0.26 vendors the
library; ten kernel.cu builds at ~2.4 s each across three novel
sizes), plus two small Triton fused kernels. Guessed warmup steps
cannot close this: the config heuristic's boundaries are irregular
(warming 128 and 256 covered a 198-token chunk; 165 and 423 stayed
cold). The generator enumerates the boundaries instead of guessing.

## Measured

All on 4B, BIO-F3, one container per run; raw records on the
quail-results volume (`packing_sweep_qwen3-4b-fp8_coldA/coldB.json`,
logs `results/dg_buckets.log`, `dg_validate.log`):

| Configuration (cold cache) | tail chunks (423/198/165 tok) | query GPU |
|---|---|---|
| no ladder, guessed steps | 12.9 s / 9.3 s / 9.5 s | 40.7 s |
| ladder, guessed steps | 3.1 s / 14 ms / 5.4 s | 17.6 s |
| ladder + generator sizes | 12 ms / 42 ms / 48 ms | - |

The final row's run created ZERO new kernel-cache files during the
measured query - on a fully cold cache, a gated chain now runs with
no mid-run compile at all. Every tail chunk sits at the per-chunk
launch floor.

The warm sweep grew from 332 to 1,024 items, but the added items are
small matmuls; the cold sweep (every compile included) took ~4 min
against ~11 min for the old 332-item sweep. The sweep loop now
reuses one input buffer per linear instead of allocating a random
tensor per item. Boot cost on the shared (warm) volume: 4B boots
measured 78-94 s with the new sweep against 59-62 s before it - but
per-container weight-load speed varies by tens of seconds (one 32B
container loaded checkpoint shards at 2.6 min each), so the sweep's
own share of that delta is bounded by, not equal to, the difference.
First boot after this change pays the new configurations once into
the shared cache (+33 s observed at 4B) and commits them.

Known residual, unchanged: above 4,096 tokens a chunk size between
grid steps can still compile once per software stack (observed once,
+0.46 s); chunk sizes there are packer-controlled, and closing it
with the generator's full list to the 110k budget would multiply
warm-boot compute ~20x.
