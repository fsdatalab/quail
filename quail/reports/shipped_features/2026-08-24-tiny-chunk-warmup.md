# Tiny-chunk warmup ladder in warm_kernels

## What changed

`warm_kernels` (quail/executor/loop.py) now also runs a ladder of
tiny single-document chunks - 64, 128, 256, 512, 1024, and 2048
tokens (`TINY_WARM_TOKENS`) - as real forward passes, under both
attention modes, with arena writes on. The packing_sweep cell gained
`--cold-cache` (kernel caches in a container-local directory instead
of the shared volume) and `--no-warm-tiny`, the two switches the
validation below used.

## Why

The existing warmup runs a bare-matmul size sweep and full-size warm
chunks. It never builds an actual tiny chunk. A gated multi-stage
run's trailing chunks are 66-523 tokens, and on a kernel cache that
has not seen those shapes, each one paid a one-time compile stall in
the middle of the measured run: 0.7-0.9 s per chunk at 4B and up to
9.3 s at 32B in the 2026-08-24 packing sweep, +30-38% on the whole
query once.

## Before/after

Measured on two cold caches (4B, BIO-F3, one container each; raw:
quail-results volume `ablations/packing_sweep_qwen3-4b-fp8_coldA.json`
/ `_coldB.json`, logs `results/packing_coldA_4b.log` / `_coldB_`):

- Without the ladder: tail chunks of 423 / 198 / 165 tokens stalled
  12.9 / 9.3 / 9.5 s; query GPU time 40.7 s.
- With the ladder: the same chunks ran 3.1 s / 14 ms / 5.4 s; query
  GPU time 17.6 s - 23.1 s less.

The ladder warms fixed sizes and real tail sizes depend on gating,
so a size between ladder points can still compile once (423 and 165
did, partially). Every such compile is once-ever per software stack:
the caches ride the shared kernel-cache volume, and on the warm
volume the same tail chunks measure 14-27 ms (the per-chunk launch
floor). Warm-cache boot cost of the ladder is ~0.5 s (12 chunks at
the 20-30 ms floor). Large chunks were already covered: both cold
runs measured the 9 big chunks at 9.1 s total, matching the warm
run.
