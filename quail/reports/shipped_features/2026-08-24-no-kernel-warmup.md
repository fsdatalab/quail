# No kernel warmup at boot

Date: 2026-08-24.

## What changed

`warm_kernels` is deleted. The worker and the GPU cells no longer
run a GEMM sweep or dummy filter/join forwards at boot.

Compiled DeepGEMM cubins and Triton binaries stay on
`quail-kernel-cache`. The first real chunk loads each file into
this process. A missing shape compiles on first use.
FlashAttention-3 ships in the vLLM wheel.

## Why

The volume stores files, not a live CUDA context. Replaying 7,583
dummy GEMMs so those files would load before a timed query was
4.46 s on a warm volume and 209 s on an empty one. The query does
not need that. First-use load on a warm volume added about 9 s to
the first 10k five-filter query (chunk 0 was 8.74 s, compared with
~1 s when the cubins were already in process).

## Before / after

- Warm volume, new container: no `warm_kernels` phase. Boot is
  `load_model` + arena + pipeline.
- Same process: still the `_BOOTED` path (~0 s).
- Empty volume: first real chunk compiles the shapes it hits.
