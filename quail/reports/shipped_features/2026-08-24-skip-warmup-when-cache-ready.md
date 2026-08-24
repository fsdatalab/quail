# Skip kernel warmup when the volume already has cubins

Date: 2026-08-24.

## What changed

`warm_kernels` no-ops when both caches on the kernel volume already
have compiled binaries: DeepGEMM `kernel.cubin` files and Triton
`.so` / `.cubin` files. FlashAttention-3 is not on the volume; it
ships in the vLLM wheel.

A new container still loads weights (~30-38 s). It does not replay
the GEMM sweep or dummy filter/join forwards. The first real chunk
loads each cubin into this process (milliseconds per shape). A
missing shape still compiles on first use.

An empty cache (first container, or `boot_profile.py --cold-cache`)
still runs the full sweep and writes the volume.

## Why

The volume stores files, not a live CUDA context. The old warm-cache
path still launched 7,583 dummy GEMMs so those files would load
before the timed query. That was 4.46 s of work the query does not
need. The cold-cache path (209 s) stays for an empty volume.

## Before / after

- Warm volume, new container: `warm_kernels_s` 4.46 s mean → skip
  (~0 s). `load_model_s` unchanged.
- Same process: still the `_BOOTED` path (~0 s).
- Empty volume: still the compile sweep (measured 209.22 s at 4B).
