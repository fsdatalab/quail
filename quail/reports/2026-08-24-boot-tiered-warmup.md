# Tiered boot warmup: compile once ever, touch per container

## What this is

An audit of boot-time kernel warming found that the per-container
warmup sweep was doing work that only ever needed to happen once.
This report describes the audit, the new tiered boot, and the
verification run: boot phase timings, a py-spy profile of where boot
time goes, and the standard filter query re-measured against the
committed numbers and against stock vLLM.

Code: `quail/executor/loop.py` (warmup section). Verification cell:
`tests/gpu/boot_profile.py`.

## The audit

Three kernel families run in the engine, with different compile
behavior:

- DeepGEMM (the fp8 matmuls): JIT-compiles one kernel per
  configuration; the configuration is picked from the token count
  and the weight shape. ~2.4 s of nvcc per new configuration.
  Compiled files persist on the `quail-kernel-cache` Modal volume,
  so each configuration compiles once ever, across all containers.
- Triton kernels (merge_quant, kv_scatter, norm, silu, qk):
  specialize on model constants plus a small bounded set of argument
  variants. A handful of compiles, also cached once ever.
- FlashAttention-3 and cuBLAS: precompiled binaries. Nothing to
  compile.

No kernel keys on token values, document content, or which query is
running. Warmup therefore needs nothing from the query, and the
compile work is per (software stack, GPU, model, budget) - not per
container, and not per query.

What a container still pays after everything is compiled: loading
each cached kernel binary into the process, milliseconds per
configuration, on its first call. If nothing runs at boot, those
loads land inside the first measured query.

The previous code ran the full GEMM sweep in every cold container.
With a warm cache that sweep compiles nothing; it only re-launches
every matmul (3.7-4.9 s measured in the committed
`results/boot_profile.json`). It also could not tell whether the
cache was complete, so it could neither skip itself nor safely
extend to full coverage (the generator's complete size list up to
the budget costs ~20x the swept compute - fine once, not per boot).

## The change

- **Compile pass** (`compile_kernels`), once per (stack, GPU, model,
  budget): DeepGEMM sweep over vLLM's config-boundary generator up
  to the full budget - provable coverage, closing the known +0.46 s
  residual above 4,096 tokens - plus every attention-path shape as
  real forward passes (budget-sized chunk and tiny-chunk ladder
  under both attention modes, one join chunk, the fast path). A
  marker file next to the kernel caches records the identity; one
  volume commit persists kernels and marker together.
- **Touch pass** (`touch_kernels`), every container whose marker
  matches: the forward passes only, no sweep. Each hot kernel runs
  once so cached binaries load at boot.
- Warmup inputs are synthetic token ids; the payload coupling
  (first alias's documents, first filter question) is gone.

## Prediction (stated before the run)

- Compile boot: minutes once ever on this volume (the generator
  sizes above 4,096 have never been compiled here).
- Touch boot `warm_kernels_s`: 2-4 s (three budget-sized chunks at
  ~1 s each plus two tiny-chunk ladders), against 3.7-4.9 s for the
  swept warmup. Cold `boot_s` stays load_model-dominated (28-38 s).
- Query: best fast-path wall within 3% of the committed 28.30 s
  (`results/m1_filter1.json`), 0 wrong answers, faster than stock
  vLLM's 33.41 s (`results/baseline_filter1.json`, submission:
  separate requests per document).
- py-spy: boot time concentrated in weight loading; warmup time in
  GPU synchronization.

## Measured

TO FILL AFTER RUN.

Figure: plots/boot_tiered.png

## What the numbers mean

TO FILL AFTER RUN.

## Data

- Committed summary: `results/boot_tiered.json`,
  `results/baseline_filter1.json` (refreshed same day).
- Raw: `/results/boot/boot_tiered.json` and the py-spy profiles
  `/results/boot/pyspy_*.speedscope.json` on the `quail-results`
  volume (load them at speedscope.app).
- Before-numbers: `results/boot_profile.json` (swept warmup, kept
  as the historical reference), `results/m1_filter1.json`.
