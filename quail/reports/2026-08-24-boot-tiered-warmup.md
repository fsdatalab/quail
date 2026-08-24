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

All on Qwen3 4B fp8 / H100 SXM, vLLM 0.26.0, one container per
trial. Committed summaries: `results/boot_tiered.json` (py-spy
attached), `results/boot_tiered_nospy.json` (profiler off),
`results/baseline_filter1.json` (stock, refreshed same day).

Compile pass, once ever (force_compile, py-spy attached):

- warmup phase 103.8 s (the generator's full size list to the
  110,376-token budget plus all forward-pass shapes), inside the
  predicted "minutes once ever". This pass also wrote the marker;
  every following boot took the touch path.

Touch boots, profiler off (the clean walls; 2 trials):

- `warm_kernels_s` 3.44 / 3.81 s (mean 3.62 s) - inside the
  predicted 2-4 s, and below the swept warmup's 3.68-4.9 s in the
  committed `results/boot_profile.json`.
- `boot_s` 37.5 / 40.4 s, of which model loading is 32.6 / 34.8 s -
  load-dominated, as predicted.

Touch boots, py-spy attached (3 trials): `warm_kernels_s`
7.51-8.94 s (mean 8.25 s). The profiler itself adds ~4.6 s to the
warm phase at 100 Hz sampling, so its numbers locate time but do
not stand as walls.

Where boot time goes (py-spy MainThread profiles, on the
`quail-results` volume): model loading splits between network/file
reads and vLLM's per-tensor weight loaders, with a visible ~10 s of
Python module imports (torch/vLLM) inside the measured
load-model span; the warm phase is GPU-bound (cuda synchronize on
the warm chunks) plus DeepGEMM cache reads.

The query after a touch boot (10,000 IMDB documents, one filter
question, 2 trials x 2 reps x both paths, profiler off):

- best fast-path wall 28.42 s, best arena wall 28.75 s, 0 wrong
  answers - within 0.4% of the committed 28.30 / 28.65 s
  (`results/m1_filter1.json`).
- stock vLLM, same corpus, submission = separate requests per
  document with document-cap admission, rerun the same day: 34.18 /
  34.51 s. Quail's 28.42 s is 20% faster.

The gated five-filter query (the KV write path: multi-stage keeps
each document's KV, writes the shared question preamble, gates
between stages, and its trailing chunks are data-dependent tiny
shapes), run in a fresh container after a touch boot
(`results/m1_filter.json`, touch warmup 4.85 s):

- rep 0 wall 34.91 s, rep 1 wall 35.07 s, against the committed
  34.6 s reference; 4,645 survivors and 0 wrong of 40,052 answered,
  exactly matching the reference.
- rep 0 is the container's first run ever, and its per-chunk times
  match rep 1 to within a few milliseconds - every slow chunk is
  just a large chunk at the normal per-token rate. No compile stall
  anywhere, which is the tiny-chunk ladder plus generator coverage
  doing exactly what the compile pass promised.
- stock vLLM on the same five-filter workload, rerun the same day
  (submission: separate requests per (document, stage), pipelined,
  token-budget admission; `results/baseline_filter.json`): 43.25 /
  42.80 s, same 4,645 survivors. Quail's 34.91 s is 18% faster.

Figure: plots/boot_tiered.png

### 32B

The same verification on Qwen3 32B fp8, one H100
(`results/boot_tiered_32b.json`; no prior 32B numbers existed for
this query, so these rows establish the reference):

- Compile pass: 38.0 s once ever - shorter than 4B's 103.8 s
  because 32B's chunk budget is smaller, so the sweep range is
  smaller.
- Touch boots: 19.4 / 22.1 s on two slow containers (weight loads
  of 347 / 377 s on the same containers), 11.2 s on a faster one
  (weight load 103.7 s) - host speed moves both phases together.
  The py-spy profile of the 11.2 s boot shows ~70% of it waiting
  for the GPU: at 32B the three budget-sized warm chunks are
  compute, not overhead.
- Boot is dominated by the 32B weight load (1.7-6.3 min observed;
  container-dependent).
- Query (10,000 documents, one filter): best fast-path wall
  201.4 s, best arena wall 203.1 s, 0 wrong of 10,000.

Figure: plots/boot_tiered_32b.png

Figure: plots/boot_touch_timeline_32b.png

### Inside the touch pass

Figure: plots/boot_touch_timeline.png

A 0.1-second-binned timeline of the touch pass, from the py-spy
MainThread samples of 4B touch trial 0 (the 32B one above reads the
same way), categorized by what the CPU was doing (committed summary: `results/boot_touch_timeline.json`).
Read it with the caveat that the profiler was attached: the phase
runs 8.9 s here against 3.62 s clean, and sampling stretches
CPU-side launch work more than GPU waits.

- The first ~2.5 s is kernel-launch work for the first budget-sized
  chunk: GEMM + quant launches, attention and Triton launches, with
  red bursts of DeepGEMM reading cached kernel binaries off the
  volume on first sight (~0.5 s total across the phase - the loads
  the touch pass exists to absorb).
- The solid band after it is the CPU waiting while the GPU runs
  that chunk. In a real query this time is never idle: the CPU
  packs and launches the next chunk while the GPU runs the current
  one. Warmup is the degenerate case - the whole warm corpus fits
  one chunk, so there is no next chunk to build.
- The second launch band is the other attention mode's budget chunk
  plus the tiny-chunk ladder, again with cache-read bursts.
- The tail is pure GPU wait: the fast-path chunk running to
  completion and the final synchronize.

So the clean 3.62 s decomposes as roughly 2.8 s of GPU forward
compute on the three budget-sized warm chunks, ~0.3-0.4 s of tiny
ladder, and ~0.5 s of one-time cache loads - the phase is
GPU-compute-bound, not overhead-bound. Shrinking it further means
warming with smaller forward chunks plus bare budget-sized GEMMs
(~1 s touch), at the price of the first real query paying a few
tens of milliseconds of binary loads itself.

## What the numbers mean

- The per-container warmup no longer re-runs the GEMM sweep, and
  per-container boot cost did not regress: 3.62 s of touch against
  3.7-4.9 s of sweep before, with strictly more compile coverage
  behind it (the generator's complete list to the budget, which the
  per-boot sweep could never afford).
- The one-time price of that coverage is 103.8 s, paid once per
  (software stack, GPU, model, budget) into the shared volume - not
  per container, not per query.
- Query behavior is unchanged, as the audit predicted: kernels key
  on token counts, not on query content, so decoupling warmup from
  the payload could not move the query numbers - and it did not
  (28.42 s vs 28.30 s committed, 0 wrong).
- Boot is now load-model-bound: ~33 s of the ~39 s cold boot is
  weight loading (network + per-tensor loaders + imports). That is
  the next thing to attack if boot matters; warmup is no longer on
  the critical path.

## Data

- Committed summary: `results/boot_tiered.json`,
  `results/baseline_filter1.json` (refreshed same day).
- Raw: `/results/boot/boot_tiered.json` and the py-spy profiles
  `/results/boot/pyspy_*.speedscope.json` on the `quail-results`
  volume (load them at speedscope.app).
- Before-numbers: `results/boot_profile.json` (swept warmup, kept
  as the historical reference), `results/m1_filter1.json`.
