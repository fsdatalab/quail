# Principled kernel warmup, measured on a cold cache

The previous `warm_kernels` guessed a token grid and never called
`run_join`. It only flipped `attention_mode` on a filter chunk. This
change uses vLLM's DeepGEMM config-boundary generator for M, and
warms attention through the real loops.

## Setup

One H100, Qwen3 4B fp8, `tests/gpu/boot_profile.py --cold-cache
--trials 1 --no-stock`. DeepGEMM and Triton caches pointed at `/tmp`
so the container did not read the shared kernel volume. The cell is
attached to `quail-milestone1`.

Committed summary: `results/boot_profile_cold.json`. Raw record:
`/results/boot/cold.json` on the `quail-results` volume. Function
call id `fc-01M0TSWDBEV0TEA25FAWTYT1WX` (in
`results/boot_profile_cold.log`).

The warmup now:

- GEMM: one M list per linear from
  `vllm.model_executor.warmup.deep_gemm_warmup._generate_optimal_warmup_m_values`,
  up to the chunk budget (110,376 at 4B).
- `run_filter`: unified with arena writes, the no-arena fast path,
  and tiny chunks at 64, 128, 256, 512, 1,024, 2,048 tokens.
- `run_join`: long prefix + many short suffixes, short prefix +
  longer suffixes, and a tiny tail.

## Prediction, stated before the run

`warm_kernels_s` 90-240 s. `load_model_s` 30-40 s (committed
warm-cache mean 35.1 s). Warm reuse ~0. Compared with the committed
warm-cache `warm_kernels_s` of 4.46 s mean.

## Result

| Phase | This run (cold cache) | Committed (warm cache, 3-trial mean) |
|---|---|---|
| load_model_s | 30.02 s | 35.1 s |
| arena_s | 0.34 s | 0.45 s |
| warm_kernels_s | 209.22 s | 4.46 s |
| boot_s | 240.97 s | 41.28 s |
| warm reuse boot_s | 0.0 s | 0.0 s |

Warmup internals: 7,583 GEMM launches (`m_source=vllm`), 8 filter
forwards, 3 join forwards. `run_join` ran. The GEMM bar finished in
3 min 15 s (195 s of the 209 s warmup); the packed forwards were
the remaining 14 s.

Figure: plots/boot_profile_cold.png

## What the numbers mean

- The 90-240 s prediction held: 209.22 s, compared with 4.46 s when
  the same machine reads a warm kernel volume (47x). Almost all of
  that is first-time DeepGEMM compiles. After the first ~100
  launches of each linear, the rest of the 7,583 are cache hits at
  hundreds of GEMMs per second.
- `load_model_s` 30.02 s sits on the low end of the 30-40 s band
  and below the committed 35.1 s mean. One trial; container
  variance on weight load is already 28.6-38.5 s in the committed
  profile.
- Warm reuse is 0.0 s. The compiled kernels stay in process.
- 7,583 M values is the generator at the full chunk budget (about
  1,896 per linear, mostly multiples of 64). That is complete
  relative to DeepGEMM's heuristic. It is also more launches than
  a run will ever need. If boot time on a cold volume matters, the
  next cut is to keep wave-boundary M values only above 4,096.

## Changes

- `quail/executor/loop.py`: `deepgemm_m_values`, `join_warmup_jobs`,
  `run_join` in `warm_kernels`.
- `tests/gpu/boot_profile.py`: `--cold-cache` and `--no-stock`.
- `tests/test_kernel_warmup.py`: CPU tests for the helpers.
