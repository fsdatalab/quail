# Boot and one filter query after skipping warm_kernels

Date: 2026-08-24. One H100. Qwen3 4B fp8. Warm kernel volume.
Existing cells: `tests/gpu/boot_profile.py --trials 1` and
`tests/gpu/milestone1.py::run_filter --reps 1`.

Committed summary: `results/boot_query_after_skip.json`.
Raw: `/results/boot/compare.json` and `/results/m1/m1_filter.json`
on `quail-results`. Boot function call ids
`fc-01M0TTZA8D6MXYFEC7F8SDA6Z0` (Quail) and
`fc-01M0TTZADHWMPW8J6TESP791T9` (stock).

## Prediction, stated before the run

Quail `warm_kernels` skipped (~0 s). Quail cold boot 30-38 s
(`load_model`). Quail first 10k five-filter query 35-42 s (34.6 s
warmed plus lazy cubin load). Stock cold boot ~250 s. Stock query
the committed 48.7-49.0 s.

## Result

| | Quail | Stock vLLM |
|---|---|---|
| Cold boot | 81.43 s | 242.65 s |
| `warm_kernels_s` | 0.05 s (`skipped=True`) | (inside `LLM(...)`) |
| Warm reuse | 0.0 s | 0.0 s |
| 10k five-filter query | 43.88 s (first query) | 48.88 s mean (committed) |

Figure: plots/boot_query_after_skip.png

Quail boot breakdown this trial: `load_model_s` 79.41 s, `arena_s`
0.44 s, `warm_kernels_s` 0.05 s.

Quail query: 4,097,571 fresh tokens, 40,052 answered, 4,645
survivors, 0 wrong. First chunk 8.74 s, second 1.91 s, later full
chunks about 0.14-0.86 s.

## What the numbers mean

- The skip worked. `warm_kernels` was 0.05 s and launched nothing.
- This trial's 79.41 s `load_model` is not the solo cost. The filter
  cell and the boot cell loaded weights at the same time against
  `quail-hf-cache`. Safetensors copy was 50-63 s here, compared with
  2.4 s on the earlier solo cold-cache run. Prior solo
  `load_model_s` is 30-38 s (committed `boot_profile.json` mean
  35.1 s). A solo Quail boot with the skip should sit in that 30-38 s
  band. This trial does not settle that. A second solo boot would.
- Stock boot 242.65 s matches the ~250 s prediction (committed mean
  249.68 s). Most of that is still the `LLM(...)` constructor, not
  a missing cubin cache.
- Against this trial's 81.43 s Quail boot, stock is 3.0x slower.
  Against the 35.1 s solo load mean plus 0.05 s skip, stock is about
  6.9x slower. The second figure is the one to use for a container
  that is not sharing the HF volume.
- The first query paid the cubin loads. 43.88 s, compared with the
  34.6 s warmed reference: 9.3 s extra. Almost all of that is chunk
  0 (8.74 s, compared with ~1 s on a warmed run). Chunk 1 is still
  high (1.91 s). After that the chunk times look like a warmed run.
- Stock query 48.88 s is the committed client (separate requests per
  stage, document-cap admission), not re-run this session. That file
  is the older YES/NO corpus (23,381 requests). Quail this session
  is the TRUE/FALSE corpus (40,052 answers). Token counts are in the
  same band (Quail 4.10M fresh, stock 4.90-4.94M). Treat the 43.88
  vs 48.88 comparison as same workload class, not a matched pair.

## Changes

No new cell. The skip is in `warm_kernels` from the previous commit.
