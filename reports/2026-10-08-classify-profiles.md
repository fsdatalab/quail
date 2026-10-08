# GPU profiles of three classification queries, Quail and stock vLLM

Date: 2026-10-08.

In every profiled 5 s window, Quail keeps the GPU busy 98 to 100% of
the time. Stock vLLM leaves it idle on Decision-2.0-Kai 0.6B: busy 72%
on the classification queries and 39% in BIO-6's join. On Qwen3 4B,
stock vLLM is busy 99% on IMDB-11 and IMDB-14, where its extra time
comes from more tokens and slower kernels, not idle GPU, and 92% in
BIO-6.

The windows only sample stretches when the engine runs forward passes.
On BIO-6, stock vLLM went at least 236 s (Kai) and 362 s (Qwen3 4B)
with no engine step at all, while the baseline built one prompt per
join pair and added them all to vLLM before its first step.

[![GPU busy share and longest gap](plots/classify_profiles_gpu_busy_and_gap.png)](plots/classify_profiles_gpu_busy_and_gap.png)

## Setup

- Queries: IMDB-11, IMDB-14, and BIO-6. They are the classification
  queries farthest from the SoL estimate in the sf 0.5 runs of
  2026-10-01, among those that run long enough to sample a steady
  state. IMDB-11 is one classification. IMDB-14 is a filter, then
  classifications with 4 and 8 labels. BIO-6 is a join, then a
  classification with 26 labels.
- Models: Qwen3 4B FP8, DiffusionGemma 26B-A4B FP8, and
  Decision-2.0-Kai 0.6B BF16. Stock vLLM's planner refuses AI.CLASSIFY
  on DiffusionGemma, whose engine returns no text, so it has no vLLM
  runs.
- Engines: Quail, and stock vLLM with operator-at-a-time submission.
  Scale factor 0.5, one H100 per run on Modal, each query in a fresh
  process.
- Capture: `experiments/cells/classify_profiles.py`, torch.profiler
  with CPU and CUDA activity. A window starts and stops at a forward
  pass boundary: a Quail chunk or a vLLM engine step. The first window
  starts 10 s into model execution, then one every 120 s. Model
  loading and planning run unprofiled.
- Stock vLLM ran with its engine core in the client process
  (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) so the profiler sees its
  kernels. The benchmark runs keep the engine core in its own process
  for Qwen3 and Kai.
- No prediction was written down before these runs.

Runs, on the `quail-results` volume:

| Directory | Runs | Code | Function call ids |
|---|---|---|---|
| `/results/ablations/classify-profiles/20261008T042520Z/` | IMDB-11 on all three engines and models; stock vLLM IMDB-14 | `9611eae` (Quail), `2189e98` (vLLM) on `claude/elegant-turing-ucd81t` | `fc-01M4CW4XNDNF7TTFZZN5N2QZP5`, `fc-01M4CW4XRRJW6PEB0S6TVS8KYT`, `fc-01M4CW4XVGKS88JDYP179EBXWM`, `fc-01M4CWHXXV0JMMKQC5KNS7ZGYQ`, `fc-01M4CWHY3QKR72AEZVKQN5G3QH` |
| `/results/ablations/classify-profiles/20261008T052000Z/` | Quail IMDB-14 and BIO-6; stock vLLM BIO-6 | `d2c6a60` | `fc-01M4CY97D7CAR1V06VX3SWRWMK`, `fc-01M4CY97GCBF2T2K1SHFMQHEDP`, `fc-01M4CY97JZA6RQHWPNWWJVHYMV`, `fc-01M4CY8XH0JEYN207M1G6GC6JX`, `fc-01M4CY8XM04AQWRWX1HAVNYJKZ` |

The first directory also holds Quail IMDB-14 and BIO-6 runs whose
window never armed, and a one-window vLLM BIO-6 run. The later
directory replaces both.

## Result

GPU busy share is GPU-active seconds over the seconds from each
window's first device event to its last, summed over the run's full
windows. Overlapping kernels count once. A window cut short by the
query's end is left out.

| Model | Engine | Query | Windows | GPU busy | Longest idle gap |
|---|---|---|---:|---:|---:|
| Qwen3 4B | Quail | IMDB-11 | 1 | 99.9% | 0.01 ms |
| Qwen3 4B | stock vLLM | IMDB-11 | 1 | 99.2% | 12.5 ms |
| Qwen3 4B | Quail | IMDB-14 | 1 | 98.6% | 54.1 ms |
| Qwen3 4B | stock vLLM | IMDB-14 | 1 | 99.2% | 11.6 ms |
| Qwen3 4B | Quail | BIO-6 | 2 | 99.9% | 4.8 ms |
| Qwen3 4B | stock vLLM | BIO-6 | 4 | 92.5% | 268.9 ms |
| DiffusionGemma | Quail | IMDB-11 | 1 | 99.7% | 2.0 ms |
| DiffusionGemma | Quail | IMDB-14 | 1 | 99.7% | 0.3 ms |
| DiffusionGemma | Quail | BIO-6 | 3 | 99.2% | 95.1 ms |
| Kai | Quail | IMDB-11 | 1 | 99.4% | 0.01 ms |
| Kai | stock vLLM | IMDB-11 | 1 | 71.9% | 348.6 ms |
| Kai | Quail | IMDB-14 | 1 | 98.4% | 41.3 ms |
| Kai | stock vLLM | IMDB-14 | 1 | 71.7% | 377.5 ms |
| Kai | Quail | BIO-6 | 1 | 99.6% | 0.1 ms |
| Kai | stock vLLM | BIO-6 | 5 | 39.2% | 6,179.7 ms |

### Where stock vLLM's GPU waits

- **Kai classification, per request copies.** Each engine step runs
  about 160 ms on the GPU for about 165 requests. It opens with about
  165 small host-to-device copies and closes with about 165
  device-to-host copies of the pooled decision rows, one per request.
  Then the GPU waits about 21 ms for the next step. Qwen3 4B text
  generation makes 2 device-to-host copies per step.
- **Kai BIO-6 join, host-bound steps.** Each step uploads about 850
  pair requests, computes for 0.3 s, reads 850 pooled rows back, then
  waits 0.35 to 0.45 s for the host. The prefix cache serves almost
  every prompt: 1,169M input tokens against 35M fresh. The longest gap,
  6.2 s, is at the end of the join, in the last window.
- **Qwen3 4B BIO-6 join, scheduling gaps.** Steps run 0.3 s on the
  GPU with 50 to 270 ms gaps between them.
- **BIO-6, request admission.** The baseline submits every join pair
  in one `generate` call, and vLLM adds all the requests before its
  first step. A window due at 306 s on Qwen3 4B opened at 668 s, and
  one due at 201 s on Kai opened at 437 s. The engine ran no step in
  between, so the GPU was idle for at least 362 s and 236 s.

### Where Quail's time goes

Quail's GPU stays busy in the windows, and on IMDB-11 and IMDB-14 it
computes within 2% (Qwen3 4B) and 13% (DiffusionGemma) of the SoL
estimate's tokens. Its time over SoL, 2.3x on Qwen3 4B IMDB-11 and up
to 4.4x on DiffusionGemma BIO-6, is per-token kernel time. Kernel time
in the windows splits as follows:

- Qwen3 4B: GEMM 49 to 65%, attention 15 to 35%, fused quantize and
  activation 11 to 14%.
- DiffusionGemma: MoE 25 to 39%, GEMM 19 to 27%, attention 7% on
  IMDB-11 and 31% on BIO-6.
- Kai: GEMM 37 to 60%, attention 15 to 45%.

On every IMDB-14 and BIO-6 run, Quail's first window opened 2 to 13 s
after it was due. All of Quail's forward passes go through the hooked
function, so Quail ran no forward pass in those seconds. The stage
records do not show what it did instead.

## Screenshots

Each pair shows the same model and query, stock vLLM first. In every
screenshot, `stream N N` is the GPU and `thread N (python)` is the
engine's main thread. Time runs left to right across the window.

| Pair | Stock vLLM | Quail |
|---|---|---|
| Kai, IMDB-11 | [72% busy](plots/classify_profiles_kai_imdb-11_stock_vllm.png) | [99% busy](plots/classify_profiles_kai_imdb-11_quail.png) |
| Kai, BIO-6 join | [38% busy](plots/classify_profiles_kai_bio-6_stock_vllm.png) | [100% busy](plots/classify_profiles_kai_bio-6_quail.png) |
| Qwen3 4B, BIO-6 join | [79% busy](plots/classify_profiles_qwen3_4b_bio-6_stock_vllm.png) | [100% busy](plots/classify_profiles_qwen3_4b_bio-6_quail.png) |
| Qwen3 4B, IMDB-11 | [99% busy](plots/classify_profiles_qwen3_4b_imdb-11_stock_vllm.png) | [100% busy](plots/classify_profiles_qwen3_4b_imdb-11_quail.png) |

[DiffusionGemma, BIO-6](plots/classify_profiles_dgemma_bio-6_quail.png)
is Quail's window with the lowest busy share, 98%, with one 95 ms gap
at a chunk boundary.

On the Qwen3 4B IMDB-11 Quail screenshot, the GPU track starts 0.9 s
into the window. Quail had about 0.9 s of kernels queued when the
profiler started, and the profiler records only kernels launched after
it starts. The stop synchronize at the end waited for that queue.

## What is not settled

- With the engine core in its own process, as benchmarked, vLLM can
  step while the client still adds requests, and part of its host work
  overlaps the GPU. The profiled vLLM BIO-6 run on Qwen3 4B took 873 s
  against 630 s in the 2026-10-01 benchmark; the IMDB runs were within
  8%. The idle shares above may overstate vLLM's idle time on BIO-6. A
  profile with the engine core in its own process, using vLLM's own
  profiler hooks, would settle it.
- Windows sample only stretches with forward passes. Recording the
  start time of every forward pass, which is cheap, would give each
  run's full timeline and the total time with no pass, for both
  engines.
- Each configuration ran once. Quail's profiled times differ from the
  2026-10-01 runs by 2 to 15% in both directions, on different
  physical GPUs and a newer engine.
- Kai has no sf 0.5 benchmark run and no SoL estimate at that scale.

## Reproduce

```
W=/tmp/classify-profiles; mkdir -p "$W"
P=ablations/classify-profiles
uv run modal volume get quail-results "$P/20261008T042520Z" "$W/"
uv run modal volume get quail-results "$P/20261008T052000Z" "$W/"
uv run --with matplotlib --with playwright==1.55.0 \
  python reports/make_classify_profiles_plots.py "$W" --screenshots
```
