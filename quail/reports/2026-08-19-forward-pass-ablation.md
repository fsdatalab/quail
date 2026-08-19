# Forward-pass ablation: what made it faster

Date: 2026-08-19. One H100 on Modal, Qwen3 4B fp8.

We ran the same workload (10,000 documents, five filter stages,
~3.83 million input tokens) through four configurations, each adding
one change to the previous one. The question: where does the speed
come from? Figure: `plots/ablation_ladder.png`.

## The four configurations

| rung | what it is | what changed from the previous one |
|---|---|---|
| A0 | stock vLLM with fp8 KV storage | nothing — the starting point |
| A1 | stock vLLM with bf16 KV storage | stopped converting KV values to 8-bit on every write and back on every read |
| A2 | our executor with vLLM's small kernels | replaced the vLLM engine with our own loop: we pack documents into large batches ourselves, keep each document's KV in a page-managed GPU buffer instead of vLLM's cache, and read YES/NO answers from 12 output-matrix rows instead of the full 151k-token vocabulary. But the small operations between the matrix multiplies (normalize, quantize, rotate) still use vLLM's code |
| A3 | our executor with our three fused kernels | replaced vLLM's normalize+quantize, silu+quantize, and qk-norm+rotate with our own Triton kernels that do each group in one pass instead of two or more. This is the current code |

## Results

Each configuration ran twice. The two runs agreed within 0.5 seconds.

| rung | time | tokens/s | step from previous |
|---|---|---|---|
| A0 | 40.9 s | 95,800 | — |
| A1 | 39.5 s | 99,400 | 1.4 s faster (stopped the 8-bit KV conversion) |
| A2 | 42.3 s | 90,800 | **2.8 s slower** (replaced the engine, still using vLLM's small kernels) |
| A3 | 33.0 s | 116,200 | 9.3 s faster (our three fused kernels) |

End to end: 39.5 s (stock bf16) to 33.0 s (our executor) = 1.20x.

## What we learned

**Replacing the engine made things slower, not faster.** A2 (42.3 s)
is slower than stock bf16 (39.5 s). A kernel-level profile of both
packed configurations (`results/ablation_profile.json`) shows why.
GPU kernel time per token:

| work | stock vLLM fp8 (banked engine profile) | A2 | A3 |
|---|---|---|---|
| matrix multiplies | 5.52 µs | 5.08 µs | 5.42 µs |
| attention | 0.81 µs | 0.79 µs | 0.85 µs |
| small kernels (normalize, quantize, silu, rotate) | 4.00 µs | 4.21 µs | 1.87 µs |
| KV storage work | 0.43 µs | 0.84 µs | 0.38 µs |
| **total** | **10.77 µs** | **11.01 µs** | **8.60 µs** |

Three facts fall out:

- The matrix multiplies and attention are the same kernels on both
  sides, and replacing the engine cannot speed them up — it doesn't.
  (The multiplies are even slightly faster per token at our larger
  batch size.)
- The small kernels in A2 are vLLM's own, and they cost the same per
  token in our loop as inside the engine (4.21 vs 4.00 µs). No win
  there.
- What tips A2 under stock is the price of keeping KV our way:
  writing each document's KV into pages, building the block tables
  that attention reads through, the answer-merge write-back, plus two
  extra copies per layer inside vLLM's unfused qk-norm path. That is
  0.84 µs/token against the engine's 0.43 for a plain cache write.
  (The write itself is one small kernel per layer, `kv_row_scatter`,
  added after the first version of this study showed the old
  gather-plus-scatter cost 0.50 µs/token on its own.) A2 also avoids
  re-reading 4.7 million cached tokens (the stock client pays
  attention over them across stages), but those re-reads are cheap
  next to the matrix multiplies and do not close the gap.

**All the speed comes from three fused kernels.** A2 to A3 is a 9.3
second drop, 2.4 microseconds per token of wall. The profile shows
2.8 µs/token of GPU work removed: 2.34 from fusing the small kernels
(four quantize passes per layer become one; normalize+add+quantize
and silu+multiply+quantize each become one pass) and 0.46 from the
copies the fused qk-norm kernel avoids. Both A2 and A3 are GPU-bound
(wall equals GPU-busy time within 0.1 s), so removed GPU work lands
directly in the wall. The same fusion measured 2.9 µs/token of kernel
time at the exploration's 25,305-token batches — the GPU-work saving
is flat in batch size; at larger batches more of it lands in the wall
because the loop has fewer gaps to hide it in.

**The 8-bit KV conversion costs 1.4 seconds on the stock side.**
A0 to A1 = 3.5% of the run. Converting attention state to fp8 on
every write and back on every read is work that buys nothing in this
workload (the saved memory does not matter when the pool is not under
pressure). Consistent with the exploration's per-token measurement
of 5.9%.

## Consistency checks

- A3 produced the same answers as the banked current-executor run:
  1,807 documents survived all five stages, 6,294 answers disagreed
  with the planted flags, out of 23,113 total answers. This matches
  `results/m1_filter.json` exactly.
- A2 and A3 disagreed on 2,273 of 23,113 answers (9.8%). This is
  expected: different quantization kernels round differently at thin
  YES/NO margins. The exploration measured 1,768 flips per 10,000
  answers from changing one kernel. The token counts differ too
  (A2 processed 4,420 more tokens) because a different answer at one
  stage changes which documents reach the next stage.
- A1 matched the committed stock baseline exactly: 1,873 survivors,
  6,229 wrong answers. The baseline is trustworthy.
- Wrong-answer rates are in the same band across all four
  configurations (6,194 to 6,294). No configuration is more accurate
  than another.

## What is not in this study

**Pinned-memory staging** (building each batch's GPU input tensors
through page-locked host memory instead of normal memory) is on in
both A2 and A3. Issue #12 already measured its contribution: 1.4
seconds. This study treats it as part of the base configuration, not
as a separate rung. The five-rung version that separated it is in git
history.

**The arena host-index cache** (keeping the page-table mappings on
the CPU instead of rebuilding them per batch) is also on in A2. Its
contribution, ~4 seconds, is measured by comparing A3 against the
banked pre-#12 code that lacked both the cache and the pinned staging.
That comparison uses runs from different containers, so it carries
the +/-3% container-to-container variation (~1.2 seconds).

## How to reproduce

From the `quail/` directory:

    uv run modal run ablations/forward_pass.py::run_all 2>&1 | tee results/ablation_forward.log

Stock and packed configurations run in separate containers (the vLLM
engine process holds the GPU until it exits, so a second engine boot
in the same container fails). The packed configurations share one
container and one model load.

## Data files

- `results/ablation_forward.json` — per-rep walls, token counts,
  answer counts, CPU phase timings, consistency checks.
- `results/ablation_profile.json` — per-kernel-category GPU time for
  A2 and A3 (kernel events only, recomputed from the chrome traces;
  the traces are on the Modal results volume).
- `reports/plots/ablation_ladder.png` — generated by
  `reports/make_ablation_plots.py`.
- Logs: `results/ablation_packed_postfix.log`,
  `results/ablation_stock_a0.log`, `results/ablation_stock_a1.log`,
  `results/ablation_profile_postfix.log`.
