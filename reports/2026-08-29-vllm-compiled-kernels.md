# What the custom kernels add over vLLM's compiled kernel set

Date: 2026-08-29. One H100 on Modal, Qwen3 4B fp8.
Cell: `ablations/vllm_compiled_kernels.py`.

The question: how much speed do our custom JIT kernels add, measured
against the kernels a user would get from vLLM's own torch.compile of
this model? An earlier ablation (2026-08-19; deleted as superseded by
this report, see git history) compared our kernels against vLLM's ops
called one by one, unfused. That comparison could be generous to us:
stock vLLM compiles the model at boot, and its compiled graph could
fuse the same operation pairs we fuse. This experiment adds the
missing configuration and runs the comparison on one filter query and
one join query.

Answer: the fused kernels are worth 26% per token on both queries
against the compiled kernel set (29% against the unfused ops),
because stock vLLM's compiled graph does not fuse the quantization
work for this model. Figure: `plots/kernel_source_rates.png`.

## What stock vLLM's compiled graph runs (measured, not assumed)

We booted stock vLLM 0.26.0 at its defaults on the same image and
profiled a prefill pass in-process
(`/results/ablations/kernel_source_stock.json`):

- The default optimization level is -O2, which compiles the model
  through torch.compile and Inductor. Custom ops resolve to native
  implementations except `quant_fp8`, which a blocked-fp8 checkpoint
  forces on.
- The config enables the RMSNorm+quant and SiLU+quant fusion passes
  (`fuse_norm_quant` and `fuse_act_quant` both true) — but the
  profiled kernel list contains no fused norm+quant or silu+quant
  kernel. The group-quant that this model traces under DeepGEMM's
  ue8m0 scale format (on by default on this image, for stock and for
  our engine alike) does not match the patterns the passes register,
  so the rewrite never fires. What actually runs, per layer:
  - two Inductor-generated add+rms_norm kernels,
  - one Inductor-generated silu*mul kernel,
  - two Inductor-generated kernels for the q/k head norms and rotary
    embedding (the dedicated qk-norm+rope fusion pass is off at
    every -O level in 0.26.0),
  - four standalone CUDA group-quant launches
    (`per_token_group_quant_8bit_kernel`), one per GEMM input. This
    is the largest small-kernel cost in the stock profile: 332 ms of
    the profiled pass, against 132 ms for silu*mul and 94 ms for
    both norms together.
- The GEMMs (DeepGEMM sm90 fp8) and attention (FlashAttention-3) are
  the same kernels the packed executor calls. They and the KV-cache
  write sit outside the compiled graph.

So for this model, torch.compile's contribution is Inductor's fusion
of the elementwise chains. The quantization stays unfused, and that
is where most of our kernels' win lives.

## The three configurations

Same engine, same packing, same DeepGEMM matmuls, same
FlashAttention-3 calls, same KV write. Only the small kernels between
the matmuls (and the join merge) change. The non-quail paths live in
a Pipeline subclass inside the ablation cell; the engine is
unchanged.

| configuration | between-GEMM kernels | join merge |
|---|---|---|
| quail | our three fused Triton kernels | our fused merge+quant Triton kernel |
| vllm_ops | vLLM's ops one by one, unfused (fused_add_rms_norm + quant, silu_and_mul + quant, per-head norms + rotary as five launches) | vLLM's merge_attn_states + separate group quant |
| vllm_compiled | the set stock's compiled graph was measured to run: torch.compile over the native add+rms_norm, silu*mul, and q/k-norm+rope math (vLLM's Inductor settings, dynamic token count), plus the same standalone group-quant per GEMM input | vLLM's merge_attn_states + separate group quant (stock never merges inside its compiled graph; its cascade merge lives in the attention backend) |

A per-kernel probe (`/results/ablations/kernel_source_probe.json`)
checked the wiring before the measured runs: the compiled segments
hold one graph across token counts, the add+rms_norm segment folds
the residual write into one generated kernel plus the quant launch
(no extra copy), dequantized outputs sit within one fp8 rounding
step of our kernels', and a 32-document filter with planted flags
answers identically (0 wrong) through all three configurations.

## The two queries

- Filter query: the committed 10,000-document IMDB five-filter
  workload (4.10M fresh tokens), unified attention — the filter
  path.
- Join query: 100 BioDEX reports x 256 reaction terms, 25,600 pairs
  (1.09M fresh tokens), merge_quant attention — the join path.

Two repetitions per configuration, one container, one model load.
The tables use each configuration's second repetition, matching the
cell's comparison rows; the container's very first measured run
(quail, repetition 0) was 2.4 s slower than its repetition 1, while
every other configuration's repetitions agreed within 0.05 s.

## Predictions (stated before the run)

- Filter, vllm_ops: +2.3 to +2.6 us/token over quail (the earlier
  2026-08-19 ablation measured a 2.4 gap). Measured: +2.48. Correct.
- Filter, vllm_compiled: +1.7 to +2.2 us/token, reasoning that the
  stock inventory keeps all four group-quant launches per layer, so
  most of the fusion saving stays with quail. Measured: +2.23, just
  above the top of the band.
- Join: the filter gap plus the unfused merge — +2.5 to +3.0
  (vllm_ops) and +2.0 to +2.7 (vllm_compiled). Measured: +3.13 and
  +2.83, each about 0.1 above its band.

Both misses are small and on the same side: the vLLM-side
configurations cost slightly more than predicted, mostly because
Inductor's silu*mul kernel turned out slower than vLLM's CUDA op
(below).

## Results

Wall time and rate, second repetition
(`/results/ablations/kernel_source_filter.json`,
`/results/ablations/kernel_source_join.json`):

| configuration | filter wall | filter us/token | join wall | join us/token |
|---|---|---|---|---|
| quail | 34.7 s | 8.48 | 11.9 s | 10.93 |
| vllm_compiled | 43.9 s | 10.71 (+26%) | 15.0 s | 13.75 (+26%) |
| vllm_ops | 44.9 s | 10.96 (+29%) | 15.3 s | 14.06 (+29%) |

- Our fused kernels save 9.1 s of the 43.9 s filter query against
  the compiled kernel set, and 3.1 s of 15.0 s on the join query —
  1.26x on both. Against the unfused ops it is 1.29x on both.
- The compiled set beats the unfused ops by only 0.25 (filter) to
  0.31 (join) us/token — about a tenth of the gap to quail.
  torch.compile is not where the speed is for this model.
- The join gap exceeds the filter gap by 0.60-0.65 us/token on both
  vLLM-side configurations. That is the price of composing the join
  merge from vLLM's pieces (gather the rows with cached context,
  merge_attn_states, scatter back, quantize — five launches per
  layer) against our one fused merge+quant kernel.
- quail's join rate here (10.93 us/token at 100x256) matches the
  10x256 measurement in the attention-paths report (11.04).

Answers: on the filter query all three configurations returned
byte-identical answers — 40,052 answers, 4,645 survivors, 0 wrong
against the planted flags, 0 disagreements — and the quail counts
equal the banked `results/attention_paths.json` values exactly. On
the join query the vLLM-side configurations each flipped about 0.2%
of pairs (61 and 56 of 25,600) with yes-counts within 27 of quail's
25,546; this workload saturates the 4B model near all-TRUE, and
thin-margin flips under different kernel rounding are the known
behavior from the accuracy-vs-stock study.

### Where the GPU time goes

GPU kernel microseconds per fresh token on a profiled 3,000-document
filter run (`/results/ablations/kernel_source_profile.json`).
Figure: `plots/kernel_source_profile.png`.

| category | quail | vllm_ops | vllm_compiled |
|---|---|---|---|
| matrix multiplies (DeepGEMM) | 5.46 | 5.07 | 5.15 |
| attention (FlashAttention-3) | 0.79 | 0.72 | 0.74 |
| our fused Triton kernels | 1.46 | — | — |
| vLLM norm and rotary ops | 0.00 | 1.83 | 0.00 |
| vLLM silu*mul op¹ | 0.00 | 0.73 | 0.00 |
| Inductor-generated kernels | — | — | 2.65 |
| standalone group-quant | 0.39 | 1.64 | 1.68 |
| copies | 0.23 | 0.70 | 0.22 |
| **total** | **8.33** | **10.70** | **10.44** |

¹Stored in the data file's "other" bucket: the act_and_mul kernel's
template name escapes the category matcher. Per-kernel rows in the
file confirm the split: rms_norm 0.88, rotary 0.47,
fused_add_rms_norm 0.48 (together the 1.83), act_and_mul 0.73.

Readings:

- The standalone group-quant is the cost torch.compile does not
  remove: 1.68 us/token in the compiled set against quail's 0.39
  (our one remaining quant, the attention output feeding o_proj).
  Fusing the other three quant passes into their producers is most
  of our win.
- Inductor's generated kernels (2.65) also lose to our fused
  kernels (1.46) on the work they do fuse. The single biggest
  reason: its silu*mul kernel costs 1.49 us/token where vLLM's CUDA
  op costs 0.73 and our fused silu+quant kernel does the activation
  and the quantization together in 0.78 (per-kernel rows in the
  data file: 1.837 s, 0.907 s, 0.968 s over 1.24M tokens).
- Inductor's fusion is a wash on the ops themselves: it replaces
  2.56 us/token of vLLM norm, rotary, and silu*mul kernels with
  2.65 us/token of generated kernels. What torch.compile actually
  recovers is the data movement around the unfused sequence — the
  copies bucket falls from 0.70 to 0.22 (mostly the two contiguous
  copies per layer the vLLM q/k path needs). Net, that is the
  0.25-0.31 us/token difference between the two vLLM-side
  configurations.
- Small-kernel work in total (everything but matmuls and
  attention): quail 2.08, vllm_compiled 4.55, vllm_ops 4.90
  us/token. The GPU-time deltas (2.11 and 2.37) account for about
  95% of the measured wall deltas — all three configurations are
  GPU-bound.

## Scope notes

- One model (Qwen3 4B fp8). At 32B the same absolute per-token
  saving would sit on ~60 us/token of matmul-dominated work
  (attention-paths report), so the relative win shrinks by roughly
  8x. Not re-measured here.
- Stock vLLM at -O2 also runs CUDA graphs for decode-sized batches.
  This workload is large-chunk prefill, where CUDA graphs do not
  apply, and the packed executor launches eagerly in every
  configuration — so graph capture is outside this comparison.
- The vllm_compiled configuration reproduces stock's kernel set
  inside our executor; it is not stock vLLM end to end. The stock
  engine baseline for these queries is the separate stock-vllm
  reports.

## How to reproduce

    uv run modal run ablations/vllm_compiled_kernels.py::run_stock_kernels
    uv run modal run ablations/vllm_compiled_kernels.py::run_probe
    uv run modal run ablations/vllm_compiled_kernels.py::run_queries
    uv run modal run ablations/vllm_compiled_kernels.py::run_profile

Modal app `quail-milestone1`, one H100 per cell. Function calls:
queries `fc-01M17GRV662HJ9D4EQT7YMPA9Y`, profile
`fc-01M17H1SNZM6D40470RD0YYERY`, stock inventory
`fc-01M17H1VWNYPSD3AFRMXFR03FM`, probe
`fc-01M17GNH10ZBAK23Y4034NKCVV`.

## Data files

All on the `quail-results` volume:

- `/results/ablations/kernel_source_filter.json` — filter walls,
  rates, counts, comparisons, banked gate.
- `/results/ablations/kernel_source_join.json` — join walls, rates,
  yes-counts, disagreements.
- `/results/ablations/kernel_source_profile.json` — per-category GPU
  time and launch counts for the three configurations.
- `/results/ablations/kernel_source_stock.json` — stock vLLM's
  resolved compilation config and kernel inventory.
- `/results/ablations/kernel_source_probe.json` — per-kernel parity,
  compiled-segment launch counts, planted-flag check.

Plots are rebuilt by `reports/make_kernel_source_plots.py` (the
`modal volume get` commands are in its docstring). Local tee logs:
`results/kernel_source_*.log`.
