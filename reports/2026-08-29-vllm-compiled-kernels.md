# What the custom kernels add over vLLM's compiled kernel set

Date: 2026-08-29. One H100 on Modal, Qwen3 4B fp8.
Cell: `experiments/vllm_compiled_kernels.py`.

The question: how much speed do our custom JIT kernels add, measured
against the kernels a user would get from vLLM's own torch.compile of
this model? Stock vLLM compiles the model at boot, and its compiled
graph could in principle fuse the same operation pairs we fuse — so
comparing only against vLLM's ops called one by one would be generous
to us. This experiment measures both alternatives on two QUAIL-B
queries.

Answer: on QUAIL-B queries, the fused kernels are worth 27% per token
on the filter path and 23% on the join path against the compiled
kernel set (30% and 26% against the unfused ops). The reason: stock
vLLM's compiled graph does not fuse the quantization work for this
model, and quantization is the largest small-kernel cost.
Figure: `plots/kernel_source_rates.png`.

## The four kernel sequences, side by side

Figure: `plots/kernel_paths.png` (source:
`reports/kernel_paths_diagram.html`). One layer of the 36-layer loop
in each measured configuration. The matrix multiplies (DeepGEMM
fp8_gemm_nt), FlashAttention-3, and the KV page write are the same
kernels in every column; the columns differ only in the small kernels
between them:

- vLLM ops, unfused: 18 launches per layer on the unified path.
- vLLM compiled-graph set: 15 launches per layer.
- Quail fused, unified path (filters): 11 launches per layer.
- Quail fused, merge_quant path (joins): 13 launches per layer (two
  attention calls plus the fused merge, which also does the o_proj
  quantization).

## What stock vLLM's compiled graph runs (measured, not assumed)

We booted stock vLLM 0.26.0 at its defaults on the same image and
profiled a prefill pass over IMDB-1-shaped prompts in-process
(`/results/ablations/kernel_source_stock.json`):

- The default optimization level is -O2, which compiles the model
  through torch.compile and Inductor. Custom ops resolve to native
  implementations except `quant_fp8`, which a blocked-fp8 checkpoint
  forces on.
- The config enables the RMSNorm+quant and SiLU+quant fusion passes
  (`fuse_norm_quant` and `fuse_act_quant` both true) — but the
  profiled kernel list contains no fused norm+quant or silu+quant
  kernel. The group-quant this model traces under DeepGEMM's ue8m0
  scale mode (on by default on this image, for stock and for our
  engine alike) does not match the patterns the passes register, so
  the rewrite never fires. What actually runs, per layer: two
  Inductor-generated add+rms_norm kernels, one Inductor silu*mul
  kernel, two Inductor kernels for the q/k head norms plus rotary
  (the dedicated qk-norm+rope fusion pass is off at every -O level in
  0.26.0), and four standalone CUDA group-quant launches
  (`per_token_group_quant_8bit_kernel`), one per GEMM input. The
  standalone quant is the largest small-kernel cost in the stock
  profile: 317 ms of the profiled pass, against 124 ms for silu*mul
  and 89 ms for both norms together.

So for this model, torch.compile's contribution is Inductor's fusion
of the elementwise chains. The quantization stays unfused, and that
is where most of our kernels' win lives.

## The three kernel sources

Every run goes through the real planner and the real worker execution
core (`quail.runtime.worker._execute_single`); only the pipeline's
kernel source is swapped, so packing, admission, the join search, KV
retention, GEMMs, and attention are identical across configurations.
The non-quail paths live in a Pipeline subclass inside the ablation
cell; the engine is unchanged.

| kernel source | between-GEMM kernels | merge_quant-path join merge |
|---|---|---|
| quail | our three fused Triton kernels | our fused merge+quant Triton kernel |
| vllm_ops | vLLM's ops one by one, unfused (fused_add_rms_norm + quant, silu_and_mul + quant, per-head norms + rotary as five launches) | vLLM's merge_attn_states + separate group quant |
| vllm_compiled | the set stock's compiled graph was measured to run: torch.compile over the native add+rms_norm, silu*mul, and q/k-norm+rope math (vLLM's Inductor settings, dynamic token count), plus the same standalone group-quant per GEMM input | vLLM's merge_attn_states + separate group quant (stock never merges inside its compiled graph; its cascade merge lives in the attention backend) |

A per-kernel probe (`/results/ablations/kernel_source_probe.json`)
checked the wiring before the measured runs: the compiled segments
hold one graph across token counts, the add+rms_norm segment folds
the residual write into one generated kernel plus the quant launch
(no extra copy), dequantized outputs sit within one fp8 rounding step
of our kernels', and IMDB-1 plus BIO-2 at scale factor 0.01 return
near-identical rows through all three sources.

## The two queries

Both from the QUAIL-B catalog at scale factor 0.1 (reviews 5,000;
reports 500; terms 1,127):

- IMDB-7: three filters over the reviews table (F1 -> F4 -> F5, no
  join) — the unified attention path, 1.80M fresh tokens.
- BIO-2: the reports x terms join, 563,500 document pairs — the
  merge_quant attention path, 10.37M fresh tokens.

Two repetitions per kernel source after one unmeasured warm run;
repetitions agreed within 0.2% everywhere. The tables use the second
repetition, matching the cell's comparison rows.

## Predictions (stated before the run)

- IMDB-7, vllm_ops: +25 to +32% wall. Measured: +29.7%. Correct.
- IMDB-7, vllm_compiled: +22 to +29% wall, reasoning that the
  compiled set keeps all four standalone group-quant launches per
  layer and recovers only the q/k segment. Measured: +26.9%. Correct.
- BIO-2, vllm_ops: +26 to +33% wall. Measured: +25.6%, just under
  the band.
- BIO-2, vllm_compiled: +23 to +30%. Measured: +23.3%, at the bottom
  edge.

One anchor in the stored prediction text was stale: it cited BIO-2 at
32 s from the QUAIL-B sf0.1 report, which predates the BioDEX corpus
scale-up (the benchmark's cache schema version 5). Today's BIO-2 is
563,500 pairs and runs about 129 s on the quail source. The
percentage bands, which is what the predictions were about, held.

## Results

Second repetition, one H100, `$3.9492/hour`
(`quail.bench.evaluate.H100_USD_PER_HOUR`). Query time excludes model
startup, as everywhere in this repo.

IMDB-7 — unified attention path, 5,000 documents:

| kernel source | query time | us/fresh token | documents/s | $/query |
|---|---|---|---|---|
| quail (unified) | 15.0 s | 8.36 | 333.1 | $0.0165 |
| vllm_compiled (unified) | 19.1 s | 10.60 (+27%) | 262.3 | $0.0209 |
| vllm_ops (unified) | 19.5 s | 10.84 (+30%) | 256.7 | $0.0214 |

BIO-2 — merge_quant attention path, 563,500 document pairs:

| kernel source | query time | us/fresh token | pairs/s | $/query |
|---|---|---|---|---|
| quail (merge_quant) | 128.9 s | 12.42 | 4,372.6 | $0.1414 |
| vllm_compiled (merge_quant) | 158.9 s | 15.31 (+23%) | 3,547.4 | $0.1743 |
| vllm_ops (merge_quant) | 161.8 s | 15.60 (+26%) | 3,482.0 | $0.1775 |

- The compiled set beats the unfused ops by only 0.24-0.29 us/token —
  about a tenth of its gap to quail. torch.compile is not where the
  speed is for this model.
- The join-path gap exceeds the filter-path gap by 0.64-0.69
  us/token on both vLLM sources. That is the price of composing the
  join merge from vLLM's pieces (gather the rows with cached
  context, merge_attn_states, scatter back, quantize) against our
  one fused merge+quant kernel.
- BIO-2's fresh-token count is identical across sources (a
  single-stage join evaluates every pair); IMDB-7's differs by under
  0.07% because survivor sets differ slightly between stages.

Output rows: IMDB-7 returned 727 / 730 / 747 surviving documents
(quail / vllm_ops / vllm_compiled), with 83-86 documents in the
symmetric difference — about 1.7% of the 5,000 scanned documents flip
at thin stage margins between kernel stacks. BIO-2 returned 116,295 /
111,263 / 119,608 pairs, a 6.5-6.7% pair flip rate; the BioDEX
term-matching task sits near the 4B model's floor with TRUE/FALSE
margins close to zero (the 32B model resolves it), so small rounding
differences between kernel stacks move many pairs. No kernel source
is more accurate than another here; the flips measure margin
thinness, not correctness.

### Where the GPU time goes

GPU kernel microseconds per fresh token on a profiled IMDB-7 run
(`/results/ablations/kernel_source_profile.json`).
Figure: `plots/kernel_source_profile.png`.

| category | quail (unified) | vllm_ops (unified) | vllm_compiled (unified) |
|---|---|---|---|
| matrix multiplies (DeepGEMM) | 5.58 | 5.25 | 5.32 |
| attention (FlashAttention-3) | 0.70 | 0.65 | 0.66 |
| our fused Triton kernels | 1.49 | — | — |
| vLLM norm, rotary, silu ops | 0.00 | 2.58 | 0.00 |
| Inductor-generated kernels | — | — | 2.76 |
| standalone group-quant | 0.41 | 1.71 | 1.77 |
| copies | 0.22 | 0.70 | 0.22 |
| **total** | **8.41** | **10.89** | **10.73** |

Readings:

- The standalone group-quant is the cost torch.compile does not
  remove: 1.77 us/token in the compiled set against quail's 0.41
  (our one remaining quant, the attention output feeding o_proj).
  Fusing the other three quant passes into their producers is most
  of our win.
- Inductor's fusion is a wash on the ops themselves: it replaces
  2.58 us/token of vLLM norm, rotary, and silu*mul kernels with 2.76
  us/token of generated kernels — its silu*mul kernel alone is about
  twice as slow as vLLM's hand-written CUDA op. What torch.compile
  actually recovers is the data movement around the unfused
  sequence: the copies bucket falls from 0.70 to 0.22 (mostly the
  two contiguous copies per layer the vLLM q/k path needs).
- Small-kernel work in total (everything but matmuls and attention):
  quail 2.12, vllm_compiled 4.75, vllm_ops 4.99 us/token. The
  GPU-time deltas match the measured wall deltas within a few
  percent — all three configurations are GPU-bound.
- The matmul bucket reads about 0.3 us/token higher on the quail row
  even though all three rows launch the same DeepGEMM kernels the
  same number of times (3,024 launches, identical shapes) on
  near-identical token counts, sequentially on the same GPU in the
  same container. The cause was measured, not guessed: the SM clock,
  sampled every 50 ms during an unprofiled run of each source. All
  three configurations sit at the 700 W power limit (686-696 W
  mean), and the GPU's power governor holds the quail run at a mean
  1427 MHz against 1645 (vllm_ops) and 1621 (vllm_compiled). Packing
  the same work into fewer, denser compute kernels leaves the
  governor less headroom, so the same matmul kernel takes about 6%
  longer per launch under quail. Pinning the clock to equalize the
  comparison was refused by the driver in this environment (recorded
  in the data file). The effect works against the fused kernels in
  this table, and the headline wall-clock numbers come from separate
  unprofiled runs.

## Scope notes

- One model (Qwen3 4B fp8). The per-token saving is roughly constant
  in model size while the matmul work grows about 8x at 32B, so the
  relative win shrinks accordingly. Not re-measured here.
- Stock vLLM at -O2 also captures CUDA graphs for decode-sized
  batches. These queries are large-chunk prefill, where CUDA graphs
  do not apply, and the packed executor launches eagerly in every
  configuration — graph capture is outside this comparison.
- The vllm_compiled source reproduces stock's kernel set inside our
  executor; it is not stock vLLM end to end. The stock engine
  baseline for QUAIL-B is the stock-vllm-joins report and the
  QUAIL-B sf0.1 report.

## How to reproduce

    uv run modal run experiments/vllm_compiled_kernels.py::run_stock_kernels
    uv run modal run experiments/vllm_compiled_kernels.py::run_probe
    uv run modal run experiments/vllm_compiled_kernels.py::run_queries
    uv run modal run experiments/vllm_compiled_kernels.py::run_profile

Modal app `quail-milestone1`, one H100 per cell. Function calls:
queries `fc-01M17PPYE6361DV4C9N57MAWBF`, profile
`fc-01M17PQ42KN5YS8ABRS3K2G06K`, stock inventory
`fc-01M17PQ620HWV0579W35VWJTFS`, probe
`fc-01M17NG322VDE687JR8RQY444E`.

## Data files

All on the `quail-results` volume:

- `/results/ablations/kernel_source_imdb7.json` — IMDB-7 walls,
  rates, rows, comparisons.
- `/results/ablations/kernel_source_bio2.json` — BIO-2 walls, rates,
  rows, comparisons.
- `/results/ablations/kernel_source_profile.json` — per-category GPU
  time and launch counts for the three sources on IMDB-7.
- `/results/ablations/kernel_source_stock.json` — stock vLLM's
  resolved compilation config and kernel inventory.
- `/results/ablations/kernel_source_probe.json` — per-kernel parity,
  compiled-segment launch counts, sf 0.01 row agreement.

Plots are rebuilt by `reports/make_kernel_source_plots.py` (the
`modal volume get` commands are in its docstring; it also prints the
throughput and $/query table above). The kernel diagram is
`reports/kernel_paths_diagram.html`, rendered to
`plots/kernel_paths.png`. Local tee logs: `results/kernel_source_*.log`.
