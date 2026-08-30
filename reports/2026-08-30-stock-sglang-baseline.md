# Stock SGLang baseline on BIO-2 and IMDB-3

Date: 2026-08-30.

## What this is

A second stock engine baseline next to stock vLLM. SGLang is an open
source inference engine with its own scheduler and its own prefix
cache (RadixAttention). The new runner `baselines/stock_sglang/run.py`
submits exactly the same work as the stock vLLM runner: one request
per document per filter stage (stage-major waves) and one request per
document pair for joins (full cross product). It imports the query
definitions, prompt construction, and per-query bookkeeping from
`baselines.stock_vllm.run`, so the two baselines cannot drift apart.

The run covers the two representative queries at scale factor 0.1 with
Qwen3 4B fp8 on one H100!:

- BIO-2: one join, REACTION over 500 reports x 1,127 terms =
  563,500 pairs.
- IMDB-3: one filter (F1 over 5,000 reviews), then one join,
  DISCUSS_ASPECT over the surviving reviews x 12 aspects.

## Configuration

Both baselines were configured, per the project rule that a baseline
gets the analytically equivalent settings:

| Setting | Stock vLLM 0.26.0 | Stock SGLang 0.5.18 |
|---|---|---|
| GPU memory fraction | `gpu_memory_utilization=0.91` | `mem_fraction_static=0.78` (see below) |
| Max concurrent requests | `max_num_seqs=4096` | `max_running_requests=4096` |
| Scheduled token budget | `max_num_batched_tokens=25305` | `chunked_prefill_size=25305`, `max_prefill_tokens=25305` |
| Prefix caching | on, 16-token blocks | radix cache on (default), 1-token pages |
| Answer decoding | greedy, `allowed_token_ids` = the 8 TRUE/FALSE first tokens | greedy, `logit_bias=+1000` on the same 8 token ids |
| Scheduling policy | first come, first served | first come, first served (default) |
| Checkpoint | `Qwen/Qwen3-4B-FP8` | `Qwen/Qwen3-4B-FP8` |
| KV dtype | BF16 | BF16 |

The logit bias reproduces vLLM's restricted decoding exactly: SGLang
adds the bias to float32 logits before its greedy argmax, so adding
the same constant to all eight allowed ids picks the same token as
restricting the argmax to those ids, as long as no other token's raw
logit is 1,000 higher. Raw logit gaps are two orders of magnitude
smaller than that.

The memory fraction cannot be copied literally, and finding that out
took three crashed attempts on BIO-2:

- The two fractions mean different things. vLLM sizes its KV pool
  after profiling a full-size forward, including the final-position
  logits for `max_num_seqs` requests, so
  `gpu_memory_utilization=0.91` already accounts for the activation
  working set. SGLang's `mem_fraction_static` reserves that fraction
  for weights plus KV only; everything else must fit in the
  remainder.
- At `mem_fraction_static=0.91`, SGLang's prefill CUDA graph capture
  (91 shapes up to the 25,305-token budget, each retaining about
  130 MB) ran the remaining 6.3 GB to zero and crashed the boot. The
  runner therefore sets `disable_prefill_cuda_graph=True`. That costs
  little here: every batch packs hundreds of cached-prefix requests,
  so per-batch launch overhead is already amortized, and this
  workload never reaches a decode batch because every request
  generates its one token during the prefill forward.
- Still at 0.91 (a 489,403-token pool), the BIO-2 join ran out of GPU
  memory fourteen minutes in. At 0.85 (a 455,074-token pool) it ran
  out again, and the failing allocation identified the mechanism: it
  was 2.32 GiB, exactly a float32 logits tensor for 4,096 requests
  over the 151,936-token vocabulary. SGLang's 1-token-page radix
  cache leaves so few fresh tokens per pair that its scheduler packs
  batches up to `max_running_requests=4096`, and each such batch's
  answer step holds those logits plus an equally sized logit-bias
  tensor. On top of that, 6.5 GiB of the GPU sat in non-PyTorch
  allocations (kernel workspaces). vLLM never sees this peak: its
  16-token-block cache leaves about 19 fresh tokens per pair, so its
  batches stop near 1,350 requests, and its profiling reserved the
  logits memory anyway.
- The run uses `mem_fraction_static=0.78`: 79.18 GiB total minus
  about 6.5 GiB non-PyTorch, about 7 GiB answer-step and forward
  activations, and a safety margin. That gives SGLang a
  415,024-token KV pool, 13.4% smaller than vLLM's
  479,248. The two queries submit join pairs anchor by anchor, so the
  live prefix working set stays far below either pool size and the
  difference does not change what gets cached.

Differences that remain, reported rather than hidden:

- SGLang's prefix cache matches at page granularity
  (1-token pages); vLLM reuses complete 16-token blocks.
- The vLLM image sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`;
  the SGLang image does not, because SGLang allocates its KV pool
  statically inside `mem_fraction_static`.
- Attention backend: vLLM uses its default FlashAttention path;
  SGLang used fa3 (its default on H100!).
- The two runs used different H100! containers on different days. The
  stock vLLM numbers come from the 2026-08-29 family run.

## Prediction

Stated before the run. The stock vLLM constants are BIO-2 in
1,532.3 seconds and IMDB-3 in 52.7 seconds (2026-08-29 run). BIO-2's
wall time is dominated by per-request host work: 563,500 requests at
2.72 ms each, with only 10.5M fresh prompt tokens. SGLang does the
equivalent host work per request (zmq transfer, radix match over a
~4,100-token cached prefix, batch assembly), so the prediction is a
wide band around parity:

- BIO-2 finishes in 1,100 to 2,100 seconds (0.7x to 1.4x of stock
  vLLM's 1,532 seconds).
- IMDB-3 finishes in 40 to 75 seconds (0.75x to 1.4x of stock vLLM's
  52.7 seconds).
- The F1 filter passes 4,374 +/- 10 of 5,000 reviews (stock vLLM
  passed 4,374; both engines greedy-decode the same prompts, and fp8
  kernel differences flip only near-tie answers).
- Join TRUE counts land within 1% of stock vLLM's 121,406 (BIO-2) and
  10,767 (IMDB-3).
- Answer accuracy lands within 0.5 points of stock vLLM's 80.92%
  (BIO-2) and 78.97% (IMDB-3), against the same ground truth labels.

## Result

RESULT_TBD

## Main metrics

Figure: plots/stock_sglang_vs_stock_vllm.png

METRICS_TBD

## What the numbers mean

MEANING_TBD

## Source data

- Stock SGLang, function call `fc-01M18BR9BP6A74BT3TK5EFM9Z3`:
  `/results/stock_sglang/LABEL_TBD/summary.json`
- Stock vLLM (2026-08-29 family run):
  `/results/stock_vllm/20260829T185407Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json`

The plot script `reports/make_stock_sglang_baseline_plots.py` takes a
work directory holding the two pulled files; its docstring has the
`modal volume get` commands.
