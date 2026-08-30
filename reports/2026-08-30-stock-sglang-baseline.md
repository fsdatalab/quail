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

## Result

- BIO-2: stock SGLang took 1,204.2 seconds, compared with stock
  vLLM's 1,532.3 seconds. Stock SGLang was 1.27 times faster.
- IMDB-3: stock SGLang took 213.2 seconds, compared with stock
  vLLM's 52.7 seconds. Stock SGLang was 4.05 times slower.
- The filter stage alone was a tie: 21.5 seconds on SGLang and
  21.2 seconds on vLLM for the same 5,000 reviews.
- The engines disagree on borderline answers. The F1 filter passed
  4,230 reviews on SGLang and 4,374 on vLLM. BIO-2 returned 114,869
  TRUE pairs on SGLang and 121,406 on vLLM.
- Answer accuracy against the same Qwen3 32B ground truth labels was
  higher on SGLang for both queries: 82.03% vs 80.92% on BIO-2 and
  79.94% vs 78.97% on IMDB-3.
- Model startup, excluded from query time: 314.4 seconds for SGLang
  (with warm kernel caches), compared with 143.8 to 235.2 seconds
  for the four vLLM family containers.

## Main metrics

Figure: plots/stock_sglang_vs_stock_vllm.png

| Query | System | Query time (s) | Document pairs/s | $/query | Answer accuracy |
|---|---|---:|---:|---:|---:|
| BIO-2 | stock vLLM | 1,532.25 | 367.8 | $1.6809 | 80.92% |
| BIO-2 | stock SGLang | 1,204.21 | 467.9 | $1.3210 | 82.03% |
| IMDB-3 | stock vLLM | 52.65 | 996.9 | $0.0578 | 78.97% |
| IMDB-3 | stock SGLang | 213.19 | 238.1 | $0.2339 | 79.94% |

Document pairs/s divides each system's own evaluated join pairs
(563,500 for BIO-2; 52,488 for vLLM and 50,760 for SGLang on IMDB-3,
because the filters passed different survivor sets) by full query
runtime. $/query uses $3.9492 per H100! hour and excludes startup.

## Prediction

Stated before the run. The stock vLLM constants were BIO-2 in
1,532.3 seconds and IMDB-3 in 52.7 seconds (2026-08-29 run). BIO-2's
wall time is dominated by per-request host work, and the assumption
was that SGLang does equivalent host work and equivalent prefix
caching per request. The prediction:

- BIO-2 finishes in 1,100 to 2,100 seconds. Hit: 1,204.2 seconds.
- IMDB-3 finishes in 40 to 75 seconds. Missed by 2.8x: 213.2
  seconds. The caching assumption was wrong for this query shape; the
  next section explains the mechanism.
- The F1 filter passes 4,374 +/- 10 of 5,000 reviews. Missed: 4,230,
  144 fewer than stock vLLM.
- Join TRUE counts land within 1% of stock vLLM's. Missed: BIO-2
  returned 5.4% fewer TRUE pairs.
- Answer accuracy lands within 0.5 points of stock vLLM's. Missed in
  SGLang's favor: +1.11 points on BIO-2, +0.97 on IMDB-3.

## What the numbers mean

The split result comes from when each engine's prefix cache becomes
usable, not from raw speed:

- SGLang's radix cache stores a prompt's KV when a request finishes.
  All pairs of one anchor document are submitted consecutively, so
  every pair that starts while the anchor's first pair is still
  running recomputes the anchor document from scratch.
- IMDB-3 is the worst case: each review has only 12 pairs, and all 12
  are in flight before the first finishes. SGLang reused only 1.18M
  of the join's 18.19M prompt tokens (6.5%), recomputing 335 tokens
  per pair. vLLM reuses completed 16-token blocks of still-running
  requests, reused 88.4%, and recomputed 41 tokens per pair. That is
  why SGLang's join took 191.7 seconds against vLLM's 31.5.
- BIO-2 mostly hides the same effect: each report has 1,127 pairs, so
  only the pairs admitted before the anchor's first completion pay
  the penalty. SGLang still computed 2.5 times the fresh tokens
  (26.3M vs 10.5M) but won anyway, 2.14 ms per pair against vLLM's
  2.72 ms, because its per-request scheduling overhead is lower and
  the extra prefill compute is cheap at this scale.
- The filter tie (21.5 vs 21.2 seconds) supports that reading:
  filters share no document prefix, so both engines prefill about
  1.75M fresh tokens, and they do it at the same rate.
- The answer divergence is fp8 kernel noise at the decision boundary,
  not degradation: the engines use different fp8 matrix kernels and
  different batch shapes, which shifts logits on near-tie pairs.
  About 1.2% of BIO-2 pairs net flipped. Accuracy against ground
  truth went up slightly on both queries, so neither engine is
  cleanly more correct; they disagree on pairs the 4B model finds
  ambiguous either way.

The IMDB-3 number is a property of stock SGLang's cache design on
this submission pattern, not of the shared submission code: the same
ordered pair stream produced an 88.4% hit rate on vLLM. A rerun that
throttles submission so each anchor's first pair completes before its
siblings start would recover the reuse but would no longer be the
stock submission strategy, so it was not done here.

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
| Attention backend | vLLM default FlashAttention | fa3 (SGLang default on H100!) |

The logit bias reproduces vLLM's restricted decoding exactly: SGLang
adds the bias to float32 logits before its greedy argmax, so adding
the same constant to all eight allowed ids picks the same token as
restricting the argmax to those ids, as long as no other token's raw
logit is 1,000 higher. Raw logit gaps are two orders of magnitude
smaller than that.

The memory fraction cannot be copied literally, and finding that out
took two crashed runs:

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
  over the 151,936-token vocabulary. SGLang packs batches up to
  `max_running_requests=4096` (each pair adds few fresh tokens), and
  each such batch's answer step holds those logits plus an equally
  sized logit-bias tensor. On top of that, 6.5 GiB of the GPU sat in
  non-PyTorch allocations (kernel workspaces). vLLM never sees this
  peak: its batches stop near 1,350 requests, and its profiling
  reserved the logits memory anyway.
- The run uses `mem_fraction_static=0.78`: 79.18 GiB total minus
  about 6.5 GiB non-PyTorch, about 7 GiB answer-step and forward
  activations, and a safety margin. That gives SGLang a
  415,024-token KV pool, 13.4% smaller than vLLM's 479,248. The two
  queries submit join pairs anchor by anchor, so the live prefix
  working set stays far below either pool size and the difference
  does not change what gets cached.

One driver-side change was also needed to finish at all. Handing
SGLang all 563,500 BIO-2 pairs in one `generate()` call creates one
asyncio task per request in the driver process; the Modal health
heartbeat thread then starved for over twenty minutes and Modal
warned it would kill the container. The client therefore submits in
slices of 16,384 requests with a one second pause between slices.
Each slice is still four times deeper than `max_running_requests`,
so the engine's queue never runs dry inside a slice; the pauses and
slice boundaries add at most a few seconds to a query. vLLM's own
in-process `generate()` needs no equivalent because it has no
separate driver.

Differences that remain, reported rather than hidden:

- SGLang's prefix cache matches at 1-token pages but only after a
  request completes; vLLM reuses completed 16-token blocks of
  requests that are still running. This is the IMDB-3 story above.
- The vLLM image sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`;
  the SGLang image does not, because SGLang allocates its KV pool
  statically inside `mem_fraction_static`.
- The two baselines ran in different H100! containers on different
  days. The stock vLLM numbers come from the 2026-08-29 family run.

## Source data

- Stock SGLang, function call `fc-01M18BR9BP6A74BT3TK5EFM9Z3`:
  `/results/stock_sglang/2026-08-30_033502_05712d88/summary.json`
- Stock vLLM (2026-08-29 family run):
  `/results/stock_vllm/20260829T185407Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json`

Both runs scored accuracy against ground truth collection
`gt_363b5ab570635c33894e1a030c21f57e` labels on corpus
`c_3bd14ed0758287cba9d88fb68de8b7b8` (the vLLM run recorded the same
label sets under their earlier collection id
`gt_02ffa2a5720006e8236aa993760e9e29`).

The plot script `reports/make_stock_sglang_baseline_plots.py` takes a
work directory holding the two pulled files; its docstring has the
`modal volume get` commands.
