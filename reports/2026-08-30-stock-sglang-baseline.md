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

Two join submission orders were measured for SGLang, because its
prefix cache turned out to reward a different order than vLLM's:

- anchor-major, the stock vLLM order: all pairs of one anchor
  document, then the next anchor.
- suffix-major in anchor tiles: anchors are grouped into tiles that
  fit in half the KV pool; within a tile, every anchor runs once per
  partner document (a1 b1, a2 b1, ..., a1 b2, a2 b2, ...).

The runs cover the two representative queries at scale factor 0.1
with Qwen3 4B fp8 on one H100!:

- BIO-2: one join, REACTION over 500 reports x 1,127 terms =
  563,500 pairs.
- IMDB-3: one filter (F1 over 5,000 reviews), then one join,
  DISCUSS_ASPECT over the surviving reviews x 12 aspects.

## Result

- Neither submission order wins both queries, and the two orders
  bracket stock vLLM:
  - BIO-2: anchor-major SGLang took 1,204.2 seconds (1.27 times
    faster than stock vLLM's 1,532.3); tiled SGLang took
    1,384.5 seconds (1.11 times faster than vLLM, 15% slower than
    anchor-major SGLang).
  - IMDB-3: anchor-major SGLang took 213.2 seconds (4.05 times
    slower than stock vLLM's 52.7); tiled SGLang took 83.9 seconds
    (2.54 times faster than anchor-major SGLang, still 1.59 times
    slower than vLLM).
- The tiled order recovered the prefix cache as designed. On IMDB-3
  the join's fresh prompt tokens fell from 17.0M (6.5% cached) to
  1.94M (89.3% cached, matching vLLM's 88.4%). On BIO-2 they fell
  from 26.3M to 7.0M (99.70% cached, above vLLM's 99.55%) — and the
  query still got slower, so BIO-2's remaining time is not prefill
  compute.
- The two SGLang orders returned bit-identical answers on both
  queries (4,230 filter survivors; 114,869 and 11,351 TRUE pairs),
  so the reorder is free of answer effects. The engines still
  disagree with each other: stock vLLM passed 4,374 reviews and
  returned 121,406 TRUE pairs on BIO-2.
- Answer accuracy against the same Qwen3 32B ground truth labels was
  higher on SGLang for both queries: 82.03% vs 80.92% on BIO-2 and
  79.94% vs 78.97% on IMDB-3, identically for both orders.
- The filter stage is order-free and stayed a tie: 22.7 seconds on
  SGLang and 21.2 seconds on vLLM for the same 5,000 reviews.
- Model startup, excluded from query time: 314 to 355 seconds for
  SGLang (warm kernel caches), compared with 143.8 to 235.2 seconds
  for the four vLLM family containers.

## Main metrics

Figure: plots/stock_sglang_vs_stock_vllm.png

| Query | System | Query time (s) | Document pairs/s | $/query | Answer accuracy |
|---|---|---:|---:|---:|---:|
| BIO-2 | stock vLLM, anchor-major | 1,532.25 | 367.8 | $1.6809 | 80.92% |
| BIO-2 | stock SGLang, anchor-major | 1,204.21 | 467.9 | $1.3210 | 82.03% |
| BIO-2 | stock SGLang, tiled | 1,384.50 | 407.0 | $1.5188 | 82.03% |
| IMDB-3 | stock vLLM, anchor-major | 52.65 | 996.9 | $0.0578 | 78.97% |
| IMDB-3 | stock SGLang, anchor-major | 213.19 | 238.1 | $0.2339 | 79.94% |
| IMDB-3 | stock SGLang, tiled | 83.88 | 605.1 | $0.0920 | 79.94% |

Document pairs/s divides each system's own evaluated join pairs
(563,500 for BIO-2; 52,488 for vLLM and 50,760 for SGLang on IMDB-3,
because the filters passed different survivor sets) by full query
runtime. $/query uses $3.9492 per H100! hour and excludes startup.

## Prediction, first run (anchor-major)

Stated before the run. The stock vLLM constants were BIO-2 in
1,532.3 seconds and IMDB-3 in 52.7 seconds (2026-08-29 run), and the
assumption was that SGLang does equivalent host work and equivalent
prefix caching per request:

- BIO-2 finishes in 1,100 to 2,100 seconds. Hit: 1,204.2 seconds.
- IMDB-3 finishes in 40 to 75 seconds. Missed by 2.8x: 213.2
  seconds. The caching assumption was wrong for this query shape.
- The F1 filter passes 4,374 +/- 10 of 5,000 reviews. Missed: 4,230,
  144 fewer than stock vLLM.
- Join TRUE counts land within 1% of stock vLLM's. Missed: BIO-2
  returned 5.4% fewer TRUE pairs.
- Answer accuracy lands within 0.5 points of stock vLLM's. Missed in
  SGLang's favor: +1.11 points on BIO-2, +0.97 on IMDB-3.

## Prediction, second run (tiled order)

Stated before the run:

- IMDB-3 drops to 50-85 seconds, join fresh tokens from 17.0M to
  about 3.5M. Time hit (83.9 seconds); fresh tokens beat the
  prediction at 1.94M, because 1-token pages recompute only the
  partner document and answer cue once the anchor is cached.
- BIO-2 drops to 950-1,200 seconds, fresh tokens from 26.3M to about
  13M. Missed in both directions: fresh fell further than predicted
  (7.0M) and the query still got slower (1,384.5 seconds).
- TRUE counts within 0.5% of the anchor-major SGLang run and
  accuracy within 0.3 points. Hit exactly: both orders returned
  identical answers.

## What the numbers mean

The IMDB-3 gap and its fix confirm the cache mechanism:

- SGLang's radix cache stores a prompt's KV when a request finishes.
  Anchor-major submission puts all 12 pairs of one review in flight
  before the first finishes, so the review is recomputed for almost
  every pair: 335 fresh tokens per pair, 6.5% of join prompt tokens
  cached, a 191.7 second join. vLLM does not have this problem
  because it reuses computed 16-token blocks without waiting for the
  request that produced them to finish (88.4% cached, 41 fresh
  tokens per pair, 31.5 seconds).
- The tiled order removes the co-admission: within a tile no two
  in-flight pairs share an uncached anchor, and from the second
  suffix pass on, every anchor is cached. IMDB-3's join fell to 38
  fresh tokens per pair and 61.1 seconds. The tile bound matters:
  suffix-major over all 4,230 anchors at once (1.5M anchor tokens
  against a 415k-token pool) would evict every anchor before its
  next use and gain nothing.
- Reordering inside SGLang's scheduler would not have achieved this.
  The cache-aware policy (`lpm`) is no longer the default (`fcfs`
  is), it falls back to FCFS whenever more than 128 requests wait,
  and sorting by cached-prefix length cannot delay a pair so that
  its sibling can populate the cache first. The order had to come
  from the client.

BIO-2 shows the limit of ordering alone:

- The tiled order raised BIO-2's cache hit rate to 99.70% and cut
  fresh tokens from 26.3M to 7.0M, yet the query slowed from 1,204.2
  to 1,384.5 seconds. Prefill compute is therefore not what BIO-2's
  1,200 seconds buy: at 2.1 milliseconds per pair, per-request
  scheduling and answer work dominates.
- The leading explanation for the 15% regression is batch shape.
  With about 9 fresh tokens per pair, a 25,305-token budget admits
  thousands of requests per batch instead of about 540, and each
  batch's answer step materializes float32 logits and a logit-bias
  tensor for every request over the 151,936-token vocabulary, while
  the attention kernel pads each request's few query tokens to a
  full tile. This is unsettled; per-step logs (a rerun with
  `log_level=info`) showing batch composition and step times for
  both orders would settle it.
- The answer bits were identical between the two SGLang orders on
  both queries, so batch shape moves time, not answers. The
  remaining answer divergence is between engines (different fp8
  kernels; about 1.2% of BIO-2 pairs net flipped, accuracy slightly
  up on SGLang), not between submission orders.

The runner keeps the tiled order as its default: it caps the
downside on queries with few pairs per anchor (IMDB-3 was 4.05 times
slower than vLLM anchor-major, 1.59 times slower tiled) and costs
15% on deep-anchor queries like BIO-2. Anchor-major remains
selectable through `join_submission`.

## Configuration

Both baselines were configured, per the project rule that a baseline
gets the analytically equivalent settings:

| Setting | Stock vLLM 0.26.0 | Stock SGLang 0.5.18 |
|---|---|---|
| GPU memory fraction | `gpu_memory_utilization=0.91` | `mem_fraction_static=0.78` (see below) |
| Max concurrent requests | `max_num_seqs=4096` | `max_running_requests=4096` |
| Scheduled token budget | `max_num_batched_tokens=25305` | `chunked_prefill_size=25305`, `max_prefill_tokens=25305` |
| Prefix caching | on, 16-token blocks | radix cache on (default), 1-token pages |
| Join pair order | anchor-major | anchor-major and suffix-major tiled, both measured |
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

The tiled join order packs anchors greedily into tiles whose anchor
tokens plus one suffix stay under half the measured KV pool
(207,512 tokens here): about 528 reviews per tile on IMDB-3 and 50
reports per tile on BIO-2. Half the pool bounds a tile because
in-flight suffixes and the previous tile's leftovers share the pool
with the tile's anchors. Answers are returned in anchor-major order
either way, so the shared bookkeeping is unchanged
(`baselines.stock.suffix_major_tiled_order`).

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
  415,024-token KV pool, 13.4% smaller than vLLM's 479,248. The
  tiled order sizes its tiles from the measured pool, so the pool
  difference does not change what gets cached.

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
  request completes; vLLM reuses computed 16-token blocks of
  requests that are still running. This is why the two engines get
  different orders.
- The vLLM image sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`;
  the SGLang image does not, because SGLang allocates its KV pool
  statically inside `mem_fraction_static`.
- The baselines ran in different H100! containers on different days.
  The stock vLLM numbers come from the 2026-08-29 family run.

## Source data

- Stock SGLang, tiled order, function call
  `fc-01M18ENA18EMP18BF43RXQVZK1`:
  `/results/stock_sglang/2026-08-30_042550_e5f6d2e8/summary.json`
- Stock SGLang, anchor-major order, function call
  `fc-01M18BR9BP6A74BT3TK5EFM9Z3`:
  `/results/stock_sglang/2026-08-30_033502_05712d88/summary.json`
- Stock vLLM (2026-08-29 family run):
  `/results/stock_vllm/20260829T185407Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json`

All runs scored accuracy against ground truth collection
`gt_363b5ab570635c33894e1a030c21f57e` labels on corpus
`c_3bd14ed0758287cba9d88fb68de8b7b8` (the vLLM run recorded the same
label sets under their earlier collection id
`gt_02ffa2a5720006e8236aa993760e9e29`).

The plot script `reports/make_stock_sglang_baseline_plots.py` takes a
work directory holding the three pulled files; its docstring has the
`modal volume get` commands.
