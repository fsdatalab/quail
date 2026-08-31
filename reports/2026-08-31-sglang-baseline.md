# SGLang baseline on BIO-2 and IMDB-3

Date: 2026-08-31. Supersedes the 2026-08-30 numbers, which ran the
same submission strategy on a costlier per-request configuration.

## What this is

An SGLang baseline next to the two vLLM baselines. SGLang is an open
source inference engine with its own scheduler and its own prefix
cache (RadixAttention). The runner `baselines/stock_sglang/run.py`
has one configuration, `pipelined_sglang`, the SGLang counterpart of
pipelined vLLM:

- Filters chain per document: a document holds one of `doc_cap`
  token-budget admission slots for its whole chain, and a TRUE answer
  sends its next stage while other documents are still on earlier
  stages. SGLang's scheduler runs in a separate process with no
  synchronous add_request/step surface, so the chain advances in
  waves of blocking `generate()` calls; no part of the client uses
  asyncio, matching the synchronous vLLM clients.
- Joins run the full cross product, one request per pair, submitted
  suffix-major in anchor tiles (`suffix_major_tiled_order`) — the
  order SGLang's cache can exploit, where vLLM keeps anchor-major.
- The engine is configured to spend as little host time per request
  as SGLang allows from the client side: 16-token cache pages,
  no tokenizer or detokenizer in the request path
  (`skip_tokenizer_init`), and token-id-only exchange. The section
  on how the configuration was reached has the measurements.
- The prompts, query definitions, and per-query bookkeeping are
  imported from `baselines.stock_vllm.run`, so the baselines measure
  exactly the same work.

The run covers the two representative queries at scale factor 0.1
with Qwen3 4B fp8 on one H100!: BIO-2 (one join, REACTION over 500
reports x 1,127 terms = 563,500 pairs) and IMDB-3 (one filter, F1
over 5,000 reviews, then one join, DISCUSS_ASPECT over the survivors
x 12 aspects).

## Result

Figure: plots/sglang_baseline_comparison.png

| Query | System | Query time (s) | Document pairs/s | $/query | Answer accuracy |
|---|---|---:|---:|---:|---:|
| BIO-2 | SoL estimate | 62.01 | 9,087 | $0.0680 | not applicable |
| BIO-2 | Quail | 128.22 | 4,395 | $0.1406 | see the QuailB report |
| BIO-2 | stock vLLM | 1,532.25 | 367.8 | $1.6809 | 80.92% |
| BIO-2 | pipelined vLLM | 1,427.45 | 394.8 | $1.5659 | 80.91% |
| BIO-2 | pipelined SGLang | 1,028.73 | 547.8 | $1.1285 | 82.03% |
| IMDB-3 | SoL estimate | 9.56 | 5,024 | $0.0105 | not applicable |
| IMDB-3 | Quail | 32.79 | 1,603 | $0.0360 | see the QuailB report |
| IMDB-3 | stock vLLM | 52.65 | 996.9 | $0.0578 | 78.97% |
| IMDB-3 | pipelined vLLM | 48.21 | 1,088.8 | $0.0529 | 78.97% |
| IMDB-3 | pipelined SGLang | 66.98 | 757.8 | $0.0735 | 79.94% |

- BIO-2: pipelined SGLang is the fastest of the three baseline
  engines — 1.39 times faster than pipelined vLLM and 1.49 times
  faster than stock vLLM. Quail is still 8.0 times faster than it,
  and the SoL estimate 16.6 times.
- IMDB-3: pipelined SGLang is the slowest measured system — 1.39
  times slower than pipelined vLLM and 2.0 times slower than Quail's
  32.8 seconds. The gap to pipelined vLLM was 1.65x on the 2026-08-30
  configuration.
- The filter is now a tie: 17.6 seconds for the same 5,000 reviews
  pipelined vLLM filters in 17.3.
- Its answer accuracy against the shared Qwen3 32B ground truth is
  the highest of the three engines on both queries (82.03% and
  79.94%).
- Model startup, excluded from query time: 307.6 seconds (warm
  kernel caches), compared with 143.8 to 235.2 seconds for the vLLM
  family containers.

Each system's pairs/s divides its own evaluated join pairs by its
query time. On BIO-2 every system evaluates the same 563,500 pairs.
On IMDB-3 the filters differ, so the pair counts do too: 48,048 for
the SoL estimate (4,004 expected survivors), 52,560 for Quail (its
filter passed 4,380 reviews), 52,488 for the vLLM baselines (4,374),
and 50,760 for SGLang (4,230). The Quail times are its
post-scan-ring runs (`2026-08-30-kv-ring-fix.md`), which cut its
IMDB-3 from the 75.0 seconds in the 2026-08-29 family run to 32.8;
BIO-2 was unchanged. Quail's per-query accuracy is in
`2026-08-27-quailb-sf01-4b.md` (its 30-query weighted answer
accuracy is 70.06%). $/query uses $3.9492 per H100! hour and excludes
startup; the SoL row's cost is the floor implied by its time.

## Prediction

Stated before the headline run (the final configuration,
mem_fraction_static=0.76 with the 1.0 second slice pause):

- BIO-2 in 1,080 to 1,150 seconds. Miss low: 1,028.7 seconds. The
  two page-16 runs differ by 44.6 seconds (4.3%) on identical join
  code, so single-run arithmetic at the tens-of-seconds level is
  inside this query's run-to-run variance.
- IMDB-3 in 66 to 72 seconds, join 48 to 54, filter 17 to 19. All
  hit: 67.0, 49.3, 17.6.
- No allocation retries at the lowered memory fraction. Miss: five
  flush-and-retry warnings, all recovered (see below).
- Answers within a pair or two of the earlier runs. Hit: BIO-2
  114,868 TRUE pairs against 114,870 and 114,869 in the two runs
  before it; IMDB-3 identical (4,230 survivors, 11,351 TRUE).

The 2026-08-30 report's prediction for the previous configuration
(BIO-2 1,250-1,500, IMDB-3 75-100) also hit; the overhead work moved
the numbers out from under it deliberately.

## What the numbers mean

- Both engines' joins are host-bound, not GPU-bound. The final BIO-2
  join moved 10,916 fresh tokens per second through a GPU that
  prefills tens of thousands per second, while per-request
  bookkeeping ran at 548 pairs/s. The direct proof: the page-16
  configuration computes 61% more fresh tokens than the page-1 run
  (11.23M against 6.97M, the cost of rounding cache hits down to
  16-token pages) and still finished 18% sooner, because it does
  less host work per request.
- Per-pair time fits a fixed cost plus a per-prompt-token cost.
  Pipelined SGLang: about 0.89 ms per request plus 0.23
  microseconds per prompt token (down from 1.09 ms and 0.28 with the
  2026-08-30 configuration). Pipelined vLLM: about 0.40 ms plus 0.52
  microseconds. The crossover sits near 1,600 prompt tokens: vLLM
  wins IMDB-3's 358-token pairs (0.59 against 0.97 ms per pair),
  SGLang wins BIO-2's 4,124-token pairs (1.83 against 2.53 ms).
  SGLang's larger fixed cost is its process architecture — every
  request crosses zmq to the scheduler process and back — and that
  floor is not reachable from the client side.
- Cache hit rates are engine-equal: 99.5% of join prompt tokens on
  BIO-2 for both engines, 87.7% (SGLang) against 88.4% (vLLM) on
  IMDB-3. The join order does its job; caching explains none of the
  remaining gap.
- Answers are stable but not bit-identical across configurations.
  The three page-1 runs returned bit-identical answers; the two
  page-16 runs each flipped one to two BIO-2 pairs (114,868 to
  114,870 TRUE of 563,500) because page rounding changes which
  prefix tokens are recomputed, which nudges fp8 numerics on
  borderline pairs. IMDB-3's answers were identical in all five
  SGLang runs. The engine-to-engine divergence is unchanged: about
  1.2% of BIO-2 pairs net flipped against vLLM, accuracy slightly up
  on SGLang.
- The single-stage IMDB-3 filter cannot show pipelining's real
  benefit (overlapping stage k+1 with stage k stragglers); the
  multi-stage chains in IMDB-4..7, BIO-4/5, FEV-4/6, and LEP-4..8
  would. Those queries run with `--query ""` (all 30) but were not
  measured here.

## How the configuration was reached

Seven measured findings; the raw evidence is in the crashed and
completed runs cited under Source data.

- Memory fraction, first pass: 0.78, not vLLM's 0.91. vLLM profiles
  a full-size forward (including final-position logits for
  `max_num_seqs` requests) before sizing its KV pool, so its 0.91
  covers the activation working set; SGLang's `mem_fraction_static`
  covers only weights plus KV. At 0.91 the boot crashed capturing
  prefill CUDA graphs (91 shapes up to the token budget, about 130
  MB retained each), so prefill graphs are off — cheap here because
  every request generates its one token during the prefill forward
  and decode batches never run. Still at 0.91, and again at 0.85,
  the BIO-2 join ran out of GPU memory mid-forward; the failing
  2.32 GiB allocation was exactly a float32 logits tensor for 4,096
  requests over the 151,936-token vocabulary, sitting next to an
  equally sized logit-bias tensor and 6.5 GiB of non-PyTorch kernel
  workspaces.
- Sliced submission. One `generate()` call with all 563,500 pairs
  creates one asyncio task per request inside SGLang's driver
  process; the Modal health heartbeat thread starved for twenty
  minutes and Modal threatened to kill the container. The client
  submits in slices of 16,384 requests with a one second pause
  between slices; the engine's queue never runs dry inside a slice.
  Under join load, heartbeat attempts still fail in stretches up to
  about four minutes whether the pause is 0.1 or 1.0 seconds (both
  2026-08-31 runs survived them), so the pause does not govern
  heartbeat health; 1.0 seconds is the conservative setting every
  completed run used, and it costs BIO-2 about 34 idle seconds.
- The tiled join order. SGLang's radix cache stores a prompt's KV
  only when its request finishes, so vLLM's anchor-major pair order
  recomputes an anchor for every sibling in flight — on IMDB-3 that
  meant 6.5% of join prompt tokens cached and a 191.7 second join.
  Suffix-major submission within anchor tiles sized to half the KV
  pool restores the reuse (87.7% cached here). Plain suffix-major
  without tiles would gain nothing: a full pass over more anchor
  tokens than the pool holds evicts every anchor before its next
  use. On BIO-2, where 1,127 pairs per report already hid the
  co-admission miss, the tiled order measured 4 to 15% slower than
  anchor-major on the 2026-08-30 configuration; the runner accepts
  that cost to cap the worst case.
- Filter chains in waves. vLLM's pipelined client drives the
  in-process engine with a synchronous add_request/step loop. SGLang
  has no such surface, so every alive document keeps exactly one
  request per blocking wave, advancing one stage per wave, with new
  documents admitted into freed slots at wave boundaries. Admission
  uses the same token-budget formula as the vLLM client (KV pool
  tokens divided by mean request size, capped at 4,096): doc_cap was
  1,146 for the IMDB-3 filter.
- 16-token cache pages (`page_size=16`, vLLM's block size). SGLang
  defaults to 1-token pages, which cost a radix-tree node and a KV
  index entry per token on every request. On the host-bound joins
  that bookkeeping, not the GPU, set the pace. Pages of 16 cut it
  16x for about 8 extra fresh tokens per request (hits round down to
  a page multiple), which the idle GPU absorbs. sglang requires the
  scheduled token budget to divide by the page size, so vLLM's
  25,305 becomes 25,296 (0.04% less).
- No tokenizer in the request path (`skip_tokenizer_init=True`). The
  client sends token ids and reads token ids back, but by default
  every finished request still crossed a detokenizer process that
  assembled text nobody read. With the flag, the scheduler sends
  results straight back to the driver (verified in the 0.5.18
  source: the detokenizer process is spawned but does no
  per-request work). Together with the pages and the shorter-lived
  experiments below, per-request fixed cost fell from about 1.09 to
  0.89 ms and BIO-2 from 1,256.9 to 1,028.7 seconds.
- Memory fraction, second pass: 0.76. Under 16-token pages the
  extend batch's row count varies step to step, so the float32
  logits and logit-bias tensors keep churning new segment sizes
  through PyTorch's allocator cache; occasionally the cache fills
  and a large allocation stalls CUDA to flush and retry. That
  happened at 0.78 and 0.76 alike (three and five retries in the two
  2026-08-31 runs, all recovered by PyTorch), but 0.78 ran the
  post-flush headroom down to 5 MB free, while 0.76 kept it near a
  gigabyte at the cost of 2.8% of the KV pool (403,568 tokens
  against vLLM's 479,248). The retries' timing impact is inside the
  4.3% run-to-run variance measured on BIO-2.

## Configuration

| Setting | vLLM baselines 0.26.0 | SGLang baseline 0.5.18 |
|---|---|---|
| GPU memory fraction | `gpu_memory_utilization=0.91` | `mem_fraction_static=0.76` (see above) |
| Max concurrent requests | `max_num_seqs=4096` | `max_running_requests=4096` |
| Scheduled token budget | `max_num_batched_tokens=25305` | `chunked_prefill_size=25296`, `max_prefill_tokens=25296` (page-multiple) |
| Prefix caching | on, 16-token blocks, usable while a request runs | radix cache on, 16-token pages, usable at request completion |
| Tokenizer in request path | in-process, prompts passed as token ids | none (`skip_tokenizer_init=True`), token ids in and out |
| Filter submission | stage-major (stock) or pipelined step loop | pipelined wave-advanced chains |
| Join pair order | anchor-major | suffix-major in anchor tiles |
| Answer decoding | greedy, `allowed_token_ids` = the 8 TRUE/FALSE first tokens | greedy, `logit_bias=+1000` on the same 8 token ids |
| Scheduling policy | first come, first served | first come, first served (0.5.18 default, recorded per run) |
| Checkpoint | `Qwen/Qwen3-4B-FP8` | `Qwen/Qwen3-4B-FP8` |
| KV dtype | BF16 | BF16 |
| Attention backend | vLLM default FlashAttention | fa3 (SGLang default on H100!) |

The logit bias reproduces vLLM's restricted decoding exactly: SGLang
adds the bias to float32 logits before its greedy argmax, so adding
the same constant to all eight allowed ids picks the same token as
restricting the argmax to those ids, as long as no other token's raw
logit is 1,000 higher. Raw logit gaps are two orders of magnitude
smaller than that.

The vLLM numbers come from the 2026-08-29 family run; the SGLang runs
used their own H100! containers on 2026-08-30 and 2026-08-31.

## Source data

- Pipelined SGLang, final configuration (the headline run), function
  call `fc-01M1ASD1MC633MZMTZ5DHZGF3R`:
  `/results/pipelined_sglang/2026-08-31_021203_e40e08d2/summary.json`
- Pipelined SGLang at mem_fraction_static=0.78 with the 0.1 second
  pause (the diagnostic run behind the 0.76 and 1.0 second
  choices), function call `fc-01M1AQSQKZ2VMS1MX2G36BRESS`:
  `/results/pipelined_sglang/2026-08-31_014401_6fffe2c6/summary.json`
- Pipelined SGLang on the 2026-08-30 configuration (1-token pages,
  detokenizer in the path), function call
  `fc-01M18NRQ1GB73P0CAZ9C2B9M7B`:
  `/results/pipelined_sglang/2026-08-30_063002_bf050db3/summary.json`
- Stock vLLM:
  `/results/stock_vllm/20260829T185407Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json`
- Pipelined vLLM:
  `/results/pipelined_vllm/20260829T185407Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json`
- Quail (scan-ring runs, current engine):
  `/results/ablations/ringfix_bio2.json` and
  `/results/ablations/ringfix_tokens_head_imdb3.json`
- SoL estimate: `/results/sol/sol_quailb_sf0.1.json`

All engine runs scored accuracy against ground truth collection
`gt_363b5ab570635c33894e1a030c21f57e` labels on corpus
`c_3bd14ed0758287cba9d88fb68de8b7b8` (the vLLM runs recorded the same
label sets under their earlier collection id
`gt_02ffa2a5720006e8236aa993760e9e29`).

The plot script `reports/make_sglang_baseline_plots.py` takes a work
directory holding the pulled files; its docstring has the
`modal volume get` commands.
