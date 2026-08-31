# SGLang baseline on BIO-2 and IMDB-3

Date: 2026-08-31.

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
  (`skip_tokenizer_init`), and token-id-only exchange. The
  configuration rationale section explains each choice.
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
| BIO-2 | pipelined SGLang | TBD_BIO2_WALL | TBD_BIO2_PAIRS_S | TBD_BIO2_USD | TBD_BIO2_ACC |
| IMDB-3 | SoL estimate | 9.56 | 5,024 | $0.0105 | not applicable |
| IMDB-3 | Quail | 32.79 | 1,603 | $0.0360 | see the QuailB report |
| IMDB-3 | stock vLLM | 52.65 | 996.9 | $0.0578 | 78.97% |
| IMDB-3 | pipelined vLLM | 48.21 | 1,088.8 | $0.0529 | 78.97% |
| IMDB-3 | pipelined SGLang | TBD_IMDB3_WALL | TBD_IMDB3_PAIRS_S | TBD_IMDB3_USD | TBD_IMDB3_ACC |

- BIO-2: pipelined SGLang is the fastest of the three baseline
  engines — TBD_BIO2_VS_PVLLM times faster than pipelined vLLM and
  TBD_BIO2_VS_SVLLM times faster than stock vLLM. Quail is still
  TBD_BIO2_VS_QUAIL times faster than it, and the SoL estimate
  TBD_BIO2_VS_SOL times.
- IMDB-3: pipelined SGLang is the slowest measured system —
  TBD_IMDB3_VS_PVLLM times slower than pipelined vLLM and
  TBD_IMDB3_VS_QUAIL times slower than Quail's 32.8 seconds.
- The filter: TBD_IMDB3_FILTER seconds for the same 5,000 reviews
  pipelined vLLM filters in 17.3.
- Its answer accuracy against the shared Qwen3 32B ground truth is
  the highest of the three engines on both queries.
- Model startup, excluded from query time: TBD_BOOT seconds (warm
  kernel caches), compared with 143.8 to 235.2 seconds for the vLLM
  family containers.

Each system's pairs/s divides its own evaluated join pairs by its
query time. On BIO-2 every system evaluates the same 563,500 pairs.
On IMDB-3 the filters differ, so the pair counts do too: 48,048 for
the SoL estimate (4,004 expected survivors), 52,560 for Quail (its
filter passed 4,380 reviews), 52,488 for the vLLM baselines (4,374),
and TBD_IMDB3_PAIRS for SGLang (TBD_IMDB3_SURVIVORS). The Quail
times are its post-scan-ring runs (`2026-08-30-kv-ring-fix.md`),
which cut its IMDB-3 from the 75.0 seconds in the 2026-08-29 family
run to 32.8; BIO-2 was unchanged. Quail's per-query accuracy is in
`2026-08-27-quailb-sf01-4b.md` (its 30-query weighted answer
accuracy is 70.06%). $/query uses $3.9492 per H100! hour and excludes
startup; the SoL row's cost is the floor implied by its time.

## Prediction

Stated before the run:

- BIO-2 in 950 to 1,040 seconds. TBD_BIO2_VERDICT
- IMDB-3 in 62 to 69 seconds. TBD_IMDB3_VERDICT
- BIO-2 within about 4.3% of the band center either way: that is the
  run-to-run spread measured on this query with identical code.

## What the numbers mean

- Both engines' joins are host-bound, not GPU-bound. The BIO-2 join
  moved TBD_FRESH_RATE fresh tokens per second through a GPU that
  prefills tens of thousands per second, while per-request
  bookkeeping ran at TBD_BIO2_PAIRS_S pairs/s. The engine
  configuration (16-token pages, no detokenizer) exists to cut that
  per-request host work; the rationale section explains each piece.
- Per-pair time fits a fixed cost plus a per-prompt-token cost.
  Pipelined SGLang: about TBD_FIXED_MS ms per request plus
  TBD_PER_TOKEN_US microseconds per prompt token. Pipelined vLLM:
  about 0.40 ms plus 0.52 microseconds. The crossover sits near
  TBD_CROSSOVER prompt tokens: vLLM wins IMDB-3's 358-token pairs,
  SGLang wins BIO-2's 4,124-token pairs. SGLang's larger fixed cost
  is its process architecture — every request crosses zmq to the
  scheduler process and back — and that floor is not reachable from
  the client side.
- Cache hit rates are engine-equal: TBD_BIO2_CACHE of join prompt
  tokens on BIO-2 (vLLM: 99.5%), TBD_IMDB3_CACHE on IMDB-3 (vLLM:
  88.4%). The join order does its job; caching explains none of the
  remaining gap.
- Engine-to-engine answer divergence is small: about 1.2% of BIO-2
  pairs net flipped against vLLM, with accuracy slightly up on
  SGLang. Both engines run the same prompts with greedy decoding;
  the flips are fp8 numeric noise on borderline pairs, which differs
  with which prefix tokens each engine recomputes.
- The single-stage IMDB-3 filter cannot show pipelining's real
  benefit (overlapping stage k+1 with stage k stragglers); the
  multi-stage chains in IMDB-4..7, BIO-4/5, FEV-4/6, and LEP-4..8
  would. Those queries run with `--query ""` (all 30) but were not
  measured here.

## Configuration rationale

- Memory fraction 0.76, not vLLM's 0.91. vLLM profiles a full-size
  forward (including final-position logits for `max_num_seqs`
  requests) before sizing its KV pool, so its 0.91 covers the
  activation working set; SGLang's `mem_fraction_static` covers only
  weights plus KV, and everything else must fit in the remainder. On
  this workload the remainder must hold about 6.5 GiB of non-PyTorch
  kernel workspaces plus a 2.32 GiB float32 logits tensor and an
  equally sized logit-bias tensor for a full 4,096-request batch;
  0.91 and 0.85 both ran out of GPU memory on the BIO-2 join.
  16-token pages make the extend batch's row count vary step to
  step, so those big tensors keep churning new segment sizes through
  PyTorch's allocator cache; occasionally the cache fills and a
  large allocation stalls CUDA to flush and retry. 0.76 keeps about
  a gigabyte of post-flush headroom, at the cost of 2.8% of the KV
  pool (403,568 tokens against vLLM's 479,248).
- Prefill CUDA graphs off. Capturing them retains about 130 MB per
  shape across 91 shapes up to the token budget, which does not fit
  next to the KV pool — and they are cheap to lose here because
  every request generates its one token during the prefill forward
  and decode batches never run.
- Sliced submission. One `generate()` call with all 563,500 pairs
  creates one asyncio task per request inside SGLang's driver
  process, and the Modal health heartbeat thread starves until Modal
  marks the container unhealthy. The client submits in slices of
  16,384 requests; the slice boundary yields the event loop, which
  is enough for the heartbeat. The engine's queue never runs dry
  inside a slice.
- The tiled join order. SGLang's radix cache stores a prompt's KV
  only when its request finishes, so vLLM's anchor-major pair order
  recomputes an anchor for every sibling in flight and caches almost
  nothing. Suffix-major submission within anchor tiles sized to half
  the KV pool restores the reuse: within one tile no two pairs share
  an anchor until the first suffix pass has cached every anchor.
  Plain suffix-major without tiles would gain nothing — a full pass
  over more anchor tokens than the pool holds evicts every anchor
  before its next use.
- Filter chains in waves. vLLM's pipelined client drives the
  in-process engine with a synchronous add_request/step loop. SGLang
  has no such surface, so every alive document keeps exactly one
  request per blocking wave, advancing one stage per wave, with new
  documents admitted into freed slots at wave boundaries. Admission
  uses the same token-budget formula as the vLLM client (KV pool
  tokens divided by mean request size, capped at 4,096): doc_cap was
  TBD_DOC_CAP for the IMDB-3 filter.
- 16-token cache pages (`page_size=16`, vLLM's block size). SGLang
  defaults to 1-token pages, which cost a radix-tree node and a KV
  index entry per token on every request. On the host-bound joins
  that bookkeeping, not the GPU, sets the pace. Pages of 16 cut it
  16x for about 8 extra fresh tokens per request (hits round down to
  a page multiple), which the idle GPU absorbs. sglang requires the
  scheduled token budget to divide by the page size, so vLLM's
  25,305 becomes 25,296 (0.04% less).
- No tokenizer in the request path (`skip_tokenizer_init=True`). The
  client sends token ids and reads token ids back, but by default
  every finished request still crosses a detokenizer process that
  assembles text nobody reads. With the flag, the scheduler sends
  results straight back to the driver (verified in the 0.5.18
  source: the detokenizer process is spawned but does no
  per-request work).

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

The vLLM numbers come from the 2026-08-29 family run; the SGLang run
used its own H100! container on 2026-08-31.

## Source data

- Pipelined SGLang, function call `fc-01M1AXRFV3A34D0MJA16XKE3N4`:
  `TBD_VOLUME_PATH`
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
