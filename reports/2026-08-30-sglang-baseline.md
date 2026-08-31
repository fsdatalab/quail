# SGLang baseline on BIO-2 and IMDB-3

Date: 2026-08-30.

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
| BIO-2 | pipelined SGLang | 1,256.91 | 448.3 | $1.3788 | 82.03% |
| IMDB-3 | SoL estimate | 9.56 | 5,024 | $0.0105 | not applicable |
| IMDB-3 | Quail | 32.79 | 1,603 | $0.0360 | see the QuailB report |
| IMDB-3 | stock vLLM | 52.65 | 996.9 | $0.0578 | 78.97% |
| IMDB-3 | pipelined vLLM | 48.21 | 1,088.8 | $0.0529 | 78.97% |
| IMDB-3 | pipelined SGLang | 79.38 | 639.5 | $0.0871 | 79.94% |

- BIO-2: pipelined SGLang is the fastest of the three baseline
  engines — 1.14 times faster than pipelined vLLM and 1.22 times
  faster than stock vLLM. Quail is still 9.8 times faster than it,
  and the SoL estimate 20.3 times.
- IMDB-3: pipelined SGLang is the slowest measured system — 1.65
  times slower than pipelined vLLM and 2.4 times slower than Quail's
  32.8 seconds.
- Its answer accuracy against the shared Qwen3 32B ground truth is
  the highest of the three engines on both queries (82.03% and
  79.94%).
- The wave-chained filter took 18.8 seconds for the same 5,000
  reviews the stage-major client filtered in 21.5 and 22.7 seconds
  in the earlier runs.
- Model startup, excluded from query time: 360 seconds (warm kernel
  caches), compared with 143.8 to 235.2 seconds for the vLLM family
  containers.

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

Stated before the run:

- BIO-2 in 1,250 to 1,500 seconds. Hit: 1,256.9 seconds.
- IMDB-3 in 75 to 100 seconds, with a 20-30 second filter. Time hit
  (79.4 seconds); the filter came in just under the band at 18.8
  seconds.
- Answers identical to the earlier SGLang runs. Hit exactly: 4,230
  filter survivors, 114,869 and 11,351 TRUE pairs, accuracy 82.03%
  and 79.94%.

## What the numbers mean

- All three baseline engines sit far above Quail and the SoL
  estimate on BIO-2 because the stock submission pattern pays
  per-request host work for every one of 563,500 pairs: 2.2
  milliseconds each on pipelined SGLang against the 0.11
  milliseconds the SoL model compute alone would need. Pipelined
  SGLang wins among the engines there because its scheduler
  overhead per request is lowest.
- On IMDB-3 the ordering reverses: with only 12 pairs per review,
  per-request cost dominates a small join, and SGLang's request path
  (tokenizer manager process, zmq, per-request result assembly) is
  more expensive than vLLM's in-process loop — 1.2 milliseconds per
  pair against vLLM's 0.6 — even though the tiled order gives SGLang
  the same cache hit rate (89.3% of join prompt tokens, against
  pipelined vLLM's 88.4%).
- Batch shape moves time, not answers: earlier SGLang runs with
  anchor-major and stage-major submission returned bit-identical
  answers to the pipelined run. The remaining answer divergence is
  between engines (different fp8 kernels; about 1.2% of BIO-2 pairs
  net flipped, accuracy slightly up on SGLang).
- The single-stage IMDB-3 filter cannot show pipelining's real
  benefit (overlapping stage k+1 with stage k stragglers); the
  multi-stage chains in IMDB-4..7, BIO-4/5, FEV-4/6, and LEP-4..8
  would. Those queries run with `--query ""` (all 30) but were not
  measured here.

## How the configuration was reached

Getting SGLang to run this workload took four measured findings; the
raw evidence is in the crashed and completed runs cited under Source
data.

- Memory fraction 0.78, not vLLM's 0.91. vLLM profiles a full-size
  forward (including final-position logits for `max_num_seqs`
  requests) before sizing its KV pool, so its 0.91 covers the
  activation working set; SGLang's `mem_fraction_static` covers only
  weights plus KV. At 0.91 the boot crashed capturing prefill CUDA
  graphs (91 shapes up to the 25,305-token budget, about 130 MB
  retained each), so prefill graphs are off — cheap here because
  every request generates its one token during the prefill forward
  and decode batches never run. Still at 0.91, and again at 0.85,
  the BIO-2 join ran out of GPU memory mid-forward; the failing
  2.32 GiB allocation was exactly a float32 logits tensor for 4,096
  requests over the 151,936-token vocabulary, sitting next to an
  equally sized logit-bias tensor and 6.5 GiB of non-PyTorch kernel
  workspaces. 0.78 leaves room for all of it and a 415,024-token KV
  pool, 13.4% smaller than vLLM's 479,248.
- Sliced submission. One `generate()` call with all 563,500 pairs
  creates one asyncio task per request inside SGLang's driver
  process; the Modal health heartbeat thread starved for twenty
  minutes and Modal threatened to kill the container. The client
  submits in slices of 16,384 requests with a one second pause
  between slices; the engine's queue never runs dry inside a slice.
- The tiled join order. SGLang's radix cache stores a prompt's KV
  only when its request finishes, so vLLM's anchor-major pair order
  recomputes an anchor for every sibling in flight — on IMDB-3 that
  meant 6.5% of join prompt tokens cached and a 191.7 second join.
  Suffix-major submission within anchor tiles sized to half the KV
  pool restores the reuse (89.3% cached, 61 second join). Plain
  suffix-major without tiles would gain nothing: a full pass over
  more anchor tokens than the pool holds evicts every anchor before
  its next use. On BIO-2, where 1,127 pairs per report already hid
  the co-admission miss, the tiled order measures 4 to 15% slower
  than anchor-major; the runner accepts that cost to cap the
  worst case.
- Filter chains in waves. vLLM's pipelined client drives the
  in-process engine with a synchronous add_request/step loop. SGLang
  has no such surface, so every alive document keeps exactly one
  request per blocking wave, advancing one stage per wave, with new
  documents admitted into freed slots at wave boundaries. Admission
  uses the same token-budget formula as the vLLM client (KV pool
  tokens divided by mean request size, capped at 4,096): doc_cap was
  1,152 for the IMDB-3 filter.

## Configuration

| Setting | vLLM baselines 0.26.0 | SGLang baseline 0.5.18 |
|---|---|---|
| GPU memory fraction | `gpu_memory_utilization=0.91` | `mem_fraction_static=0.78` (see above) |
| Max concurrent requests | `max_num_seqs=4096` | `max_running_requests=4096` |
| Scheduled token budget | `max_num_batched_tokens=25305` | `chunked_prefill_size=25305`, `max_prefill_tokens=25305` |
| Prefix caching | on, 16-token blocks, usable while a request runs | radix cache on (default), 1-token pages, usable at request completion |
| Filter submission | stage-major (stock) or pipelined step loop | pipelined wave-advanced chains |
| Join pair order | anchor-major | suffix-major in anchor tiles |
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

The vLLM numbers come from the 2026-08-29 family run; the SGLang runs
used their own H100! containers on 2026-08-30.

## Source data

- Pipelined SGLang (the headline run), function call
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
