# SGLang baseline on BIO-2 and AGENT-1

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

The measurements cover two queries at scale factor 0.1 with Qwen3
4B fp8 on one H100!. BIO-2 comes from a run with zero CUDA OOM
retries and AGENT-1 from a second run on the identical client code;
Source data gives the per-query attribution.

- BIO-2: one join, REACTION over 500 reports x 1,127 terms =
  563,500 pairs. Long prompts (about 4,124 tokens per pair).
- AGENT-1: one filter (AGENT_RECOVERED) over 1,772 cumulative agent
  trace snapshots from SWE-Next, each capped at 24,000 tokens. A
  later snapshot of a trajectory contains the earlier one as a
  prefix, so the query measures prefix reuse across different
  documents.

## Result

Figure: plots/sglang_baseline_comparison.png

BIO-2 join (document pairs/second divides 563,500 evaluated pairs by
query time):

| System | Query time (s) | Document pairs/s | $/query | Answer accuracy |
|---|---:|---:|---:|---:|
| SoL estimate | 62.01 | 9,087 | $0.0680 | not applicable |
| Quail | 128.22 | 4,395 | $0.1406 | see the QuailB report |
| stock vLLM | 1,532.25 | 367.8 | $1.6809 | 80.92% |
| pipelined vLLM | 1,427.45 | 394.8 | $1.5659 | 80.91% |
| pipelined SGLang | 868.50 | 648.8 | $0.9527 | 82.03% |

AGENT-1, filter only (documents/second divides the 1,772 input
documents by query time; the SoL file does not cover the agent
queries):

| System | Query time (s) | Documents/s | $/query | Answer accuracy |
|---|---:|---:|---:|---:|
| Quail | 235.09 | 7.5 | $0.2579 | 75.00% |
| stock vLLM | 129.54 | 13.7 | $0.1421 | 74.15% |
| pipelined vLLM | 96.33 | 18.4 | $0.1057 | 74.15% |
| pipelined SGLang | 218.18 | 8.1 | $0.2393 | 73.93% |

- BIO-2: pipelined SGLang is the fastest of the three baseline
  engines — 1.64 times faster than pipelined vLLM and
  1.76 times faster than stock vLLM. Quail is still
  6.78 times faster than it, and the SoL estimate
  14.01 times.
- AGENT-1: pipelined SGLang took 218.2 seconds, 2.26 times slower
  than pipelined vLLM's 96.3 and 1.68 times slower than stock vLLM's
  129.5. Quail took 235.1 seconds. The ranking follows the cache hit
  rate: vLLM shared 68.22% of prompt tokens from KV via in-flight
  prefix sharing, SGLang shared 24.85% via completion-only radix
  caching, and Quail shared 0% across documents.
- Model startup, excluded from query time: 341.2 seconds (cold
  start with CUDA graph captures), compared with 143.8 to 235.2
  seconds for the vLLM family containers.

On BIO-2 every system evaluates the same 563,500 pairs. Which run
supplies which number: the BIO-2 vLLM columns come from the
2026-08-29 family run and the Quail column from its post-scan-ring
run (`2026-08-30-kv-ring-fix.md`); AGENT-1's Quail and vLLM numbers
come from the 2026-08-31 family runs that introduced the agent
queries; SGLang's BIO-2 comes from the clean run (zero CUDA OOM
retries) and its AGENT-1 from the headline run — the measured client
path is byte-identical between the two. Quail's
per-query accuracy is in `2026-08-31-quailb-kv-regret.md` (its
32-query weighted answer accuracy is 70.07%). $/query uses $3.9492
per H100! hour and excludes startup; the SoL row's cost is the floor
implied by its time.

## Prediction

Stated before the AGENT-1 run (the merge that added the agent
queries did not touch BIO-2's measured path):

- BIO-2 in 830 to 910 seconds: 868.5 measured previously on the
  identical path, plus or minus the 4.3% run-to-run spread measured
  on this query. The clean run (zero OOM retries) measured 868.50,
  within band. The headline run measured 1,038.57, outside band —
  three CUDA allocator flush-and-retry events during the join added
  about 170 seconds; accuracy (82.03%) and cache rates (99.5%) are
  identical between runs. The table uses the clean run.
- AGENT-1 in 180 to 300 seconds with 20 to 45% of prompt tokens
  served from KV. The reasoning: 1,772 requests mean the per-request
  fixed cost totals about 1.6 seconds, so this query is
  prefill-bound and the cache hit rate decides it. The radix cache
  shares only completed requests, so snapshots of the same
  trajectory admitted in the same wave recompute their common
  prefix; pipelined vLLM's in-flight sharing reached 68.22%.
  Measured 218.18 seconds with 24.85% of prompt tokens cached, both
  within the predicted bands.
## What the numbers mean

- Both engines' joins are host-bound, not GPU-bound. The BIO-2 join
  moved about 12,900 fresh tokens per second through a GPU that
  prefills tens of thousands per second, while per-request
  bookkeeping ran at 648.8 pairs/s. The engine
  configuration (16-token pages, no detokenizer) exists to cut that
  per-request host work; the rationale section explains each piece.
- Per-pair time fits a fixed cost plus a per-prompt-token cost.
  Pipelined SGLang: about 0.92 ms per request plus
  0.15 microseconds per prompt token. Pipelined vLLM:
  about 0.40 ms plus 0.52 microseconds. The crossover sits near
  1,400 prompt tokens: below that, vLLM's lower fixed cost
  dominates; above it, SGLang's lower per-token cost wins. BIO-2's
  4,124-token pairs sit well above (1.54 ms predicted for SGLang
  against 2.55 ms for vLLM). SGLang's larger fixed cost
  is its process architecture — every request crosses zmq to the
  scheduler process and back — and that floor is not reachable from
  the client side.
- Join cache hit rates are engine-equal: 99.5% of join
  prompt tokens on BIO-2 (vLLM: 99.5%). The join order does its
  job; caching explains none of the remaining join gap.
- AGENT-1 is where the caches differ. The corpus is cumulative
  snapshots — a later snapshot of a trajectory contains the earlier
  one as a prefix — and vLLM's prefix cache shares that text even
  between co-scheduled requests, serving 68.22% of prompt tokens
  from KV. SGLang served 24.85%: the radix cache stores a
  request's KV only at completion, so a snapshot can only reuse a
  sibling that finished in an earlier wave. With 2.37 times more
  fresh tokens to prefill than vLLM, the 2.26-times-longer runtime
  confirms AGENT-1 is prefill-bound: the cache difference accounts
  for the speed difference.
- Engine-to-engine answer divergence is small: about 1.2% of BIO-2
  pairs net flipped against vLLM, with accuracy slightly up on
  SGLang. Both engines run the same prompts with greedy decoding;
  the flips are fp8 numeric noise on borderline pairs, which differs
  with which prefix tokens each engine recomputes.
- The single-stage AGENT-1 filter cannot show pipelining's real
  benefit (overlapping stage k+1 with stage k stragglers); the
  multi-stage chains in IMDB-4..7, FEV-4/6, and
  LEP-4..8 would. Those queries run with `--query ""` (all 32) but
  were not measured here.

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
  41 for AGENT-1's long documents.
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

Every run used its own H100! container; the sources paragraph under
Result says which run supplies which number.

## Source data

- Pipelined SGLang, BIO-2 clean run (zero CUDA OOM retries),
  function call `fc-01M1AYQ53Y29WSWWK4C85HCB7G`:
  `/results/pipelined_sglang/2026-08-31_034457_9032b486/summary.json`
- Pipelined SGLang, AGENT-1 (same client path), function call
  `fc-01M1CVW8HTARK3GN2HFH82WH1A`:
  `/results/pipelined_sglang/2026-08-31_213347_9f85216d/summary.json`
  (this run also measured BIO-2 at 1,038.57 seconds with three CUDA
  OOM flush-and-retry events; the table uses the clean run above)
- Stock and pipelined vLLM, BIO-2:
  `/results/stock_vllm/20260829T185407Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json`
  and
  `/results/pipelined_vllm/20260829T185407Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json`
- Stock and pipelined vLLM, AGENT-1:
  `/results/stock_vllm/20260831T062218Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json`
  and
  `/results/pipelined_vllm/20260831T062218Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json`
- Quail, BIO-2 (scan-ring run, current engine):
  `/results/ablations/ringfix_bio2.json`
- Quail, AGENT-1 (2026-08-31 family run):
  `/results/benchmarks/quailb/runs/qb_20260831T062218Z_1192cd76/20260831T062218Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families.json`
- SoL estimate (no agent queries):
  `/results/sol/sol_quailb_sf0.1.json`

Every engine run scored accuracy against the shared Qwen3 32B ground
truth; each cited summary records the collection and corpus ids it
used (the AGENT-1 run uses `gt_77bb8b128743a79aedddaa24c808c3f8` on
corpus `c_1aa2c4f0d0b6c816fd37aa5748c33341`, the collection that
added the agent label sets; the earlier runs record the equivalent
label sets under earlier collection ids).

The plot script `reports/make_sglang_baseline_plots.py` takes a work
directory holding the pulled files; its docstring has the
`modal volume get` commands.
