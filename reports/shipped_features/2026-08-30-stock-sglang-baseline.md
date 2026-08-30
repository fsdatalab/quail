# Pipelined SGLang baseline runner

## What changed

`baselines/stock_sglang/run.py` runs QUAIL-B queries on SGLang
(0.5.18) as `pipelined_sglang`, the SGLang counterpart of pipelined
vLLM: per-document filter chains with token-budget admission, and one
request per document pair for joins (full cross product). An empty
`--query` runs all 30 current QuailB queries.

The runner does not copy the query logic. It wraps a running
`sglang.Engine` in `StockSGLangClient`, an adapter with the
vLLM-shaped `generate()` surface, and passes that adapter to the
existing `run_query` from `baselines.stock_vllm.run`, so all
baselines build identical prompt token ids and identical per-query
records. The shared pipelined filter path dispatches to a
client-provided chain when one exists, leaving the vLLM engine loop
untouched.

## Why

Stock and pipelined vLLM were the only stock engine comparisons. A
second engine with its own scheduler and prefix cache shows which
part of the measured gap is vLLM-specific and which part is the
submission pattern itself.

## Design points

- Filter chains advance in waves of blocking `generate()` calls:
  every alive document has exactly one request per wave at its
  current stage, and freed admission slots refill at wave
  boundaries. No client code uses asyncio, matching the synchronous
  vLLM clients; SGLang's scheduler runs in a separate process with
  no add_request/step surface, so waves stand in for vLLM's engine
  step loop.
- Join pairs are submitted suffix-major within anchor tiles sized to
  half the measured KV pool
  (`baselines.stock.suffix_major_tiled_order`), because SGLang's
  radix cache stores KV only for finished requests and vLLM's
  anchor-major order would recompute an anchor for every sibling in
  flight. Answers return in anchor-major pair order; stock vLLM
  keeps anchor-major.
- vLLM's `allowed_token_ids` restriction has no SGLang equivalent;
  the runner adds a +1000 `logit_bias` to the same eight TRUE/FALSE
  token ids, which picks the same token under greedy decoding
  because the bias lands on float32 logits before the argmax.
- vLLM's `gpu_memory_utilization=0.91` maps to
  `mem_fraction_static=0.78`, not 0.91: vLLM's fraction includes the
  activation working set (it profiles a forward, with logits for
  `max_num_seqs` requests, before sizing KV), SGLang's does not, and
  0.91 and 0.85 both ran out of GPU memory on BIO-2. Prefill CUDA
  graphs are disabled for the same reason, and batch submission is
  sliced to 16,384 requests so the driver process stays responsive.

## Numbers

The measured comparison against the SoL estimate, Quail, and both
vLLM baselines on BIO-2 and IMDB-3 at sf=0.1 is in
`reports/2026-08-30-sglang-baseline.md`.
