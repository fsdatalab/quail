# Stock SGLang baseline runner

## What changed

`baselines/stock_sglang/run.py` runs QUAIL-B queries on stock SGLang
(0.5.18), the same way `baselines/stock_vllm/run.py` runs them on
stock vLLM: one request per document per filter stage (stage-major
waves) and one request per document pair for joins (full cross
product).

The runner does not copy the query logic. It wraps a running
`sglang.Engine` in `StockSGLangClient`, a small adapter with the
vLLM-shaped `generate()` surface, and passes that adapter to the
existing `run_query` from `baselines.stock_vllm.run`. Both baselines
therefore build identical prompt token ids and identical per-query
records.

## Why

Stock vLLM was the only stock engine comparison. A second engine with
its own scheduler and prefix cache shows which part of the measured
gap is vLLM-specific and which part is the submission pattern itself.

## Configuration mapping

The SGLang settings are the analytic equivalents of the stock vLLM
ones: `max_running_requests=4096`, `chunked_prefill_size=25305`,
`max_prefill_tokens=25305`, radix cache on. vLLM's
`allowed_token_ids` restriction has no SGLang equivalent; the runner
instead adds a +1000 `logit_bias` to the same eight TRUE/FALSE token
ids, which picks the same token under greedy decoding because the
bias is applied to float32 logits before the argmax.

vLLM's `gpu_memory_utilization=0.91` maps to
`mem_fraction_static=0.78`, not 0.91: vLLM's fraction includes the
activation working set (it profiles a forward, with logits for
`max_num_seqs` requests, before sizing KV), SGLang's does not, and
both 0.91 and 0.85 ran out of GPU memory on BIO-2. Prefill CUDA
graphs are disabled for the same memory reason. The client also
submits in slices of 16,384 requests: one `generate()` call with all
563,500 BIO-2 pairs starved the Modal health heartbeat thread until
Modal threatened to kill the container. The report explains all
three choices with the measured numbers.

## Numbers

The first run and the comparison against stock vLLM on BIO-2 and
IMDB-3 at sf=0.1 are in `reports/2026-08-30-stock-sglang-baseline.md`.
