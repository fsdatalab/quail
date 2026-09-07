# SGLang pair ordering without client batch limits

- SGLang now advances a document to its next filter as soon as its answer
  arrives. When a document finishes, another enters immediately. It uses
  the same admission calculation as pipelined vLLM, including KV page rounding.
- Both baselines submit all pairs for a join in one call. SGLang submits all
  anchors for one partner before moving to the next partner. The anchor is
  the document placed first in each pair prompt. Requests sharing an anchor
  are separated so earlier requests can populate reusable KV. vLLM keeps
  its existing order, with all partners for one anchor submitted together.
  SGLang no longer divides joins into half-KV groups or blocking batches of
  at most 16,384 requests.
- The SGLang benchmark now runs in a child process, using the existing
  vLLM process runner. The parent can send Modal heartbeat messages while
  inference runs. No request batch size is chosen to accommodate those messages.
  Process cleanup released GPU memory after this run.
- We reran only SGLang FEV-9. The other three FEV-9 measurements and all
  124 configurations for the other 31 queries are reused. The plots label
  those other SGLang measurements as using the earlier adapter.

[Open the FEVER vector PDF](plots/quailb_fev.pdf)

[![FEVER comparison](plots/quailb_fev.png)](plots/quailb_fev.pdf)

Figure: plots/quailb_fev.png

## Setup and prediction

- FEV-9 has four filters and three joins. Inputs are 500 claims for each
  of `c1` and `c2`, and 287 evidence documents for each of `e1` and `e2`.
  Repeated aliases use the same input sets.
- All methods use Qwen3 4B FP8 at sf=0.1 and lf=1, with one H100 each.
  Quail and vLLM ran sequentially on the same physical GPU in the saved
  comparison. SGLang ran on a separate H100. This is one measured query
  after the existing startup warmup, with no full-query warmup.
- SGLang 0.5.18 settings are unchanged: static memory fraction 0.76,
  4,096 maximum running requests, 25,296 prefill tokens, 16-token pages,
  prefix caching enabled, and prefill CUDA graphs disabled. The eight
  TRUE/FALSE token IDs still receive a logit bias of +1,000.
- vLLM 0.26.0 uses prefix caching, 4,096 maximum requests, and 25,305
  maximum batched tokens. Stock vLLM uses operator-at-a-time submission.
- Before the run, we predicted fresh computation would return near 4.10
  million tokens, compared with 20.39 million under the intermediate order.
  We expected runtime closer to 138.26 seconds than 304.92 seconds, with
  answer agreement near 69.08%. All FEV-9 anchors fit in the previous tile
  budget, so removing tiles preserves this query's earlier pair order.

## Measured results

| Method | Query seconds | Document pairs/second | $/query | Recomputed KV tokens | Fresh input tokens | Answer agreement (%) |
|---|---:|---:|---:|---:|---:|---:|
| Quail | 41.14 | 4,426.71 | 0.04513 | 7,309 | 4,314,219 | 67.77 |
| Stock vLLM | 90.07 | 2,108.23 | 0.09881 | 194,016 | 4,593,860 | 67.05 |
| Pipelined vLLM | 89.80 | 2,114.57 | 0.09851 | 193,312 | 4,593,156 | 67.05 |
| Pipelined SGLang | 135.74 | 1,261.21 | 0.14891 | 33,904 | 4,095,266 | 69.08 |

| SGLang submission | Query seconds | Document pairs/second | $/query | Fresh input tokens | Recomputed KV tokens |
|---|---:|---:|---:|---:|---:|
| Earlier order with tiles and request slices | 138.26 | 1,238.23 | 0.15167 | 4,096,274 | 33,904 |
| All partners per anchor, no tiles or slices | 304.92 | 561.45 | 0.33450 | 20,388,882 | 15,673,360 |
| All anchors per partner, no tiles or slices | 135.74 | 1,261.21 | 0.14891 | 4,095,266 | 33,904 |

- The prediction held. FEV-9 took 135.74 seconds, compared with 304.92
  seconds under the intermediate order and 138.26 seconds under the older
  limits. Fresh computation returned to 4,095,266 tokens. Recomputed KV
  returned to 33,904 tokens, exactly matching the older policy.
- The 2.52-second difference from the older policy is small. One run per
  configuration does not establish a speedup. The measurement does show
  that FEV-9 retains the earlier reuse without the half-KV tile budget or
  the 16,384-request blocking limit.
- Join generation took 131.08 seconds, compared with 300.33 seconds
  under the intermediate order and 132.27 seconds under the earlier policy.
  Filter generation took 3.05 seconds. Each alias has one filter, so
  FEV-9 does not measure advancement between filters on the same document.
  CPU tests cover that behavior.
- All three SGLang runs evaluated 171,197 document pairs and returned
  118,565,289 rows. Their accuracy counters match exactly. Answer agreement
  remains 69.08%. All seven saved predicate answer tables match exactly.
- SGLang can reuse computed prefixes, but requests selected together can all
  miss a prefix before it is computed and inserted into the shared index.
  The restored order separates requests sharing an anchor. A request trace
  would separate simultaneous duplicate computation from KV eviction.
  See SGLang's [prefix cache](https://github.com/sgl-project/sglang/blob/v0.5.18/python/sglang/srt/mem_cache/radix_cache.py).
- Only the intermediate and current runs differ solely in pair order.
  The older comparison also changed request slices and filter scheduling.
  Each measured query ran once on a separate H100 allocation. The intermediate
  run included two brief process-stack inspections and two GPU-utilization
  reads. The current run did not use a profiler or process-stack inspection.
- FEV-9 fits all anchors in the former tile budget. Queries with larger
  anchor sets may lose KV between requests without tiling; those queries
  were not rerun. Their saved results remain labeled as using the old adapter.
- The Modal call completed. Cleanup used SIGKILL and left
  4 MiB allocated on the GPU.

| Method | Evaluated document pairs | Returned rows | Matching reference rows | Output precision (%) | Output recall (%) | Startup seconds |
|---|---:|---:|---:|---:|---:|---:|
| Quail | 182,115 | 149,783,486 | 5 | 0.00000334 | 45.45 | 23.45 |
| Stock vLLM | 189,888 | 172,090,043 | 5 | 0.00000291 | 45.45 | 93.66 |
| Pipelined vLLM | 189,888 | 172,090,043 | 5 | 0.00000291 | 45.45 | 0.00 |
| Pipelined SGLang | 171,197 | 118,565,289 | 5 | 0.00000422 | 45.45 | 329.65 |

- The reference model is Qwen3 32B FP8. Answer agreement measures evaluated
  filter and join answers against saved reference labels. The reference
  has 11 final output rows. Final output precision remains very low.
- Fresh input tokens count input token positions computed instead of read
  from KV. Repeated computation counts again. Recomputed KV tokens are
  included in fresh tokens. They count reusable document or anchor prefix
  tokens computed again under the existing per-document accounting.
- Throughput is the sum of evaluated document pairs over all joins divided
  by query seconds. Cost is query seconds divided by 3,600 and multiplied
  by $3.9492 per H100 hour. Query time and primary cost exclude startup
  and result collection. One run does not measure runtime variation.
- The horizontal SoL lines credit matching prefixes across requests and
  aliases. FEV-9 remains 5.821 seconds with unlimited KV and reference-label
  survivors. Different answers change the measured amount of join work, so
  the gap from SoL is not purely execution overhead.

| Baseline | Available KV tokens | Maximum requests | Filter document admission cap |
|---|---:|---:|---|
| Stock vLLM | 479,616 | 4,096 | All input rows per operator |
| Pipelined vLLM | 479,616 | 4,096 | Claims: 4,096; evidence: 938 |
| Pipelined SGLang | 403,744 | 4,096 | Claims: 4,096; evidence: 790 |

- Admission divides available KV tokens by the mean request length rounded
  to KV pages, then limits the result to the engine's request limit. A request
  includes the document, its longest filter question, and one answer token.
  The old SGLang evidence cap was 802 because it did not round request lengths.
  All FEV-9 input sets fit within both admission caps.
- Quail retains its analytical settings of 110,376 tokens per chunk and
  362,240 usable KV tokens. All methods plan join order before execution from
  the same selectivity estimates and run the first anchor's filters last.
  No setting was tuned from measured survivors.

## Reproduce and sources

```bash
set -o pipefail
uv run modal run --detach experiments/cells/sglang_baseline.py \
  2>&1 | tee /tmp/quail-sglang-baseline.log
```

- Modal app: `quail-milestone1`.
- SGLang function call: `fc-01M1WD92E7P8HB40XXWQJFYCN4`.
- Current SGLang summary, GPU identity, and cleanup on `quail-results`: `/results/benchmarks/quailb/families/20260906T222629Z-sglang-suffix-major/fever-sglang-process.json`.
- Its query summary: `/results/benchmarks/quailb/runs/qb_20260906T222649Z_af71c85b/20260906T222629Z-sglang-suffix-major-fever-pipelined_sglang.json`.
- The intermediate anchor-major run used commit `9f929e9` and function call
  `fc-01M1WC3GQ6585E6FMHA7QAAN41`. Its source is
  `/results/benchmarks/quailb/families/20260906T220559Z-sglang-baseline-redesign/fever-sglang-process.json`.
- Saved Quail and vLLM measurements, plus the previous SGLang result:
  `/results/benchmarks/quailb/family-runs/20260906T211500Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/manifest.json`.
  Parent call: `fc-01M1W960914BX21C0BNX8RT806`. Quail/vLLM call:
  `fc-01M1W96BXB4MW1N9W6B1QDBF03`. Previous SGLang call:
  `fc-01M1W96C2673KKFH0X87JJHF3F`. Those measurements used commit `dd63fa6`.
- Corpus: `c_1aa2c4f0d0b6c816fd37aa5748c33341`.
  Reference collection: `gt_77bb8b128743a79aedddaa24c808c3f8`.
- Other 31 queries:
  `/results/benchmarks/quailb/family-runs/20260905T021527Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/manifest.json`.
- SoL: `/results/sol/2026-09-06-quailb-prefix-reuse.json`.
- Download and figure commands are in `reports/make_quailb_comparison_plots.py`.
