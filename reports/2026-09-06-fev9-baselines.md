# FEV-9 with all four methods

- We ran only FEV-9 with Quail, stock vLLM, pipelined vLLM, and pipelined
  SGLang. All four use the current query with four filters and three joins.
  The main and FEVER plots include these measurements. The other 31 queries
  retain all 124 saved configurations from September 5.
- The inputs are 500 claims for each of `c1` and `c2`, and 287 evidence
  documents for each of `e1` and `e2`. Repeated aliases use the same input
  sets. Every method uses Qwen3 4B FP8 at `sf=0.1`, `lf=1`, with one H100.
- Quail and the two vLLM configurations ran sequentially on the same physical
  H100. Quail had its own process. The vLLM configurations shared a loaded
  model and cleared prefix KV between queries. SGLang ran on a separate H100.
- The runner used its existing startup warmup and one measured query per
  method. It did not run a separate full-query warmup. Query time excludes
  model startup and result collection. The earlier 39.02-second retention
  measurement followed a full-query warmup on a different H100.
- The runtime images use vLLM 0.26.0 and SGLang 0.5.18. Source code is from
  commit `dd63fa6`. No engine code or benchmark settings changed for this run.
- Before the run, we predicted Quail would remain near 39 seconds and beat
  the baselines. We expected the four filters to reduce join work for all
  methods. We predicted Quail's answer agreement would remain near 67.77%.

[Open the FEVER vector PDF](plots/quailb_fev.pdf)

[![FEVER comparison](plots/quailb_fev.png)](plots/quailb_fev.pdf)

Figure: plots/quailb_fev.png

| Method | Query seconds | Document pairs/second | $/query | Recomputed KV tokens | Fresh input tokens | Answer agreement (%) |
|---|---:|---:|---:|---:|---:|---:|
| Quail | 41.14 | 4,426.71 | 0.04513 | 7,309 | 4,314,219 | 67.77 |
| Stock vLLM | 90.07 | 2,108.23 | 0.09881 | 194,016 | 4,593,860 | 67.05 |
| Pipelined vLLM | 89.80 | 2,114.57 | 0.09851 | 193,312 | 4,593,156 | 67.05 |
| Pipelined SGLang | 138.26 | 1,238.23 | 0.15167 | 33,904 | 4,096,274 | 69.08 |

- Quail took 41.14 seconds, compared with 39.02 seconds in the earlier run.
  It was 2.19 times faster than stock vLLM,
  2.18 times faster than pipelined vLLM, and
  3.36 times faster than pipelined SGLang.
  The prediction held. One run per method does not measure runtime variation.
- Quail's token totals and accuracy match the earlier shared-retention run.
  Pipelined vLLM and stock vLLM have the same answer agreement and output counts.
  Each method sees different survivors, so the runtime comparison also includes
  differences in the amount of join work.
- SGLang spent 132.27 seconds in the three join generation calls, compared
  with pipelined vLLM's 85.17 seconds. That is 773 microseconds per pair,
  compared with 449. SGLang processed fewer pairs and fresh tokens, so extra
  prefix computation does not explain the slowdown. Startup is excluded.
- These join timers include request handling and GPU execution. The SGLang
  adapter submits at most 16,384 requests per blocking call and uses a different
  pair order to encourage prefix reuse. The saved measurements do not separate
  those effects from engine scheduling and GPU computation. A profile is needed
  to attribute the extra time.

| Method | Evaluated document pairs | Returned rows | Matching reference rows | Output precision (%) | Output recall (%) | Startup seconds |
|---|---:|---:|---:|---:|---:|---:|
| Quail | 182,115 | 149,783,486 | 5 | 0.00000334 | 45.45 | 23.45 |
| Stock vLLM | 189,888 | 172,090,043 | 5 | 0.00000291 | 45.45 | 93.66 |
| Pipelined vLLM | 189,888 | 172,090,043 | 5 | 0.00000291 | 45.45 | 0.00 |
| Pipelined SGLang | 171,197 | 118,565,289 | 5 | 0.00000422 | 45.45 | 313.98 |

- The reference has 11 output rows. Every method returns many false positive
  rows, despite answer agreement between 67% and 70%. Final output precision
  remains very low. The latency improvement does not resolve that accuracy issue.
- Throughput counts evaluated document pairs across all three joins divided
  by query seconds. Cost is query seconds divided by 3,600 and multiplied
  by the H100 price of $3.9492 per hour. It excludes startup.

| Baseline | Available KV tokens | Maximum requests | Filter document admission cap |
|---|---:|---:|---|
| Stock vLLM | 479,616 | 4,096 | All input rows per operator |
| Pipelined vLLM | 479,616 | 4,096 | Claims: 4,096; evidence: 938 |
| Pipelined SGLang | 403,744 | 4,096 | Claims: 4,096; evidence: 802 |

- The horizontal SoL lines credit matching prefixes across requests and
  document aliases. FEV-9's estimate remains 5.821 seconds with unlimited KV
  and reference-label survivors. Methods can produce different answers and
  therefore evaluate different numbers of pairs. Their gap from SoL is not
  purely execution overhead.
- The reference labels are Qwen3 32B FP8 from collection
  `gt_77bb8b128743a79aedddaa24c808c3f8`. The corpus is
  `c_1aa2c4f0d0b6c816fd37aa5748c33341`. All seven saved Quail answer tables
  match the earlier shared-retention run exactly.
- Quail uses the existing analytical settings of 110,376 tokens per chunk
  and 362,240 usable KV tokens after rounding to 16-token pages. All methods
  plan join order before execution from the same selectivity estimates and
  run the first anchor's filters last. Baseline filter admission uses each
  engine's reported KV capacity. No setting was tuned from measured survivors.
- The vLLM configurations use prefix caching, 4,096 maximum requests, and
  25,305 maximum batched tokens. SGLang uses prefix caching, 4,096 maximum
  requests, 25,296 prefill tokens after page rounding, and half its KV capacity
  for each group of join requests. Each FEV-9 alias has only one filter, so filter pipelining
  cannot pass a document directly to another filter on that alias.

## Reproduce

```bash
set -o pipefail
uv run modal run --detach -m quail.bench.quailb_parallel \
  --sf 0.1 --lf 1 --model qwen3-4b-fp8 --query FEV-9 \
  --ground-truth-collection gt_77bb8b128743a79aedddaa24c808c3f8 \
  --prediction "Quail will remain near 39 seconds and beat the baselines; answer agreement will remain near 67.77 percent." \
  2>&1 | tee /tmp/quail-fev9-all-methods.log
```

- Modal app: `quail-milestone1`.
- Parent function call: `fc-01M1W960914BX21C0BNX8RT806`.
- Quail and vLLM function call: `fc-01M1W96BXB4MW1N9W6B1QDBF03`.
- SGLang function call: `fc-01M1W96C2673KKFH0X87JJHF3F`.
- The manifest on `quail-results` is
  `/results/benchmarks/quailb/family-runs/20260906T211500Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/manifest.json`.
  It lists the four summaries. Each query record links its raw report and
  predicate answer tables. The plot download commands are in
  `reports/make_quailb_comparison_plots.py`.
