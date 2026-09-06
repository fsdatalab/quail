# SGLang FEV-9 join profile

- All three joins' PyTorch traces point to CPU scheduling and result handling
  as the first places to investigate. GPU kernels and transfers covered
  42.11 of their combined 164.26 seconds, or 25.6%.
- Both SGLang and vLLM continuously add queued requests to GPU batches.
  Both reuse prefix KV. Our adapters submit every pair for one join in a
  single call, with no client batch barrier between subsets of that join.
- The profile measures SGLang alone. It identifies expensive parts of
  SGLang's execution, but does not measure how much faster each corresponding
  part is in vLLM. Profiling also adds overhead.

## Setup and prediction

- FEV-9 has four filters followed by three joins. Input relations are
  `c1 = 500`, `c2 = 500`, `e1 = 287`, and `e2 = 287` document rows.
  The repeated aliases refer to 787 distinct documents.
- One Modal H100 runs `Qwen/Qwen3-4B-FP8`, with FP8 weights and BF16 KV.
  SGLang is pinned to 0.5.18. The model identifier matches the saved
  vLLM 0.26.0 baseline. The profile result records the loaded model revision
  and quantization configuration.
  The loaded revision was `96b30dc13593a244a5e59e84687309f53c375cfa`,
  with dynamic FP8 E4M3 quantization and 128 by 128 weight blocks.
  PyTorch was `2.13.0+cu130`.
- SGLang settings match the saved unprofiled run. Static memory fraction
  is 0.76, measured KV capacity is 403,744 tokens, page size is 16 tokens,
  and at most 4,096 requests run at once. Prefill limits are 25,296 tokens.
  Prefix reuse is enabled and prefill CUDA graphs are disabled.
- SGLang submits every anchor for one partner before advancing to the next
  partner. Sampling uses temperature zero, one output token, and a +1,000
  logit bias for the TRUE/FALSE token IDs. vLLM uses its allowed-token list.
  These are the existing baseline settings.
- vLLM's saved configuration allowed 25,305 batched tokens and 4,096
  active requests, with 479,616 tokens of measured KV capacity. Its GPU
  memory utilization setting was 0.91. SGLang's lower KV capacity did not
  produce more fresh tokens in this comparison.
- Before the run, we predicted that CPU request handling and scheduling
  would leave the GPU inactive for more than half of each join. We also
  predicted that answers and token counts would match the saved run.
- PyTorch Profiler records CPU and CUDA events in the scheduler and a CPU
  interval around each join in the driver. Model startup and all filters
  run outside the profiler. Each trace is exported before the next join.
  The analysis aligns process clocks and restricts events to the driver's
  join interval, excluding trace export.
- GPU activity means elapsed time covered by at least one kernel or memory
  transfer. Overlapping GPU operations count once. It does not measure how
  many GPU cores were occupied. CPU intervals can overlap GPU activity.

## Where the time went

![FEV-9 GPU activity and scheduler CPU intervals](plots/sglang_join_profile.png)

Figure: plots/sglang_join_profile.png

| Join | Driver interval seconds | GPU active seconds | GPU active share | Seconds before first GPU operation |
|---|---:|---:|---:|---:|
| 1 | 56.38 | 14.48 | 25.7% | 13.90 |
| 2 | 55.41 | 14.09 | 25.4% | 13.03 |
| 3 | 52.46 | 13.54 | 25.8% | 11.68 |

The following CPU durations count only the portions with no concurrent GPU
kernel or transfer. The figure shows the full CPU intervals instead.

| Join | Batch selection while GPU idle, seconds | Result processing while GPU idle, seconds |
|---|---:|---:|
| 1 | 19.45 | 7.13 |
| 2 | 20.07 | 6.80 |
| 3 | 20.20 | 6.56 |
| Total | 59.72 | 20.50 |

- The prediction held for all three joins. The GPU had no recorded work
  during roughly three quarters of the captured join time. The long initial
  delays and repeated later gaps make CPU work a stronger immediate target
  than changing model matrix multiplication kernels.
- The scheduler's batch selection interval includes checking for work when
  the queue is empty. For example, join 1 recorded 85,977 calls to
  `scheduler.get_next_batch_to_run`, but only 92 calls to `scheduler.run_batch`.
  The selection time cannot all be attributed to finding reusable prefixes.
- Join 1 recorded 692,655 `aten::copy_` calls, 572,783 `cudaMemcpyAsync`
  calls, and 465,352 `cudaStreamSynchronize` calls. Their inclusive CPU
  durations were 9.77, 3.81, and 2.83 seconds respectively. These nested
  calls overlap other measured intervals and must not be added to them.
- The longest single GPU kernel category in join 1 was a DeepGEMM FP8
  matrix multiplication implementation, with 7.28 seconds summed over
  10,692 launches. The 14.48 seconds of total GPU activity includes those
  launches. Kernel sums and elapsed GPU activity have different definitions.
- Every join began with 11.68 to 13.90 seconds without a GPU operation.
  The driver used 18.06 to 19.80 CPU seconds across each full join.
  Request preparation and delivery are candidates for this initial delay;
  the driver's single interval does not separate those functions.
- The next targeted measurement should separate request preparation,
  prefix matching, KV bookkeeping, and result delivery inside the two
  expensive scheduler intervals. A corresponding vLLM trace is needed to
  attribute the exact 1.51 times query latency difference between engines.
  The current trace does not establish that a particular prefix data
  structure or GPU kernel causes that difference.

## Profiling overhead and correctness

| Join | Pairs | Unprofiled generation seconds | Profiled generation seconds |
|---|---:|---:|---:|
| 1, e1 with c1 | 58,136 | 43.22 | 55.32 |
| 2, e1 with c2 | 58,136 | 45.76 | 55.10 |
| 3, e2 with c2 | 54,925 | 42.10 | 52.18 |
| Total | 171,197 | 131.08 | 162.60 |

- Profiling increased total generation time by 24.1%. The driver intervals
  plotted below also include prompt construction and result conversion
  around `generate()`, so they are slightly longer than this table.
- All four filter answer tables and all three join answer tables match
  the saved unprofiled run after sorting. Accuracy counters, returned row
  count, fresh tokens, cached tokens, and recomputed prefix tokens also match.
- The recorded query clock was 942.86 seconds because it includes trace
  export between joins. With export included, throughput was 181.57 pairs
  per second and GPU cost was $1.03432, excluding 311.57 seconds of startup.
  Those are diagnostic run costs, not replacement benchmark measurements.
- The Modal function completed and committed the traces to the volume.
  The existing process cleanup helper left 4 MiB allocated on the GPU.

## Saved benchmark comparison

The unprofiled measurements remain the benchmark results. We did not rerun
vLLM or update the QUAIL-B comparison figures for this diagnostic experiment.

| Method | Query seconds | Evaluated pairs | Document pairs/second | $/query |
|---|---:|---:|---:|---:|
| Pipelined vLLM | 89.80 | 189,888 | 2,114.57 | 0.09851 |
| Pipelined SGLang | 135.74 | 171,197 | 1,261.21 | 0.14891 |

- SGLang's three joins took 131.08 seconds, compared with 85.17 seconds
  for vLLM. SGLang evaluated 9.8% fewer pairs and computed 11.3% fewer
  fresh join tokens, so more input computation does not explain the gap.
- Across the full query, SGLang computed 4,095,266 fresh input tokens,
  including 33,904 recomputed prefix tokens. vLLM computed 4,593,156 fresh
  input tokens, including 193,312 recomputed prefix tokens.
- A fresh token is an input token position processed by a model forward
  pass rather than read from existing KV. Repeated computation counts again.
  Recomputed prefix tokens are included in fresh tokens.
- SGLang agrees with the saved Qwen3 32B FP8 predicate labels on 69.08%
  of evaluated answers, compared with 67.05% for vLLM. Different filter
  answers account for the different pair counts.
- SGLang returned 118,565,289 rows, of which 5 matched the 11 reference
  output rows. Final output precision is 0.00000422% and recall is 45.45%.
  vLLM returned 172,090,043 rows, also with 5 reference matches, giving
  0.00000291% precision and 45.45% recall.
  Predicate agreement should not be read as final join output accuracy.
- GPU costs exclude startup and use `H100_USD_PER_HOUR = 3.9492` from
  `quail.bench.evaluate`. Throughput is evaluated join pairs divided by
  the full query runtime.

## Reproduction and sources

```bash
uv run modal run --detach experiments/profile_sglang_join.py \
  2>&1 | tee /tmp/quail-sglang-join-profile.log
```

- Modal function call `fc-01M1WET35ZK2NX49RZ634J1SKA`.
- Profile summary on `quail-results` at
  `/results/ablations/sglang-join-profile-20260906T225320Z/result.json`.
- Scheduler traces under the same directory at
  `join-0/join-0-TP-0.trace.json.gz`, `join-1/join-1-TP-0.trace.json.gz`,
  and `join-2/join-2-TP-0.trace.json.gz`. Each join directory also contains
  `driver.trace.json.gz`.
- Unprofiled SGLang summary at
  `/results/benchmarks/quailb/families/20260906T222629Z-sglang-suffix-major/fever-sglang-process.json`.
- Saved vLLM results are listed in
  `/results/benchmarks/quailb/family-runs/20260906T211500Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/manifest.json`.
- [Baseline report](2026-09-06-sglang-baseline.md) has the complete comparison
  and dataset figures. [Plot script](make_sglang_join_profile_plots.py)
  includes the volume download commands and rebuilds this diagnostic figure.
- SGLang's [profiling documentation](https://docs.sglang.io/docs/developer_guide/benchmark_and_profiling)
  describes CPU and GPU traces. Its
  [scheduler source](https://github.com/sgl-project/sglang/blob/v0.5.18/python/sglang/srt/managers/scheduler.py)
  defines the measured batch selection and result processing intervals.
