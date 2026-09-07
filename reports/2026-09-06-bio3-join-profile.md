# BIO-3 pipelined vLLM join profile

- GPU operations covered 295.07 of 965.50 seconds in the profiled join,
  or 30.56%. There were 670.43 seconds without a recorded GPU operation.
  The prediction of substantial GPU idle time held in the profiled run.
- Batch selection overlapped 160.47 seconds of GPU idle time. The recorded
  scheduler operations explain only part of the idle time. This trace
  does not establish a complete breakdown of request preparation,
  communication, or other Python work.

[Open the vector PDF](plots/bio3_join_profile.pdf)

[![BIO-3 join GPU and scheduler activity](plots/bio3_join_profile.png)](plots/bio3_join_profile.pdf)

Figure: plots/bio3_join_profile.png

## Setup

- The experiment runs BIO-3 with Qwen3 4B FP8 on one H100. BIO-3 filters
  reports for female patients, then joins surviving reports with reaction
  terms. Input counts are 500 reports and 1,127 terms, at sf=0.1 and lf=1.
- The filter runs normally before profiling starts, preserving its answers
  and KV. PyTorch Profiler records the entire join in the vLLM worker with
  CPU and CUDA activities. A separate CPU trace records the driver's join
  interval. Stack and shape recording are disabled.
- The configuration uses anchor-major pair submission, prefix reuse,
  `max_num_seqs=4096`, `max_num_batched_tokens=25305`, and
  `gpu_memory_utilization=0.91`. Profiling does not change these settings.
- The prediction is that request handling and scheduling leave substantial
  GPU idle time during the join. The saved unprofiled vLLM join took 474.04
  seconds for 311,052 pairs.
- GPU active time is the union of CUDA kernel, copy, and memset intervals.
  Overlapping operations count once. It measures elapsed GPU work, not
  the fraction of GPU cores in use. CPU scopes can overlap GPU activity.
- The profiled join interval excludes trace export. The full query clock
  includes export because profiling wraps the join inside query execution.
  The profile is diagnostic and does not replace the saved benchmark in
  the main QUAIL-B or BIO figures.

## Saved benchmark

| Metric | Quail | Pipelined vLLM |
|---|---:|---:|
| Query time excluding startup, seconds | 89.92 | 510.91 |
| Evaluated document pairs | 308,798 | 311,052 |
| Document pairs per second | 3,434.14 | 608.82 |
| GPU cost per query, USD | 0.09864 | 0.56047 |
| Fresh input tokens | 7,547,348 | 7,875,694 |
| Recomputed KV tokens | 919,409 | 1,079,840 |
| Predicate answer agreement, percent | 82.54 | 81.23 |
| Final output precision, percent | 14.733 | 13.912 |
| Final output recall, percent | 79.917 | 81.171 |

- Quail was 5.68 times faster. Its filter kept 274 reports, compared with
  276 for vLLM. vLLM evaluated 0.73% more pairs and computed 4.35% more
  fresh tokens. Quail's saved result does not separate filter and join time.
- vLLM spent 25.36 seconds in the filter and 474.04 seconds in join
  generation. The join accounted for 92.78% of the full query time.
- Fresh input tokens are token positions computed by a model forward pass
  instead of read from KV. Repeated computation counts again. Recomputed
  KV tokens are included in the fresh-token total.
- Accuracy uses the saved Qwen3 32B FP8 reference labels. Throughput divides
  evaluated pairs by query seconds. Cost divides query seconds by 3,600
  and multiplies by `H100_USD_PER_HOUR`, $3.9492. Startup is excluded.

## Profiled repeat

| Recorded operation | Elapsed seconds | Seconds while GPU idle |
|---|---:|---:|
| Choose next batch | 189.47 | 160.47 |
| Add request to scheduler | 24.94 | 14.30 |
| Process model answers in scheduler | 51.41 | 21.82 |

- The driver trace's join annotation spans 965.50 seconds. Its small
  difference from the wrapper's 965.53-second timer is profiler context
  setup and teardown. GPU activity is clipped to the annotation's
  interval using the absolute timestamps in both traces.
- The first GPU operation begins 36.87 seconds after the join annotation
  starts. The driver records only the complete join scope, so it cannot
  assign that initial delay to individual Python functions.
- The trace records 3,549 scheduler calls and 311,052 calls to add a
  request. GPU idle gaps continue throughout the join; they are not
  confined to its initial 36.87 seconds.
- The measured GPU activity fraction is specific to this profiled run.
  Generation was almost twice as slow as the saved baseline, so applying
  the 30.56% fraction to the original 474.04-second join would be invalid.
  There is no matching BIO-3 Quail trace in this experiment.
- The loaded model was `Qwen/Qwen3-4B-FP8`, revision
  `96b30dc13593a244a5e59e84687309f53c375cfa`. The run used vLLM 0.26.0
  and PyTorch 2.11.0+cu130. KV capacity was 479,616 tokens in 16-token
  blocks, matching the baseline.
- The filter took 25.16 seconds and kept the same 276 reports. All 500
  filter answers match the baseline exactly. The join evaluated the same
  311,052 pairs against 1,127 terms.
- Join generation took 923.83 seconds, compared with 474.04 seconds in
  the unprofiled baseline. The profiled generation interval was 94.89%
  longer. The complete join wrapper took 965.53 seconds, including prompt
  construction and answer processing but excluding trace export.
- Fresh input tokens, cached tokens, and recomputed KV tokens match the
  saved baseline exactly. Join-only fresh computation was 5,818,365 tokens,
  with 1,079,840 recomputed KV tokens. Whole-query fresh computation was
  7,875,694 tokens.
- Fifteen join answers changed: nine true answers became false, and six
  false answers became true. Returned rows fell from 66,094 to 66,091.
  The same 9,195 output rows match the reference. Profiling and run-to-run
  scheduling can change execution, but this experiment does not identify
  the cause of these answer changes.
- Predicate answer agreement was 81.2298%, compared with 81.2288% before.
  Final output precision was 13.9126%, and recall was 81.1706%.
- The full profiled query clock was 1,632.40 seconds, including trace
  export and excluding startup. That corresponds to 190.55 pairs per
  second and $1.79074 per query. These are diagnostic run costs, not
  replacements for the benchmark. Model startup took a separate 166.15
  seconds. The worker process group was stopped after saving the result;
  GPU memory returned to 4 MiB.

## Reproduction

```bash
uv run modal run --detach experiments/profile_vllm_join.py --query BIO-3 \
  2>&1 | tee /tmp/quail-bio3-vllm-join-profile.log
```

- Modal function call: `fc-01M1WQYGGP93R7B5TSMEQAA0CG`.
- Profile directory on `quail-results`:
  `/results/ablations/vllm-join-profile-20260907T013304Z/`.
- Profile summary: `result.json` within that directory. The CPU/CUDA
  worker trace is `join-0/worker.trace.json.gz`, and the driver CPU trace
  is `join-0/driver.trace.json.gz`. The worker trace is 229,013,486 bytes.
- Profiled answers and engine report:
  `/results/benchmarks/quailb/runs/qb_20260907T013342Z_31dd2183/single/BIO-3.json`.
  Its `answer_tables` lists the filter and join Parquet files. We compared
  both tables with the baseline after sorting by document identifiers.
- Saved Quail baseline:
  `/results/benchmarks/quailb/runs/qb_20260905T021548Z_43ca0948/single/BIO-3.json`.
- Saved pipelined vLLM baseline:
  `/results/benchmarks/quailb/runs/qb_20260905T024836Z_a36d4647/single/BIO-3.json`.
- The download and plotting commands are in
  [make_bio3_join_profile_plots.py](make_bio3_join_profile_plots.py).
