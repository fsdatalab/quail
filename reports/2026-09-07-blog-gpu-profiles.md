# BIO-3 and AGENT-1 GPU profiles

- The BIO-3 windows show 4.997 seconds of GPU activity with Quail and
  1.892 seconds with pipelined vLLM, out of five seconds each.
- AGENT-1 keeps the GPU active for nearly all five seconds with both
  backends. Quail computes 17,389,113 fresh tokens, compared with vLLM's
  5,526,889. The extra computation, rather than GPU idle time, explains
  Quail's longer execution on this query.
- BIO-3 reuses the earlier vLLM trace. Quail's two profiles and the
  AGENT-1 vLLM profile are new. All use H100s, but the comparison uses
  different physical cards. The saved latency measurements remain unchanged.

[![BIO-3 GPU and CPU comparison](plots/bio3_profile_comparison.png)](plots/bio3_profile_comparison.pdf)

Figure: plots/bio3_profile_comparison.png

[![AGENT-1 GPU and CPU comparison](plots/agent1_profile_comparison.png)](plots/agent1_profile_comparison.pdf)

Figure: plots/agent1_profile_comparison.png

## Setup and prediction

- Qwen/Qwen3-4B-FP8, revision
  `96b30dc13593a244a5e59e84687309f53c375cfa`, one H100 per run,
  vLLM 0.26.0 and PyTorch 2.11.0+cu130. Scale factor 0.1, length factor 1.
- BIO-3 filters 500 reports, then joins survivors with 1,127 reaction
  terms. Quail evaluates 308,798 pairs from 274 surviving reports; the
  reused vLLM run evaluates 311,052 pairs from 276 survivors.
- AGENT-1 applies one filter to 1,772 agent trace snapshots. It has no join.
- Quail uses a 110,376-token chunk budget and 362,240 KV tokens. vLLM
  uses 479,616 KV tokens, a 25,305-token batch budget, and at most 4,096
  sequences. Both use 16-token KV blocks. The AGENT-1 vLLM admission limit
  is 48 documents, as derived by the existing baseline. No settings changed
  for the profiles.
- The prediction was less GPU idle time with Quail during BIO-3's join,
  and fewer fresh tokens with vLLM on AGENT-1 through shared prefix reuse.
  Both predictions held. The capture excludes model loading and warmup.
  BIO-3's preceding filter runs normally and retains its KV for the join.
- Quail runs the current engine from `0715099`. A benchmark reporting bug
  introduced with `ScanInput` was fixed in `8d8fd12`: prefix accounting now
  reads the scan's `.tokens`. The engine itself was unchanged.

## Recorded activity

| Query | Backend | Profiled phase, seconds | GPU active, seconds | GPU idle, seconds | Figure window, seconds after phase start |
|---|---|---:|---:|---:|---|
| BIO-3 | Quail | 68.31 | 67.05 | 1.26 | 31 to 36 |
| BIO-3 | Pipelined vLLM, reused | 965.50 | 295.07 | 670.43 | 480 to 485 |
| AGENT-1 | Quail | 238.55 | 238.40 | 0.15 | 116 to 121 |
| AGENT-1 | Pipelined vLLM | 98.49 | 97.90 | 0.59 | 46 to 51 |

- GPU active time is the union of recorded kernels, copies, and memsets.
  Overlapping operations count once. It measures whether GPU work is
  running, rather than how fully the GPU's arithmetic units are used.
- New five-second windows start at `floor(phase_seconds / 2 - 2.5)`.
  The existing BIO-3 vLLM window is preserved. Each plot starts its window
  at zero for comparison; the table gives the original offsets.
- Quail captures `quail.executor.loop.run_join` or `run_filter`.
  vLLM captures `quail.backends.request.run_join_grouped` or
  `_pipelined_filter`. The vLLM join scope includes request construction;
  Quail's scope starts after preparing its input views.
- CPU calls keep their original names and nesting. A CPU wait in
  `cudaEventSynchronize` can overlap GPU computation. Gray means no selected
  CPU operation was recorded; it does not establish that the CPU was idle.
- Profiling adds overhead. In the old BIO-3 vLLM run, generation took
  923.83 seconds under profiling, compared with 474.04 seconds without it.
  The CPU gaps can therefore be longer in the profile. The plots do not
  estimate the fraction of unprofiled execution spent idle. Trace export
  is excluded from the phase times above.

## Saved benchmark measurements

These are the existing measurements used in the blog, without profiling.
No other benchmark queries were rerun and no benchmark bars were replaced.

| Query | Backend | Query seconds | Throughput | $/query |
|---|---|---:|---:|---:|
| BIO-3 | Quail | 89.92 | 3,434.14 pairs/s | 0.09864 |
| BIO-3 | Pipelined vLLM | 510.91 | 608.82 pairs/s | 0.56047 |
| AGENT-1 | Quail | 240.49 | 7.37 documents/s | 0.26382 |
| AGENT-1 | Pipelined vLLM | 99.15 | 17.87 documents/s | 0.10877 |

Throughput divides evaluated pairs or input documents by query time. Cost
uses `quail.bench.evaluate.H100_USD_PER_HOUR`, $3.9492, and excludes startup.
The saved measurements are indexed by the volume manifest
`/results/benchmarks/quailb/family-runs/20260905T021527Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/manifest.json`.

## Work and accuracy in the profiles

Accuracy is agreement with saved Qwen3 32B FP8 labels on evaluated answers.
Fresh tokens include every input token processed by a forward pass, including
repeated prefix computation. Recomputed KV counts a document's own prefix
computed again; shared prefixes across different documents are separate.

| Query | Backend | Fresh tokens | Recomputed KV tokens | Answer agreement | Output precision | Output recall |
|---|---|---:|---:|---:|---:|---:|
| BIO-3 | Quail | 7,547,851 | 919,912 | 82.54% | 14.73% | 79.92% |
| BIO-3 | Pipelined vLLM, reused | 7,875,694 | 1,079,840 | 81.23% | 13.91% | 81.17% |
| AGENT-1 | Quail | 17,389,113 | 0 | 75.00% | 68.60% | 41.33% |
| AGENT-1 | Pipelined vLLM | 5,526,889 | 0 | 74.15% | 65.92% | 40.98% |

AGENT-1 has 11,882,610 shared prefix tokens across snapshots. vLLM reuses
11,862,224 tokens across rows. Quail does not reuse those prefixes across
different document identities. A nearly fully active GPU therefore does
not imply that the backend avoided repeated work.

## Saved artifacts and reproduction

- Quail profiles and answers are under
  `/results/ablations/blog-profiles-20260907T211904Z/quail/` on `quail-results`.
  Each query has `phase.json`, `worker.trace.json.gz`, and `analysis.json`.
  The original capture code is saved beside them as `capture-cell.py` and
  `capture-worker.py` in the parent directory. Call
  `fc-01M1YVT4BH36VC12781A63TWDM` completed Quail before the redundant vLLM
  BIO-3 run was stopped. GPU `GPU-f63b9f4d-9397-5aa6-8d36-1f772ca534ca`.
- The reused BIO-3 vLLM artifacts are under
  `/results/ablations/vllm-join-profile-20260907T013304Z/`.
  `blog-analysis.json` reuses the existing `cpu-window.json`,
  `gpu-timeline.f64.gz`, and `cpu-flamegraph.json`. The original trace is
  `join-0/worker.trace.json.gz`. Call `fc-01M1WQYGGP93R7B5TSMEQAA0CG`;
  GPU `GPU-ab3adb16-c768-33cc-2274-9948c24e0388`.
- AGENT-1 vLLM artifacts are under
  `/results/ablations/vllm-join-profile-20260907T214358Z/`, including
  `result.json`, `blog-analysis.json`, and `filter-0/worker.trace.json.gz`.
  Call `fc-01M1YX7S7QK7K6SXC6D6K8PP3H`;
  GPU `GPU-97bbc489-ce4b-8667-7ff8-0453eb31a605`.
- Run `reports/make_blog_profile_plots.py` with the downloaded workdir as
  its first argument. Its docstring lists the four volume downloads.
  New summaries use `experiments/analyze_blog_profiles.py`.
  The existing vLLM profile also remains available in
  [the earlier HTML](plots/bio3_join_profile.html).
