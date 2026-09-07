# Simpler repository layout and no standalone stock runners

Date: 2026-09-05.

## What changed

- Every Modal entry point that costs GPU time lives under
  `experiments/`: the ablation and profiling scripts at the top level,
  and the smokes, probes, and gates under `experiments/cells/`.
  `tests/` is the CPU suite only. The folders `ablations/`,
  `tests/gpu/`, `migrations/`, and `plans/` are gone. The design
  decisions and the speed of light model moved into the docs site
  under its architecture section.
- `results/` holds no committed JSON any more. Experiment data lives
  on the `quail-results` volume, and the directory stays only for
  teed logs, which are gitignored.
- `baselines/` is gone. It held the older standalone stock vLLM and
  SGLang runners (`stock_vllm/run.py`, `stock_sglang/run.py`,
  `stock_boot.py`, `stock_quailb.py`, and the archived `old_stock/`).
  The request backends under `quail/backends/` (`stock_vllm`,
  `pipelined_vllm`, `pipelined_sglang`) are the only comparison code,
  and the same-GPU benchmark runner has produced every family file
  the reports read from them. The tests that covered the shared
  scheduling loops moved to `tests/test_request_scheduling.py` and
  now import `quail.backends.request_scheduling` directly. The
  tests that covered the standalone runner's own request scheduling
  (query-set splitting, paired ordering, failure entries) went with
  it.
- `experiments/profile_stock.py` went with the runner. It wrapped the
  runner's step loop for profiling, and the system it measured no
  longer exists. The 2026-08-30 discrepancy report still cites its
  data on the volume.
- `pyproject.toml` packages only `quail`. The engine never imported
  `baselines`, so nothing under `quail/` changed.

## Why

The layout had five places for GPU-costing code and two copies of the
stock comparison. One folder for experiments and one implementation
of each backend is easier to explain in the contributing guide and
leaves less to keep in sync.

## Before and after

| | Before | After |
| --- | --- | --- |
| Top-level folders with Python | `quail`, `quail_ext_examples`, `tests`, `tests/gpu`, `ablations`, `baselines`, `migrations`, `reports` | `quail`, `quail_ext_examples`, `tests`, `experiments`, `reports` |
| Stock vLLM implementations | 2 (`baselines/stock_vllm`, `quail/backends/vllm.py`) | 1 |
| SGLang implementations | 2 | 1 |
| CPU tests | 242 | 218 (24 covered only the removed runner) |
