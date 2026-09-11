# 2026-09-11: FEV-10 is a benchmark query, with two runner fixes

## What changed

- QUAIL-B has 33 queries. FEV-10 (FEV-5's two filters, then SUPPORT
  asked only where `c.evidence_wiki_url = e.id`) merged in
  `fsdatalab/quail-bench` as commit `ec4682f2`, and this repository pins
  that commit. No labeling run was needed at any scale factor: the
  three predicates are already labeled in the sf=0.1, 0.5, and 1.0
  collections, and scoring applies the equality to the pair labels.
- FEV-10 has all four measurements. The family runner ran it once with
  Quail, stock vLLM, pipelined vLLM, and pipelined SGLang on one H100
  at sf=0.1 (run directory
  `/results/benchmarks/quailb/family-runs/20260911T201441Z-d16f87d8/`,
  function call `fc-01M291QA9AAV9JSMYN5KCRJSM4`). The main and FEVER
  plots and the saved-results report include it, and the SoL file for
  all 33 queries is `/results/sol/2026-09-11-quailb-prefix-reuse.json`.
- `run_all` in `quail/bench/quailb_parallel.py` no longer resolves the
  run directory before checking that it lies under `/results`. Inside
  the container `/results` is a mount whose real path is elsewhere, so
  the check refused every run directory since it arrived with #89. The
  first FEV-10 attempt (`fc-01M28Z7H6F8X634WWVQHMX1DGC`) failed on it.
- `gpu_problem()` in `quail/runtime/execute.py` asks PyTorch for its
  NVML based GPU check instead of `torch.cuda.is_available()`. The
  default check initializes the CUDA runtime and marks every later
  fork as bad; vLLM forks its engine process after the probe, so every
  request-backend run since the probe arrived failed with "Cannot
  re-initialize CUDA in forked subprocess". The second FEV-10 attempt
  (`fc-01M29181E51A1T4E2840AGD2XT`) failed on it, after Quail's own
  part had finished. Quail's backend never forks, which is why the
  Quail-only cells kept working.

## Numbers

| Configuration | FEV-10 seconds | FEV-5 seconds | FEV-10 fresh tokens |
|---|---:|---:|---:|
| Quail | 1.68 | 13.47 | 187,567 |
| Stock vLLM (operator-at-a-time) | 2.97 | 32.66 | 270,220 |
| Pipelined vLLM | 2.91 | 31.73 | 270,220 |
| Pipelined SGLang | 3.32 | 43.43 | 267,414 |

Predicted before the run: 3 to 5 seconds for the vLLM configurations
and 4 to 7 for SGLang. The details are in
`reports/2026-09-11-pair-join.md`.
