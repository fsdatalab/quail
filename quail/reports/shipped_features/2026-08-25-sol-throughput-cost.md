# Speed-of-light, docs/second, tokens/second, and $/query in QUAIL-B

Issue: #26

## What changed

Every QUAIL-B query result now reports, alongside wall time:

- **SOL (speed-of-light) seconds**: the theoretical minimum runtime for
  that query's workload, computed from hardware specs alone (roofline
  model in `quail/quail/planner/budgets.py`: `sol_seconds` /
  `sol_seconds_breakdown`, four terms — causal-attention prefill,
  streaming attention, dense projections, and elementwise work).
- **Efficiency**: `sol_s / wall_s`. If this ever exceeds 100%, the run
  aborts with a `SolViolation` — a measured time faster than the
  hardware floor always means a bug in either the SOL formula or the
  wall-time measurement, never a real result.
- **Docs/second and tokens/second** (warm pass, excluding boot).
- **Cost in dollars**, from `gpu_count * (wall_s + boot_s) *
  modal_gpu_rate_per_second`, reported separately for cold (includes
  boot) and warm passes. Rates come from
  `quail/quail/calibration/modal_rates.json` (H100 SXM, non-preemptible).
- A new report table and chart: `quail/quail/bench/sol_report.py`
  (`build_rows`, `render_markdown_table`, `plot_sol_comparison`) turns
  a `run_suite()` result into the Query/Docs/Tokens/Cold/Warm/SOL/
  Efficiency/Docs-per-s/Tok-per-s/Cost table the issue specifies, plus
  a measured-vs-floor bar chart.

## Why

Wall time alone doesn't say whether a query is slow because of software
overhead or because it's already close to the hardware limit, and it
doesn't let users compare queries with different corpus sizes or
estimate what a workload will cost on Modal. SOL answers "how close to
the hardware limit is this", docs/s and tokens/s answer "how fast does
this process documents" in a size-independent way, and $/query answers
the direct cost question.

## Before / after

Before: a QUAIL-B report showed Query ID, description, cold wall (s),
warm wall (s), restored docs — no sense of how good those numbers were
or what they cost.

After, measured on `sf=0.1`, Qwen3-4B fp8, 1 H100
(`results/sol_check_sf0.1_4b_corrected.json`):

| Query | Shape | Pass | Wall (s) | SOL (s) | Efficiency | Cost ($) |
|---|---|---|---|---|---|---|
| IMDB-1 | filter only | warm | 16.51 | 10.195 | 62% | 0.0210 |
| IMDB-2 | join only | warm | 52.42 | 33.058 | 63% | 0.0668 |
| IMDB-5 | 3 filters + join | warm | 20.69 | 11.742 | 57% | 0.0264 |
| BIO-2 | join only | warm | 142.87 | 87.717 | 61% | 0.1821 |
| FEV-5 | 2-sided filter + join | warm | 2.01 | 1.056 | 53% | 0.0026 |

Efficiency sits in the 53-63% range across all five query shapes tried
(filter-only, join-only, filter chains, two-sided filter+join, two
different datasets) — well under the 100% ceiling the SOL invariant
enforces, and in the same 30-90% range the earlier cost model measured
for the old serving loop.

A three-phase verification pass (formula math sanity via property
tests, formula behavior visualized via diagnostic plots, and real
`torch.profiler` kernel traces cross-checked against the formula's
per-term breakdown) found the aggregate SOL bound holds on every query
shape tried, though attention costs relatively more and elementwise
work relatively less than their individual formulas predict — a
tightness note for future work, not a correctness problem. See
`reports/2026-08-23-sol-throughput-cost.md` for the full derivation,
two rounds of adversarial review (6 bugs found and fixed), and the
verification detail.

## Known gaps (not blocking)

- `$/query`'s CPU term assumes the worker doesn't request extra CPU
  cores beyond the GPU's default allocation; if that changes, the CPU
  core-count term needs adding.
- `modal_rates.json` is a hardcoded snapshot, not a live
  `Workspace.billing.rates()` call — no runtime cross-check against
  Modal's actual billed rate yet.
