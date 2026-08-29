# QUAIL-B selectivity estimates

## What changed

Every QUAIL-B filter and join now has a fixed selectivity estimate. Each
QUAIL-B builder query asks the planner to use cost based order. Other builder
queries can request the same behavior with
`.select(..., order="by_cost")`.

The estimates are the TRUE fraction from the sf0.1 Qwen3 32B fp8 labels in
collection `gt_363b5ab570635c33894e1a030c21f57e` for corpus
`c_3bd14ed0758287cba9d88fb68de8b7b8`. The benchmark uses the same estimates
at every scale factor. A benchmark run does not read ground truth while
planning.

The source collection is
`/results/ground_truth/quailb/schema_v1/collections/gt_363b5ab570635c33894e1a030c21f57e/manifest.json`
on the `quail-results` volume.

## Why

The benchmark previously forced written order for every query. The planner
therefore did not exercise filter ordering or join ordering from the supplied
cost model.

## Verification

All 30 default queries now plan with `by_cost`. The unit test checks the
selected order rule for every query.

With the updated SoL formula, total 4B SoL time across the suite decreases
from 327.10 to 324.65 seconds. Total 32B SoL time decreases from 2,490.56 to
2,470.11 seconds. The current result is at
`/results/sol/sol_quailb_sf0.1.json` on the `quail-results` volume. These are
analytical results. A confirming GPU run has not been made yet.
