# FEV-9 filters all four join inputs

FEV-9 applies F11 to both claim inputs, c1 and c2, and F13 to both evidence
inputs, e1 and e2. F11 asks whether a claim is about a person. F13 asks whether
an evidence passage primarily describes a person.

The filtered inputs feed the existing three joins: e1 supports c1, e1 refutes
c2, and e2 supports c2. Previously, only c1 had a filter. The query now covers
filtering both document collections before multiple joins.

The prompts, ground-truth labels, and selectivity estimates already exist.
No new labels or source data are needed. The suite still has 32 default queries.

CPU checks verify planning on all four backends and the joined result using
fixed predicate answers. No GPU benchmark was run for the extended query.

The old full-suite comparison and speed-of-light reports, along with their
plots and plot scripts, were removed because their FEV-9 results and suite
totals describe the previous query. The data remains on the `quail-results`
volume under the following paths:

- `/results/benchmarks/quailb/family-runs/20260905T021527Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/manifest.json`
- `/results/sol/sol_quailb_sf0.1.json`

Those saved results must be regenerated before comparing the current full suite.
