# Multi-join benchmarks: the gate, and the re-shard trade

Date: 2026-08-24. Issue #38's benchmark slice, minus the 3-way
cross-product comparison (its arithmetic is not in doubt). Engine at
PR #42's head. qwen3-4b-fp8, one H100, through Modal. Store disabled
(`cpu_memory_gb=0`) so run order cannot contaminate comparisons.
Builder queries, so stage order is as written.

Corpora are planted: every document carries one color drawn
independently and uniformly from six, so the per-pair selectivity is
exactly 1/6 and the gate formula's independence assumption holds
exactly. Script: `tests/gpu/join_bench.py` (it prints every
prediction before running and regenerates the summary). Committed
summary: `results/join_bench.json`. Teed log:
`results/join_bench.log` (local, not committed). The workers' raw
run records land on the `quail-results` volume under `/results/runs/`.

The whole suite ran twice; the gate cell reproduced bit-for-bit
(same live counts, same selectivities, same 779,857 fresh tokens),
so the constrained TRUE/FALSE readout is deterministic across
container boots.

## Cell 1: the gate, measured against the formula

Setup: a star of three joins on one table b (120 documents, ~800
tokens each; at two joins chain and star are the same graph, and
three stages give two gate checkpoints). Partner tables: 4, 4, and
120 documents. The plan is one JoinGroup with three stages anchored
on b, no barrier (asserted before running).

Prediction, two levels:

- The formula expectation `n(1-(1-s)^partners)`: 62.1 ± 5.5 anchors
  entering stage 2, 32.2 ± 4.9 entering stage 3.
- The conditional truth of the drawn corpus, knowable from the seed
  before running: 37 and 20. The gap to the formula is real
  variance, not error: the first partner table's four color draws
  landed on only 2 distinct colors of 6 (the expectation is ~3.1),
  so fewer anchors can match.

Measured: 120 → 37 → 20 live anchors — the conditional truth
exactly. Stage tuples were 480, 148, 2400, equal to live × partners
at every stage. Observed selectivities at stages 1 and 2 were 0.1479
and 0.2703, the planted values to four decimals: the model made zero
errors on those stages.

Figure: plots/join_gate_formula.png

What it means: the gate mechanics are exact, and the formula is what
it claims to be - an expectation over corpus draws. The planner uses
it for cost only, where expectation level is the design intent; on
partner tables of a few documents, expect single-query deviations of
this size.

One accuracy caveat: stage 3's observed selectivity was 0.825
against a planted 0.1667. Its partner documents bury the color line
behind twelve filler sentences, and the model over-answers TRUE on
that shape - the same failure the session smoke documented for
buried-fact predicates. This does not affect the gate measurement
(the tuple arithmetic above is exact regardless of what the answers
are), but it inflates the final row count.

## Cell 2: the re-shard trade

Setup: a chain `ai(a, b)` then `ai(b, c)`. Table a: 40 documents of
~3,000 tokens. Tables b and c: 60 short documents each. Hints 1/6 on
both joins. Two configurations of the same query on the same data:

- Baseline: anchors forced onto b, the shared table - one group, the
  only plan the engine could run before PR #42, configured
  explicitly per the house rule that baselines get the analytically
  equivalent setting.
- Free: the planner picks. It chose a for stage 1 and c for stage 2:
  two groups with one barrier (asserted before running).

Prediction: token ratio ~11x (the two plans' own arithmetic:
6,024,123 vs 534,486 tokens), wall ratio above 4x.

Measured:

| configuration | fresh tokens | wall |
|---|---|---|
| shared anchor (forced, one group) | 6,024,300 | 58.4 s |
| planner's choice (two groups, barrier) | 534,660 | 5.2 s |

11.27x fewer tokens and 11.27x faster, and each plan's predicted
token total was within 0.03% of the measured count. Paying the
barrier - fresh KV for the second anchor, plus the plumbing - beats
streaming a 3,000-token document once per pair, by an order of
magnitude, exactly as the cost model prices it.

Figure: plots/join_reshard_cost.png

## The accuracy finding: anchor orientation is not free

Both configurations scored the same 2,400 stage-1 pairs with the
same question text; only the anchor differed. The results diverged
completely:

- Baseline (short document anchored, the 3,000-token document
  streaming as the partner): stage selectivities 0.1646 and 0.1675 -
  the planted truth to four decimals - and 4,046 final rows, the
  planted count exactly.
- Free (the 3,000-token document anchored, the short document
  streaming): stage-1 observed selectivity 1.0 - the model answered
  TRUE to every pair - and 20,280 final rows, 5x the truth. Stage 2
  drifted too (0.1408 vs 0.1675).

Figure: plots/join_orientation_accuracy.png

The engine is identical in both runs; recombination is checked
against brute force in `tests/gpu/barrier_smoke.py`. The difference
is the model, and this failure shape was cross-checked before: the
exploration's nway3 measurement ran planted-key pairs with
~4,000-token anchored documents through a trivially-correct causal
reference implementation (`tests/gpu/milestone1.py`,
`run_debug_join`) and got the same near-all-TRUE answers with zero
disagreements against the packed executor (recorded in
`tests/gpu/session_smoke.py`'s docstring). So this is not the
engine; the model itself stops discriminating.

The variable is not distance from the fact to the question. The
color line is the LAST line of every document here, so in the
failing orientation it sits only ~65 tokens before the answer
position - while the working orientation recalls a color from the
top of the context, across all 3,000 intervening tokens, exactly.
What separates every failure from every success in these cells is
the length of the anchored document the judgment must read: ~450
tokens (session smoke) and ~800 tokens (gate stages 1 and 2) answer
exactly; ~3,000 tokens answers TRUE for every pair. The failure
direction is always toward TRUE (0.825 at gate stage 3, 1.0 here):
when the model stops discriminating, it says yes.

Caveats and consequence:

- This corpus is an extreme shape: 130 copies of one filler
  sentence and a single fact line. Real prose behaves differently -
  the REACTION shape measures sensible selectivities on
  ~3,500-token anchored reports - so where the collapse threshold
  sits on real text, and whether it moves at all, is unknown. Issue
  #43's orientation check is the measurement that settles it, with
  real predicates run both ways.
- Until that measurement exists, anchor choice is not purely a cost
  decision at current model quality. The escape hatch is explicit:
  `anchor=` forces the orientation, and the planner honors it (with
  a cost remark). The cost model may eventually need an accuracy
  term; that is #43's question to answer with data, not something to
  guess into the planner now.

## Summary

- The gate is mechanically exact, and its formula is an accurate
  expectation - accurate enough for the cost decisions it feeds.
- The barrier pays off as modeled: 11.3x tokens and wall on this
  shape, with plan-predicted token counts matching measurement to
  0.03%.
- Orientation moves accuracy - on this adversarially planted corpus,
  catastrophically. The cost-optimal plan was the accuracy-broken
  one. #43 measures this properly on real text.
