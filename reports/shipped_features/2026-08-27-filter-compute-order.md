# Filter ordering uses dense and attention limits

## What changed

The `by_cost` filter order prices dense and attention separately. Each
part takes the larger of compute time and memory time. The two parts are
then added. The planner uses the mean document length, model dimensions,
device rates, and chunk capacity.

The QUAIL-B SoL calculation calls the same ordering function. It uses the
measured mean prefix length for each alias and runs the ordering separately
for Qwen3 4B fp8 and Qwen3 32B fp8.

## Why

The previous rule divided question tokens by the rejection fraction. The
rule included prompt length but treated every question token as having the
same cost. A longer question also has more causal attention within the
question, so token count alone can choose the wrong order when two scores
are close.

The first filter uses `scan`, while later filters use `ask`. The planner sorts
the asks once by cost per rejected document. Prefix survivor products and
expected costs let it price each filter as the first scan in constant time.
The search takes `O(n log n)` work for `n` filters.

## Verification

A controlled planner test has a 100 token question with selectivity 0.1 and
a 10 token question with selectivity 0.9101 over a 400 token prefix. The old
token score puts the 100 token question first. The dense and attention
limits put the 10 token question first. An end to end planner test verifies
the emitted `FilterChain` order. Another test covers a mixed case where
dense is compute bound and attention is memory bound.

In the full sf0.1 SoL result, FEV-1 is the only mixed query. Its 4B estimate
increases from 0.024524 to about 0.02468 seconds, or 0.65%. The current result is
at `/results/sol/sol_quailb_sf0.1.json` on the `quail-results` volume. No GPU
run was needed because this is a change to the analytical SoL estimate.
