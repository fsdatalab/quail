# Measured layer plan: real H100 runs on Modal

The goal is to measure real wall-clock makespans for the scheduling
policies on a rented H100 with Qwen3-4B-FP8 under vLLM, with selectivity
under our control, and to compare each run against the ideal model's
prediction for the exact same realized workload. The headline number per
run is measured seconds over ideal seconds.

## Controlled selectivity

Each of 2,000 documents from the seeded IMDb sample gets a trailing
metadata line, for example "[FLAGS] FLAG_1=YES FLAG_2=NO", where flag j is
YES with probability s_j from a recorded seed. Filter j asks the model to
read flag j and complete "The answer is" with YES or NO at temperature 0.
Outcomes are therefore planted, and the run records the model's agreement
with the planted flags, which should be near 1. Realized pass rates match
the planted ones up to sampling noise.

## Policies as wave structures on one engine

- Task-first sends [task prompt][document][answer cue] per stage, so every
  stage re-prefills the surviving documents, and only the short task prefix
  is shared between requests.
- Pipeline sends [document][question], and stage j+1 is issued only after
  stage j's outcomes are read. Document KV reuse across stages comes from
  vLLM's automatic prefix caching, and the per-request cached-token counts
  prove whether it happened.
- Lookahead k issues all k branch questions per document at once, without
  waiting, and pays for the wasted branches of failed documents.

vLLM does continuous batching inside each wave; the scheduling under test
is the wave structure, the gating, and the templates. Engine settings: fp8
KV cache so the whole corpus stays resident, prefix caching on, one
generated token per request, weights cached in a Modal volume, one warm
container for the whole grid.

## Grid

- 2 filters: s1 in {0.25, 0.5, 0.8} with s2 = 0.5, policies task, k=1, k=2.
- 3 filters: s per stage in {0.7, 0.9}, policies task, k=1, k=3.
- 4 filters: s per stage in {0.8, 0.95}, policies task, k=1, k=2, k=4.

About 23 runs and 30 to 45 GPU minutes.

## Comparison

The container returns realized token lengths, per-stage prompt lengths, the
answers (the outcome matrix the schedule actually followed), per-wave
timings, and cached-token counts. experiments/analyze_engine.py rebuilds
the exact instance, runs the analytical builders for the same policy, and
reports measured, ideal, the ratio, the achieved prefill tokens per second
against the 275,000 per second FP8 ceiling, and the cache hit fraction.
