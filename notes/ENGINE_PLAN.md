# Measured layer plan: real H100 runs on Modal

> Historical harness plan. New runs use `experiments/modal_rebuild.py` and are
> indexed in [EXPERIMENTS.md](EXPERIMENTS.md).

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

## Run isolation and request order

The prefix cache persists across runs on the shared engine, so without a
reset a run inherits document KV computed by the run before it. The first
grid had no resets, and its per wave counts show what that did. Task and
k=1 runs were effectively cold, because each task run touches about one
million tokens of KV that nothing else shares (the task prompt sits before
the document, so task KV can never match a document first prompt) and that
volume flushes the cache before the next run. Every run with k of 2 or more
followed a k=1 run of the same configuration and started with 96 to 97
percent of its prompt tokens already cached, so those runs never paid for
document prefill and their makespans measured a different regime than their
competitors. The fixed protocol resets the prefix cache before every run.

Two request order rules make a cold speculative run pay for each document
once rather than k times. Within a wave the branch requests are issued
branch major, meaning all first branches, then all second branches, so a
later branch reaches the scheduler after an earlier branch of the same
document has finished computing the shared document prefix. Within a
manifest batch the bare document prefill requests go first and the branch
requests follow in stage order.

A small warm arm deliberately skips the reset for two configurations. It
measures the regime where the corpus KV is already resident from an
earlier query over the same documents, which is the recurring query case a
document store cares about. Only the document first templates can use
resident KV. Task first cannot, whatever the cache holds, because its
template puts the task prompt before the document.

## Grid

- 2 filters: s1 in {0.25, 0.5, 0.8} with s2 = 0.5, policies task, k=1, k=2.
- 3 filters: s per stage in {0.7, 0.9}, policies task, k=1, k=3.
- 4 filters: s per stage in {0.8, 0.95}, policies task, k=1, k=2, k=4.
- Warm arm: k=1 and k=n again at n=2 with s1=0.5 and at n=4 with s=0.8,
  without the cache reset.
- Manifest driven: task, k=1, and k=n at n=2 with s1=0.5 and at n=4 with
  s=0.8, where the engine is driven batch for batch from the analytical
  builder's schedule.

That is 33 runs and roughly 45 to 60 GPU minutes.

## Comparison

The container returns realized token lengths, per-stage prompt lengths, the
answers (the outcome matrix the schedule actually followed), per-wave
timings, and cached-token counts. experiments/analyze_engine.py rebuilds
the exact instance, runs the analytical builders for the same policy, and
reports measured, ideal, the ratio, the achieved prefill tokens per second
against the 275,000 per second FP8 ceiling, and the cache hit fraction.
