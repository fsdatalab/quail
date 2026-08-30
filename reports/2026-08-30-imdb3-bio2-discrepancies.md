# Why IMDB-3 loses and BIO-2 wins: GPU timeline and KV regret

## Setup

The 2026-08-27 QuailB SF 0.1 benchmark left two results to explain.

- BIO-2 (the reports x terms join) is Quail's largest win: 128.46
  seconds, compared with 1,532.25 seconds for stock vLLM and
  1,427.45 seconds for pipelined vLLM.
- IMDB-3 (the F1 filter feeding the reviews x aspects join) is the
  only query where Quail loses: 75.0 seconds, compared with 52.65
  seconds for stock vLLM and 48.21 seconds for pipelined vLLM.
  Quail's own parts predict about 34.3 seconds: IMDB-1 runs the same
  filter alone in 15.06 seconds, and IMDB-2 runs the same join over
  all 5,000 reviews in 21.96 seconds (19.24 seconds when scaled to
  the 52,560 surviving pairs). Both vLLM configurations compose
  additively; Quail does not.

This experiment reran exactly those two queries once each on one
H100 with Qwen3 4B fp8, through the real planner and the real worker
execution core, with instrumentation wrapped around the engine from
`ablations/discrepancy_timeline.py`. No engine code changed. The
instrumentation records, per forward pass, the launch time, packed
tokens, and GPU time from the CUDA events the loops already create;
per phase, the wall time and the CPU seconds per loop step; every
eviction of retained KV; and KV regret. Each query also ran a second
time with short torch.profiler windows (kernel activity only) so the
recorded slowdowns can be inspected at kernel grain. The unprofiled
pass supplies every number below; profiled and unprofiled walls
agree within 3.5%.

KV regret: the fresh tokens spent recomputing a document prefix
whose KV this query already computed once under the same
(alias, document) key. With an unlimited KV arena every one of those
tokens would have been a hit. First computations and tuple-suffix
tokens are not regret, because no cache of any size avoids them.

Data on the `quail-results` volume:

- `/results/ablations/discrepancy_imdb3.json`
- `/results/ablations/discrepancy_bio2.json`
- `/results/ablations/discrepancy_traces/` (five chrome traces)
- Modal function calls: `fc-01M1847YH8593GPRPK3AJQM7BP` (measured,
  sf 0.1) and `fc-01M1840EENN9801ZX13H00MQD6` (sf 0.01 check).
- Recorded comparison numbers come from the 2026-08-27 report's
  files (same volume paths as listed there).

## Prediction

Stated in the cell's docstring before the run:

- IMDB-3: the filter phase is the slow part, about 45 to 55 of the
  75 seconds. Its chunks collapse from the 110,376-token budget to a
  few thousand tokens once the arena fills, about 1,170 documents
  in, and evictions run continuously from that point. Regret is
  about 1.22M tokens (the 4,260 anchors the recorded run evicted).
  The join phase alone is healthy, about 20 seconds.
- BIO-2: regret is exactly 0, chunks stay near the budget, and GPU
  busy time covers most of the join wall.

## Result

Every prediction held, with one correction to the mental picture:
the idle time in the IMDB-3 filter hides inside the chunk spans, not
between them.

### IMDB-3

The rerun took 68.93 seconds inside the worker, compared with 75.0
seconds in the recorded family run; the phase split below is from
the rerun. The 6-second difference is run-to-run container variance,
and it does not change the shape of the result.

- The filter phase took 50.28 seconds, compared with 15.06 seconds
  for the identical filter work in IMDB-1. The join phase took 18.53
  seconds, matching the 19.24 seconds predicted from IMDB-2. The
  whole composition penalty is the filter phase.
- The filter ran 2,922 forward passes: 6 before the first eviction,
  mean 65,088 tokens each, then 2,916 after it, mean 469 tokens each
  — about 1.3 documents per pass, against a 110,376-token budget.
- The eviction path ran 2,625 times and freed 4,260 retained
  documents, one blocked admission at a time. The filter's fresh
  token rate fell from 117k per second before the arena filled to
  29k per second after.
- The plan-time keep credit caused the fill: it priced 358,861 of
  the 362,250-token arena as resident survivor KV. The credit's
  headroom term reserves the working set of one document (3,389
  tokens, the longest document plus its question), but the loop
  keeps up to two full chunk budgets in flight. The runtime
  converged to 138,096 retained tokens, which is the arena minus
  about two chunk budgets.
- The chunk-level CUDA events say the GPU was 95.8% busy during the
  filter. The kernel-grain profiler window inside the churn shows
  the truth: kernels cover only 35.7% of the window, 24,064 kernels
  with a mean of 9.4 microseconds, so almost two thirds of the
  window is idle slices between tiny kernels. A chunk's event span
  includes that idle time. The filter's actual kernel work is about
  15 to 17 seconds — the same work IMDB-1 finishes in 15.06 seconds
  — stretched to 50.28 seconds.
- KV regret measured 1,220,547 tokens, within 0.01% of the value
  computed beforehand from the recorded token totals. At the join's
  109k tokens per second that is about 11 seconds of recompute. Only
  120 of 4,380 join anchors were still resident (the longest
  reviews, because eviction removes the least saved recompute per
  page first); their 137,397 hit tokens saved about 1.3 seconds.
- The balance for retention on this query: it cost 35.2 seconds in
  the filter to save 1.3 seconds in the join.

Figure: plots/discrepancy_imdb3_timeline.png

The top panel uses a log scale because chunk sizes span more than
two orders of magnitude. The x axis starts when the instrumented run
starts; planning and payload assembly occupy the first few seconds,
before the first forward pass.

Figure: plots/discrepancy_imdb3_composition.png

Figure: plots/discrepancy_window_busy.png

### BIO-2

- The rerun took 130.35 seconds, compared with 128.46 recorded. One
  join phase: 98 forward passes, mean 105,861 tokens — 96% of the
  chunk budget. GPU busy was 99.4% at chunk grain and 99.9% and
  99.2% in the two kernel-grain windows. Regret was 0; each of the
  500 report prefixes (2,033,852 tokens) was computed exactly once.
- Stock vLLM's loss is not a cache problem. Its prefix cache served
  99.55% of 2.32 billion prompt tokens (2,313,621,216 cached).
  What remains is per-pair machinery: 563,500 separate requests,
  each a prefill step plus a decode step over a mean 4,066-token
  cached report, with at most about 117 sequences resident at once
  (479,248 KV tokens divided by the report length) and a
  25,305-token step budget. That is 2.72 milliseconds of wall per
  pair, compared with 0.23 milliseconds for Quail.
- All three systems process the same 10.4 to 10.5M fresh tokens.
  The rates: SoL 167.3k, Quail 80.8k, pipelined vLLM 7.4k, stock
  vLLM 6.9k tokens per second. Quail is 2.07 times above the SoL
  time; stock vLLM is 24.7 times above it.
- Quail's own 2.07x residual sits inside kernels, not in packing or
  gaps: in the profiled windows, GEMM kernels account for 83% of
  kernel time. Closing that residual is a kernel efficiency
  question, not a scheduling one.

Figure: plots/discrepancy_bio2_rates.png

The rate figure uses a log scale because the rates span more than
one order of magnitude.

## Meaning

- IMDB-3 is not intrinsically adversarial for Quail. Composed
  without KV retention, the parts run in about 34 seconds, faster
  than both vLLM configurations. The loss is a planner decision plus
  an eviction mechanism: the keep credit reserves headroom for one
  document where the loop needs about two chunk budgets, and under
  pressure the arena frees one blocked admission's pages at a time,
  which locks the filter into 469-token forward passes. Candidate
  fixes, in order of expected value: cap the keep credit at the
  arena minus the admission working set; evict in bulk for the whole
  pending admission window; have the planner price retention's churn
  cost against its expected hit value (here: 1.3 seconds of value
  against 35.2 seconds of cost, with the counted constants able to
  see both once the working set is modeled).
- The regret metric works and localizes correctly: 1.22M tokens on
  IMDB-3 (32% of its fresh tokens, about 11 seconds), exactly 0 on
  BIO-2 for every system. It explains a third of the IMDB-3 gap and
  none of the BIO-2 win, so it must be read next to the timeline,
  not alone.
- BIO-2's 9.9x family win over stock vLLM (11.9x on this query) is
  the per-request cost of one-pair-per-request serving, not cache
  behavior. The right comparison sentence for the paper: at a 99.55%
  prompt cache rate, stock vLLM still runs 24.7 times above the SoL
  time, because every pair pays request scheduling, one prefill
  step, and one decode step over a 4,066-token context.
- Chunk-level CUDA events overcount GPU busy time: they include idle
  slices between kernels inside a chunk (95.8% at chunk grain
  against 35.7% at kernel grain in the churn regime). Future
  utilization numbers should say which grain they measure.

## Rebuild

Rerun the cell:

    uv run modal run ablations/discrepancy_timeline.py::run_queries

Rebuild the figures and the derived numbers with
`reports/make_imdb3_bio2_discrepancies_plots.py`; its docstring
holds every `modal volume get` command.
