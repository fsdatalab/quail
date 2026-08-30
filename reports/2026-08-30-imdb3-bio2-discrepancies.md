# Why IMDB-3 loses and BIO-2 wins: GPU timeline, KV regret, churn

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

This experiment measured both systems on both queries, on one H100
with Qwen3 4B fp8, with two cells that wrap instrumentation around
unmodified engine and baseline code:

- `ablations/discrepancy_timeline.py` (since generalized into
  `ablations/profile_quail.py`, which runs any QuailB query) reruns
  the two queries through
  Quail's real planner and worker core. Per forward pass it records
  launch time, packed tokens, and GPU time from the CUDA events the
  loops already create; per phase, the wall and CPU seconds per loop
  step; every eviction of retained KV; and KV regret. torch.profiler
  windows (kernel activity only) cover a few forward passes per
  regime. A second, unprofiled pass supplies every cited number;
  profiled and unprofiled walls agree within 3.5%.
- `ablations/discrepancy_stock.py` (since generalized into
  `ablations/profile_stock.py`) measures stock vLLM with the
  benchmark baseline's own prompt, boot, sampling, submission, and
  cache-reset code. It records per-request `num_cached_tokens`
  bucketed by position within the anchor group, per-request KV
  regret, and torch.profiler windows through vLLM's profiler config.
  vLLM's profiler also traces CPU work, which stretched its windows
  1.1x to 2.4x, so every stock busy number below is corrected to the
  unprofiled run: kernel time per request in the window divided by
  the unprofiled wall per request. BIO-2 ran its first 60 of 500
  reports (67,620 pairs); the recorded run supplies the full wall.

KV regret: the fresh tokens spent recomputing KV that the same query
already computed once. For Quail the unit is the document prefix
under its (alias, document) key. For stock vLLM it is the longest
common token prefix between the request and any earlier request,
rounded down to vLLM's 16-token cache block. With an unlimited KV
space every regret token would have been a hit. First computations
and tuple-suffix tokens are not regret; no cache of any size avoids
them.

Data on the `quail-results` volume:

- `/results/ablations/discrepancy_imdb3.json`, `discrepancy_bio2.json`
- `/results/ablations/discrepancy_stock_imdb3.json`,
  `discrepancy_stock_bio2.json`
- `/results/ablations/discrepancy_traces/` (five Quail chrome traces;
  `stock_kineto/` holds the three vLLM traces)
- Modal function calls: `fc-01M1847YH8593GPRPK3AJQM7BP` (Quail, sf
  0.1), `fc-01M18ANPY0AAXTRHCHZJHF5HNB` (stock, sf 0.1), and the sf
  0.01 checks `fc-01M1840EENN9801ZX13H00MQD6` and
  `fc-01M189AMV696EZ6RYFX0K7QCT8`.
- Recorded comparison numbers come from the 2026-08-27 report's
  files.

## Prediction

Stated in the cells' docstrings before the runs:

- Quail IMDB-3: the filter phase is the slow part (45 to 55 of the
  75 seconds); chunks collapse once the arena fills about 1,170
  documents in; evictions run continuously; regret about 1.22M
  tokens; the join phase alone is healthy.
- Quail BIO-2: regret exactly 0, chunks near the budget, GPU busy
  through most of the wall.
- Stock IMDB-3: the filter runs full batches with near-zero cached
  tokens; at the join, pair 0 of every review finds its prefix gone
  (the 1.76M-token scan flushed the 479,616-token cache), pairs 1-11
  hit; regret about 1.35M tokens, slightly larger than Quail's.
- Stock BIO-2: regret about 0; GPU busy well below Quail's, because
  at most about 117 sequences of 4,066 tokens fit the KV space, so
  steps stay small.

## Result

Every prediction held. One correction to the initial mental picture,
in both directions: each system's slow regime hides its idle time at
a different grain. Quail's chunk-level CUDA events read 95.8% busy
while the kernel-grain trace shows 35.7%; stock's profiled BIO-2
window reads 8.4% raw while the correction to the unprofiled rate
gives 18%.

### The two-query, two-system summary

| | Quail IMDB-3 | Stock IMDB-3 | Quail BIO-2 | Stock BIO-2 |
|---|---:|---:|---:|---:|
| Recorded wall (s) | 75.0 | 52.65 | 128.46 | 1,532.25 |
| KV regret (tokens) | 1,220,547 | 1,326,432 | 0 | 0 |
| Churn observed | 2,625 evictions, 4,260 of 4,380 kept documents lost | all 1,326,432 would-hit tokens recycled before the join | none | none |
| Slow-regime GPU busy (kernel grain) | 36% (filter churn) | 93-95% (composes fine) | 99% | 18% |

Figure: plots/discrepancy_regret.png

Regret does not separate the systems anywhere. On IMDB-3 both waste
about the same recomputation; on BIO-2 neither wastes any. What
separates them is what the misses cost: stock recomputes its misses
inside full-size batches, Quail's attempt to avoid them collapsed
its filter batching; and on BIO-2, what separates them is stock's
per-request machinery leaving the GPU idle.

### IMDB-3, Quail: retention churn

The rerun took 68.93 seconds inside the worker against 75.0
recorded; the 6-second difference is container variance and does not
change the shape.

- The filter phase took 50.28 seconds, compared with 15.06 seconds
  for the identical filter work in IMDB-1. The join phase took 18.53
  seconds, matching the 19.24 predicted from IMDB-2. The composition
  penalty is entirely the filter phase.
- The plan-time keep credit priced 358,861 of the 362,250-token
  arena as resident survivor KV. Its headroom term reserves the
  working set of one document (3,389 tokens), but the loop keeps up
  to two full chunk budgets of in-flight document KV (about 220,000
  tokens) in the same page pool.
- So after 6 forward passes of 65,088 mean tokens, the arena filled,
  and every later admission first evicted one retained document (the
  head-of-queue shortfall, about 20 pages). The filter degenerated
  into 2,916 forward passes of 469 mean tokens - 1.3 documents each
  - through 2,625 eviction calls. Fresh-token rate: 29k per second,
  compared with 117k before the arena filled.
- In the churn profiler window, kernels cover 35.7% of the time:
  24,064 kernels with a 9.4-microsecond mean, idle slices between
  them. The filter's real kernel work is the same ~15 seconds
  IMDB-1 pays, stretched to 50.28.
- The churn is worthless in expectation once the arena is full:
  each cycle either swaps one kept survivor for another (87.6% of
  admissions pass) or destroys one for a document that fails
  (12.4%). Retained value never grows; each cycle pays a forward
  pass's overhead.
- The join found 120 of 4,380 anchors resident (the longest reviews;
  eviction removes the least saved recompute per page first). Regret
  measured 1,220,547 tokens, within 0.01% of the offline prediction.
  The 137,397 hit tokens saved about 1.3 seconds. Retention cost
  35.2 seconds to save 1.3.

Figure: plots/discrepancy_imdb3_timeline.png

The top panel uses a log scale because chunk sizes span more than
two orders of magnitude. The x axis starts when the instrumented run
starts; planning and payload assembly occupy the first few seconds.

Figure: plots/discrepancy_imdb3_composition.png

### IMDB-3, stock vLLM: same regret, paid in full batches

The stock cell reproduced the recorded run exactly where it matters:
4,374 survivors, and join cached tokens of 16,622,144 - equal to the
recorded benchmark counter to the token. Its container ran 20-25%
faster (filter 17.15 seconds against 21.15 recorded, join 22.98
against 31.50); the shape is unchanged.

- Filter: full batches, 94.9% corrected kernel busy, 288 cached
  tokens out of 1.76M - a single-stage scan has nothing to reuse.
- Join: pair 0 of every one of the 4,374 surviving reviews found 0
  of its 1,326,432 would-hit prefix tokens cached. The filter scan's
  1.76M tokens had flushed the 479,616-token cache, and the join
  walks reviews in scan order - the head of the scan, which is
  exactly what LRU recycled first. Cross-operator reuse: zero.
  Pairs 1-11 cached 16,622,144 tokens (the within-join reuse works:
  the 12 aspects of one review arrive consecutively). Regret:
  1,326,432 tokens.
- The join still composes additively because vLLM's cache never owns
  memory: finished requests' blocks sit in the free list, instantly
  claimable, so cache loss shows up as recompute inside full-size
  steps (92.6% corrected busy), never as admission starvation.

### BIO-2, Quail: fully packed

130.35 seconds rerun against 128.46 recorded. One join phase: 98
forward passes at 105,861 mean tokens (96% of budget), 99.4% busy at
chunk grain, 99.9% and 99.2% in the kernel-grain windows. Regret 0;
each of the 500 report prefixes (2,033,852 tokens) computed exactly
once. GEMM kernels account for 83% of window kernel time, so the
remaining 2.07x over the SoL time is kernel efficiency, not packing
or scheduling.

### BIO-2, stock vLLM: the GPU runs kernels 18% of the time

The partial run (60 reports, 67,620 pairs) took 117.7 seconds - 1.74
milliseconds per pair on this container, against 2.72 recorded for
the full 563,500 pairs. Cache behavior matches the recorded run:
99.55% of prompt tokens served from cache, regret 0 (pair 0 of each
report is a first computation, pairs 1-1,126 hit nearly their full
would-hit prefix).

The profiler window is the finding: 0.31 milliseconds of kernel time
per pair against the 1.74-millisecond unprofiled wall - the GPU runs
kernels about 18% of the time (11% against the recorded full-run
rate). Even stock's kernel time per pair (0.31 ms) exceeds Quail's
whole wall per pair (0.23 ms).

Figure: plots/discrepancy_bio2_strips.png

The strips show two-second excerpts of the profiled windows: solid
color where the GPU is executing a kernel, hatched fill where it
idles. Quail's strip is one solid band; stock's is slivers between
idle gaps. Stock's excerpt comes from its traced run, and tracing
overhead stretches the CPU gaps about 2.4x while kernels run at
normal speed. So the hatched share of stock's strip overstates the
idle: raw, the window is 8.4% busy; against the unprofiled wall
for the same 6,762 requests it is 18%. The strip's label states
the corrected 18%, and the note says why the hatching shows more
idle than that.

The trace's Python stacks say what fills the idle. Over the
28.3-second window (6,762 requests, 2.1 seconds of merged kernel
time), on-stack seconds split exactly across the engine thread's
busy loop - execute_model 14.8 (2.1 of it kernels; the rest is
per-layer launch and quantization Python, and waiting), the
scheduler 7.8 (5.7 of it probing the prefix cache for each
request's longest cached prefix), output processing 2.4, input-queue
handling 3.3 - while the input thread, sharing the interpreter lock
beside it, spent 11.6 seconds building requests, 10.7 of them
hashing every request's blocks for the prefix cache (1.8 million
`hash_block_tokens` calls, about 268 blocks per 4,100-token
request). The prefix cache's own bookkeeping - hashing plus probe,
16.4 on-stack seconds - costs about eight times the kernel time
that runs in the window. So the idle is not one mystery gap: it is
named per-request Python work, dominated by cache bookkeeping and
small-step launch overhead. The steps are small because at most
about 117 sequences of 4,066 tokens fit the 479,616-token KV space
at once, so a step carries about 2,000 tokens where Quail's forward
pass carries 106,000.

Figure: plots/discrepancy_bio2_cpu.png

Figure: plots/discrepancy_window_busy.png

Figure: plots/discrepancy_bio2_rates.png

The rate figure uses a log scale because the rates span more than
one order of magnitude.

## Meaning

- IMDB-3 is not intrinsically adversarial for Quail, and the misses
  are not the problem - both systems have the same ~1.2-1.3M-token
  regret there. Quail loses because retention misfires: the keep
  credit models one document of admission headroom where the loop
  needs about two chunk budgets, and eviction frees one blocked
  document at a time. The fix - reserve the loop's two-chunk
  working set up front and cap retained KV at what is left (about
  arena minus two chunk budgets, exactly where the runtime's
  retained mass converged on its own) - shipped as the scan ring;
  measured result in `2026-08-30-kv-ring-fix.md`.
- BIO-2's win is per-request machinery, now measured on the GPU
  timeline and named on the CPU stacks: stock leaves the GPU idle
  about 82% of the time at a 99.55% cache hit rate - most of it
  prefix-cache hashing and probing plus small-step launch Python -
  while Quail runs 99% busy on the same
  fresh tokens. The paper sentence: caching is not the bottleneck
  on either side; batch shape is.
- vLLM's design cannot starve its own admission (cached blocks live
  in the free list), and it evicts at step grain; the cost is that
  it cannot hold KV for a later operator, so its cross-operator
  reuse on IMDB-3 was zero. Quail's arena can hold KV across
  operators - the sf 0.01 checks show both systems reusing
  everything when the corpus fits - but until the watermark fix, the
  same mechanism is what collapsed the IMDB-3 filter.
- Grain matters when reporting GPU utilization: chunk-level CUDA
  events include intra-chunk idle (95.8% against 35.7% at kernel
  grain in Quail's churn), and vLLM's CPU-tracing profiler inflates
  idle (8.4% raw against 18% corrected). Every number above says
  which grain and which correction it uses.

## Rebuild

Rerun the cells:

    uv run modal run ablations/profile_quail.py::run --queries IMDB-3,BIO-2 --out-prefix rerun
    uv run modal run ablations/profile_stock.py::run --queries IMDB-3,BIO-2 --max-join-anchors 60 --out-prefix rerun

(The recorded files came from these cells' predecessors,
`discrepancy_timeline.py` and `discrepancy_stock.py`, at the
pre-scan-ring engine; the generalized cells reproduce the same
measurements, though profiler window names now derive from the run
rather than from per-query tables, and a fresh --out-prefix keeps
the recorded files intact.)

Rebuild the figures and the derived numbers with
`reports/make_imdb3_bio2_discrepancies_plots.py`; its docstring
holds every `modal volume get` command.
