# Results from the solved schedules

## What was solved and how to read the numbers

The paper in this repo (paper.md) studies a database query that runs a chain
of yes or no language model filters over documents on one graphics card. A
document that fails a filter skips the rest, so the work depends on how many
documents pass each stage. The paper compares three scheduling policies.
Task-first reprocesses every surviving document at each filter stage under a
shared prompt. Pipeline reads each document once, keeps the model's stored
memory of it (the KV cache) on the card, and runs each filter as a prompt of
about 50 tokens on top, waiting for each answer. Speculation runs future
filter prompts without waiting, wasting them when an early filter fails.

The revised paper organizes every claim into four layers, and the repo now
computes all four: a resource lower bound that no schedule can beat, an
asymptotic latency target from a linear program over steady batch rates, a
constructed finite schedule replayed from the program's rates and checked
by an independent validator, and a measured number from real H100 runs,
reported in its own section near the end. Except in that measured section,
every number below comes from the paper's ideal cost model with its
corrected attention width and its write-through storage rule, on 10,000
real IMDb reviews (2,966,000 document tokens) tokenized for the Qwen3
models. The pass rate of a filter is the fraction of documents that pass
it.

Raw data: results/lp_two_stage.csv (the program and its replays),
results/n10k_two_stage.csv (direct schedule builders), manifests in
results/manifests/, the small instance study in experiments/run_smallN.py,
and the measured engine runs in results/engine/.

## The three computed layers agree, which is the main check

For every combination of two models (Qwen3-4B, Qwen3-32B), two cards (H100,
L40S), pass rates 0.1, 0.5, 0.9, and the three policies, all of the
following hold:

- The lower bound never exceeds the replayed finite latency, in all 36
  cells.
- The replayed finite latency lands within 0.5 percent of the program's
  asymptotic target. The signed difference runs from minus 0.07 to plus 2.0
  seconds and is legitimately either sign, because the asymptotic target is
  not a bound for a finite run.
- Where the direct schedule builders had already produced certified optima,
  the program reproduces them to the digit. For example, task-first on the
  4B model and H100 at pass rate 0.5 gives 16.56 seconds by ledger, by
  built schedule, and by program target, and the replay gives 16.62.

Selected finite latencies in seconds (replayed, validated):

| model and card | policy | pass rate 0.10 | 0.50 | 0.90 |
|---|---|---|---|---|
| 4B on H100 | task-first | **12.14** | 16.62 | 21.04 |
| | pipeline | 13.05 | **13.80** | **14.54** |
| | full speculation | 14.74 | 14.74 | 14.74 |
| 32B on L40S | task-first | **281.9** | 384.3 | 486.1 |
| | pipeline | 301.7 | **319.1** | **336.2** |
| | full speculation | 340.9 | 340.9 | 340.9 |

The conclusions from the earlier direct builders stand. Task-first wins
below a pass rate of about 0.20 (the crossing point is 500,000 over
2,466,000, or 0.203). Pipeline wins above it. Full speculation never wins
at this scale under the ideal cost model, because there is always other
work to hide filter waits behind, and even at 3.3 percent KV residency
(32B on L40S) the pipeline never recomputes anything.

## What the program layer adds

The linear program chooses long-run rates for state-conditioned batch
actions, subject to queue balance, cache-state balance, and one GPU second
per second, and its optimum is a certified throughput ceiling for its
supplied state and action set. Details of this implementation, all
recorded in the result files:

- Task-first and full speculation keep no per-document KV between batches,
  so their cache state collapses to a single state and the program runs
  over batch templates with exact empirical length types (1,037 distinct
  lengths, no averaging).
- The pipeline uses a one-dimensional quantized count of resident
  survivors. Under the paper's assumption that pass rates do not depend on
  document length, the length mix of survivors equals the admission mix
  exactly, so per-survivor costs use the exact mix and no per-length cache
  state is needed. Survivor counts above the capacity share overflow to a
  recompute queue, which is the model's eviction path.
- Costs of fluid actions are expectations over the mix, which the paper
  permits when declared. Real feasibility is enforced during replay, and
  the independent validator replays every batch of every reported schedule.

The replay converts rates into actual batches with real documents and
tags every batch as fill, core, repair, or drain. Repair batches were never
needed in these runs. One structural observation: the task-first replay
executes many batches far below their template size (its queues hold at
most a few documents of each exact length), which costs nothing under the
ideal model because those batches stay compute-bound, but it would cost
real time under a calibrated model with a fixed charge per batch. The
pipeline replay packs 7 to 100 large batches instead.

## Small documents counts and the exact reference solver

The exact solver (offline Dijkstra plus online value iteration) is now the
validation reference the revised paper assigns it. On real Qwen3-4B and
H100 numbers it solves up to 4 documents in seconds, and it found the one
schedule shape the other layers miss: holding a document back so that the
survivors' next filter prompts run inside the held document's prefill
batch, which makes filter waits free. Speculation wins only when nothing is
left to hold back (2 documents, or the tail of a query). The direct
builders run 12 to 24 percent behind the exact optimum below 8 documents,
and the program layer inherits a milder version of the same limitation
through its restricted action menu.

## The measured layer: real H100 runs against the ideal model

The fourth layer runs the policies on a rented H100 through Modal, with
vLLM 0.26 serving Qwen3-4B-FP8, an fp8 KV cache, prefix caching on, and
2,000 documents from the seeded sample. Selectivity is planted. Each
document ends with a metadata line such as "[FLAGS] FLAG_1=YES FLAG_2=NO"
drawn from a recorded seed, and filter j asks the model to output the
value of flag j as one token at temperature zero. The engine records
realized token lengths, per wave timings, and per request cached token
counts. experiments/analyze_engine.py rebuilds the exact instance from
those lengths and runs the analytical builders on the same realized
outcome matrix, so measured and ideal describe the same workload. The
model's answers match the planted flags on 96 to 99.5 percent of calls for
the task template and 82 to 97 percent for the shorter document-first
question, and the schedules follow the model's answers, so each comparison
is internally consistent. Raw data: results/engine/grid.json and
grid_analysis.csv, with the plot in results/plots/engine_measured.png.

One protocol lesson cost a full grid. The engine's prefix cache persists
across runs, and in the first grid every speculation run followed a k=1
run over the same documents and started 96 to 97 percent cached, so it
never paid document prefill and appeared to win everywhere by a factor of
two. The per wave cached token counts exposed this. That grid is kept as
results/engine/grid_run1.json and reads as a warm regime measurement for
its k of 2 or more rows only. The fixed protocol resets the prefix cache
before every run and issues speculative branches in branch major order,
all first branches then all second branches, so a cold speculative wave
prefills each document once rather than k times. The cold first wave cache
hits confirm both fixes: 47 percent at k=2, 63 at k=3, and 71 at k=4,
each exactly the in flight sharing ceiling where every branch after a
document's first reuses its KV, and zero for k=1.

Cold measured makespans in seconds, 2,000 documents, best per row in bold:

| configuration | task-first | pipeline k=1 | lookahead 2 | full speculation |
|---|---|---|---|---|
| n=2, s=0.25 | 11.2 | **9.4** | | 11.0 |
| n=2, s=0.5 | 13.6 | **10.2** | | 10.7 |
| n=2, s=0.8 | 15.6 | 10.8 | | **10.0** |
| n=3, s=0.7 | 20.0 | **11.8** | | 12.5 |
| n=3, s=0.9 | 24.7 | 13.2 | | **11.5** |
| n=4, s=0.8 | 26.4 | **12.4** | 12.5 | 13.7 |
| n=4, s=0.95 | 33.2 | 14.3 | 13.7 | **12.9** |

What the cold grid shows:

- The document-first template beats task-first in every cell, by 1.19
  times at two filters and the lowest pass rate up to 2.58 times at four
  filters and 0.95, growing with stage count and pass rate exactly as the
  re-prefill arithmetic says it must.
- The gate or speculate choice follows pass rate the way the ideal model
  predicts, with the crossover shifted toward speculation. Gating wins the
  low pass rate cells and speculation the high ones, but the real engine
  flips at a lower pass rate than the ideal model does (at n=2 the model
  flips above 0.8 while the measurement flips at 0.8; at three filters
  and 0.9 and at four filters and 0.95 the measurement inverts the
  model's order within the document-first family),
  because a real gate costs a wave barrier while a wasted branch is cheap
  when most documents survive. At n=4 and 0.8 the measured order k=1, then
  k=2, then k=4 matches the ideal order exactly.
- All 27 cold runs land between 3.31 and 4.10 times their ideal number.
  The reason is a single rate. The engine prefills at 58,000 to 78,000
  prompt tokens per second on these roughly 350 token requests, against
  the 275,000 per second dense FP8 ceiling the ideal model prices. One
  calibration constant of about 3.7 therefore puts the ideal model within
  about 12 percent of every cold measurement, and the residual spread is
  what the wave barriers and per request overheads add on top.

The warm arm reruns four cells without the reset, so the corpus KV is
already resident from the previous run over the same documents. This is
the recurring query regime, where the same document store answers query
after query. Pipeline query time collapses from 10.2 to 3.3 seconds at
n=2 and from 12.4 to 4.7 seconds at n=4, which is 1.17 and 1.48 times the
cold ideal number even though the warm run skips the document prefill the
ideal still prices. In that regime the engine processes only 7,000 to
9,000 new tokens per second, so per request overhead and the one decode
step per filter call set the floor. Warm gating beats warm speculation
decisively at n=4, 4.7 against 8.0 seconds, because with prefill gone
wasted branches are the only remaining cost, which is the ideal model's
logic exactly. Task-first can never use residency, whatever the cache
holds, because its template puts the task prompt before the document and
the prefix match fails at position zero.

The manifest arm drives the engine batch for batch from the analytical
builder's schedule instead of letting the policy loop compose waves, on
six cells. Manifest and wave times agree within 10 percent everywhere:
13.6 against 13.6 and 26.8 against 26.4 for task-first, 10.5 against 10.2
and 13.6 against 12.4 for k=1, 10.8 against 10.7 for full speculation at
n=2, and 12.7 against 13.7 at n=4, where the manifest is faster because
the builder splits the 8,000 request wave into two capacity sized batches.
The analytical schedules are executable as written.

These runs test the policy structure, meaning templates, gating, branch
order, and wave membership, not the paper's exact batch compositions,
because vLLM composes batches inside each wave through continuous
batching and chunked prefill, and the kernels are vLLM's. Each cell is a
single run with no repetition statistics, on one model and one card. The
document-first question template disagrees with the planted flags more
often than the task template, 82 to 97 against 96 to 99.5 percent, so
part of any accuracy gap between policies here is template wording, not
scheduling.

## Scheduler plan steps one and two, measured

Step one, first half. The engine's reading speed limit is about 80,000
tokens per second and it is a kernel fact, not a settings problem. Five
engine configurations (step token budgets from 8,192 to 32,768,
concurrent request limits from 256 to 1,024, prefix caching on and off)
were each measured on reading-only jobs in three forms (pre-tokenized
short documents, four-document concatenations, raw text), and every
one of the fifteen measurements landed between 74,000 and 81,000 tokens
per second, 27 to 29 percent of the theoretical 275,000 ceiling. The
engine's defaults were already at the limit. This settles an open
question in the plan: the cold first read has no recoverable scheduling
or configuration overhead, and 80,000 becomes the calibration constant
for everything below. Raw data: results/engine/speed_limit.json.

A follow-up measurement beneath the serving stack splits the 80,000
into its causes (results/engine/model_floor.json). The model's four
matrix multiply shapes, benchmarked alone in FP8, sustain about
186,000 tokens per second worth of arithmetic (1,350 trillion
operations per second against the 1,979 spec number, so the spec
ceiling itself is one third marketing at these shapes). A bare forward
pass of the bf16 model through plain transformers, with no engine and
no scheduler, reaches 48,000 against its own measured multiply ceiling
of 100,000, so the non-multiply parts of a transformer (attention,
normalization, memory movement, kernel launches) cost a factor of
about 0.48. Applying that factor to the FP8 multiply rate projects a
bare FP8 model floor near 89,000, and vLLM's measured 80,000 is about
90 percent of it. The serving stack's tax on reading is therefore
roughly ten percent, and the distance from the 275,000 paper ceiling
is mostly sustained-versus-spec silicon and the transformer's
non-multiply work, not engine overhead.

Step one, second half. The per request overhead splits half and half
between our client and the engine. On a four filter query over 2,000
documents, run cold and then warm through the async interface in a two
by two design, the warm floor per request was 1.12 milliseconds for
staged execution with raw text (today's harness), 0.74 with
pre-converted token numbers, 1.05 for streaming with raw text, and 0.55
for streaming with token numbers. So pre-tokenization and streaming,
neither of which touches the engine, recover half the floor, and 0.55
milliseconds per request is the genuinely engine-internal share that
only an in-engine scheduler can attack. Cold times moved only 9 percent
across the arms, as expected for a read-bound pass. Raw data:
results/engine/overhead.json.

Step two. At 10,000 documents the corpus needs about 3.4 times the
card's note capacity (3.1 million tokens against a measured 981,728
token pool), and execution order becomes the decisive variable, exactly
as predicted. Cold makespans in seconds, with the winner in bold:

| configuration | task-first | naive k=1 | blocked k=1 | naive k=2 | blocked k=2 |
|---|---|---|---|---|---|
| n=2, s=0.5 | 61.7 | 64.4 | 53.8 | 89.5 | **52.8** |
| n=4, s=0.8 | 122.8 | 111.3 | **74.0** | | |
| n=4, s=0.95 | 158.3 | 153.7 | **84.2** | | |

The engine's cached token counters give the mechanism directly as a
read multiplier, computed tokens over corpus tokens. Naive stage-order
execution reads 1.62 times the corpus at two filters and 3.82 times at
four filters and 0.95, statistically identical to task-first's 1.58 and
3.93, with cache hits at zero: by the time a survivor's next question
arrives, ten thousand documents have passed through the cache and its
notes are gone. The document-first advantage that was worth 25 percent
at 2,000 documents is fully erased at 10,000. Naive speculation is the
worst case, 2.30 times, because a document's second branch trails its
first by ten thousand requests. The blocked schedules from the
analytical builders read 1.18 to 1.32 times the corpus and win every
cell, by 1.15 times at two filters and up to 1.88 times over task-first
at four filters and 0.95.

Run one of this experiment failed in an instructive way and is kept as
results/engine/scale10k_run1.json.gz. The builder's schedules were
right, but the conversion submitted each batch's new document prefills
before its branch requests, and under the engine's keep-the-most-recent
rule the new writes evicted exactly the resident bodies the branches
were about to reuse, erasing the blocked advantage at two filters. The
lesson is that under recency-based eviction, submission order inside a
batch is itself a scheduling decision: requests that consume resident
notes must run before requests that produce new ones. One sort fixed
it, and it is the first concrete design requirement for the step four
in-engine scheduler, which would pin instead of relying on order.

Calibration closes the loop. Multiplying the ideal calculator by the
measured kernel factor of about 3.44 (275,000 over 80,000), task-first
lands within 3 to 4 percent of prediction in all three cells, and the
blocked schedules land at 1.03 to 1.34 times prediction, the residual
being per request overhead this arm still pays (raw text, staged calls)
plus boundary misses. The naive runs sit at 1.35 to 2.55 times
prediction, and they are the one arm the model cannot describe, because
the engine silently broke the model's retention assumption. The step
three client library (blocked order plus streaming plus
pre-tokenization) targets exactly the measured residual. Raw data:
results/engine/scale10k.json.gz and scale10k_analysis.csv.

## Scheduler plan step three: the client library closes the gap

Step three packages the three measured wins into one client scheduler
(docengine/runtime/engine_client.py) that runs against an unmodified
engine: prompts pre-tokenized once, admission controlled by a token
budget sized to the engine's KV pool so a document enters only when its
notes can stay resident, and per document streaming so each document
runs all its filters back to back while hot and returns its budget in
seconds. This produces blocked execution with no stage barriers at all.
On the same 10,000 document cold grid:

| configuration | task-first | naive k=1 | blocked batches | client library |
|---|---|---|---|---|
| n=2, s=0.5 | 61.7 | 64.4 | 53.8 | **48.9** |
| n=4, s=0.8 | 122.8 | 111.3 | 74.0 | **52.5** |
| n=4, s=0.95 | 158.3 | 153.7 | 84.2 | **57.9** |

The client reads 1.17 to 1.30 times the corpus (questions account for
about seven points of that) and beats task-first and naive execution by
2.3 to 2.7 times at four filters. Against the calculator with the one
measured kernel constant applied, it lands at 1.02 times prediction at
two filters and 0.96 at four, slightly below one because the single
constant also scales the attention share of the ideal, which is not
kernel limited in the same way. Within measurement noise, the query now
runs at the engine's own speed limit, and the paper's claim holds on
hardware: plan the query on the formula, execute it, land where the
formula said.

Two smaller findings. Speculation is now neutral or slightly harmful
(50.4 against 48.9 seconds at two filters, 56.7 against 57.9 at four),
confirming that its measured value in the earlier grids was
compensation for stage barriers, which streaming removes; its remaining
role is filling the admission tail, invisible at this scale. And the
whole result rests on client side control only, which bounds what the
step four in-engine scheduler can still add: roughly the residual
between the client and the pure read floor (about 40 seconds of reading
in a 52 to 58 second query), the per request floor of 0.55
milliseconds, and robustness where this arm relies on recency luck.
Raw data: results/engine/client10k.json.gz and client10k_analysis.csv.

## Phase C: the in-engine scheduler, built and demonstrated

The engine skeleton moves the client library's decisions inside vLLM.
A custom scheduler class (docengine/engineext/scheduler.py, loaded
through vLLM's official replacement seam) pins each live document's
notes by holding an extra block reference, frees a dead document's
notes the instant the plan learns of its death, and runs the query's
requests ahead of foreign traffic through the engine's priority queue.
Client directives ride inside request ids because the scheduler lives
in the engine core process. After these changes no query memory is
governed by the engine's keep-the-most-recent rule at all: admission
decides what enters, pins decide what stays, eager frees decide what
leaves, and the recency rule's only remaining jurisdiction is memory
belonging to traffic the plan has never heard of.

Acceptance one, no regression: on the 10,000 document grid the pinned
engine matches or slightly beats the client library (49.2, 52.1, and
54.8 seconds against 48.9, 52.5, and 57.9), with exactly 10,000 pins
taken and released per run and identical answers.

Acceptance two, guarantees against luck, took three iterations, each
of which taught something. A gentle co-tenant (25 requests per second
of 800 token junk) hurt nobody: its in-flight footprint never exceeded
the pool's spare room, the free queue never drained, and recency alone
protected both variants, so the streaming client is naturally robust
to moderate neighbors. A heavy co-tenant (60 per second of 1,500
tokens, about twice the spare room) with the tenant at the same
priority as the query's first reads protected memory perfectly (cache
hits identical to running alone, zero re-reads) but let the tenant
take most of the machine time, stretching the query from 53 to 231
seconds: the card has two scarce resources, and pinning defends only
one of them. With both defended, the demonstration landed. Query alone
52.2 seconds; query under the same heavy neighbor 52.0 seconds, cache
hits unchanged, while the neighbor still received about 4.3 million
tokens of service from the query's slack (2,851 of its requests). The
stock engine under the identical neighbor did not finish within the
one hour harness limit, its queue head blocked by the neighbor's
stuck allocations, against 52 seconds for the plan-managed engine.
Raw data: results/engine/pinned10k.json.gz (acceptance one and the
gentle co-tenant), pinned10k_v3.json.gz (the closing demonstration),
and the run logs for the intermediate iteration.

## Caveats

- Every "X never wins" statement is about the ideal cost model at the
  stated scale. The batch-count differences (for example 100 pipeline
  batches versus 53 full-speculation batches at high pass rates on the
  32B model and L40S) mean a fixed per-batch overhead of roughly 140
  milliseconds would start reversing the speculation conclusion; the
  measured layer above confirms the direction: on the real engine the
  crossover toward speculation arrives at lower pass rates than the ideal
  model predicts.
- The program's value is the optimum of a restricted state and action set,
  labeled per the paper: a fuller action search could raise it, and the
  achievability of the rate is established here only by the successful
  replays, not by the program itself.
- Weight sizes are still estimates from parameter counts, not measured
  from the released files.
- The small instance numbers come from single outcome draws.
