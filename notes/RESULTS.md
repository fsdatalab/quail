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
two. The per wave cached token counts exposed this. That grid was removed
from the repository as invalid (its numbers survive in git history and
this note; the deliberate warm arm of later runs supersedes its
accidental warm rows). The fixed protocol resets the prefix cache
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

Run one of this experiment failed in an instructive way (its file was
removed as invalid; the numbers live in git history and in this
paragraph). The builder's schedules were
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

## Single tenant strict mode, the shipping configuration, validated

Strict mode makes the plan's governance total: untagged requests are
refused at the door, every block a planned request leaves behind
unpinned is stripped on free, and a heuristic eviction raises instead
of silently substituting for the plan. The validated result on the
10,000 document grid: 48.5, 52.3, and 54.7 seconds across the three
configurations, equal to or better than every earlier mode, with zero
heuristic evictions in every run, exact pin accounting, and the
untagged canary refused.

The first strict flight also did exactly what the mode exists to do.
It surfaced an under-claim: the stage questions share their first
sixteen or so tokens, so the block straddling the document and
question boundary was genuinely reused across stages, previously kept
alive by recency luck. Tail stripping evicted it, the invariant made
the cost visible as an honest 8 to 20 percent regression, and the fix
went into the plan's claim (each document's pin now covers the common
question preamble) rather than back into a heuristic. The silent
fallback had been absorbing that accounting gap in every earlier run.

## Phase A, analytical layer: reasoning filters before any GPU run

notes/REASONING_MODEL.md defines the extended cost model (stepwise
generation priced per decode cohort, thinking notes as transient
memory, full waste on failed speculative branches), the Bellman value
recurrences for task-first and for every stage composition of the
document-first family, and the per-policy throughput program. The
implementation lives in docengine/reasoning/ with eight structural
checks in tests/test_reasoning.py, and experiments/run_reasoning.py
produces the crossover map in results/reasoning_sweep.csv.

The anchor comes first: at zero thinking the calibrated model
reproduces the measured 10,000 document world within about ten
percent with no per-cell fitting (pipeline 48.7 against 52.5 measured
seconds at four filters and 0.8, task-first 127.7 against 122.8), so
the extension stands on the measured base rather than beside it.

What the map says, sweeping mean thinking length over 0, 32, 128, and
512 tokens:

- Generation takes over. At 512 thinking tokens the pipeline's query
  grows from 48.7 to 238 seconds at four filters and 0.8, and
  throughput falls from about 205 to 42 documents per second. The
  policy gap compresses as predicted: task-first's penalty shrinks
  from 2.6 times to 1.3 times, because re-reading matters less when
  writing dominates.
- The Bellman optimum is the pure pipeline in every cell. Speculation
  never wins under streaming, and thinking makes it strictly worse
  (30 percent above the pipeline at four filters, 0.8, and 512
  thinking tokens), because a wasted branch now wastes a whole
  reasoning trace. This agrees with the phase C measurement that
  speculation's earlier value was only stage-barrier compensation.
- Blocks shrink and cohorts with them: the per-document footprint
  grows from about 340 to about 2,450 tokens at full speculation with
  512 thinking tokens, exactly the double punishment of large
  lookahead the model note predicts.

The exact layer (docengine/reasoning/exact.py) rebuilds the small-N
certification discipline for the reasoning world: integer documents,
decisions at event boundaries, outcomes as the only randomness, an
offline clairvoyant solver and an online expectimax solver over the
same transitions, with the fixed policies as restricted action sets.
Anchors are hand-computed epochs at N=1 and the exact expectation of
the clairvoyant optimum over all outcome scenarios. Three findings
from the N=3 study (results/reasoning_exact.csv):

- On a starved machine speculation wins even with heavy thinking,
  because tiny decode cohorts pay the weight-read floor regardless of
  width, so waste is free in time while parallelism cuts depth. The
  fluid layer's no-speculation verdict is a saturated-machine
  statement, and the two layers now bracket the regimes.
- When memory binds under thinking, the adaptive optimum beats every
  fixed composition by 33 percent, by mixing speculation depths
  across documents to exactly fill the budget. This is the first cell
  in the project where adaptivity strictly beats all fixed policies,
  it is invisible to the fluid layer by construction, and it is the
  concrete preview of phase B's thesis that binding memory turns the
  mix into a real decision.
- Clairvoyance is worth less than one percent: knowing outcomes in
  advance barely helps, so the value of scheduling lies in resource
  orchestration, not prediction.

The measured phase that follows has a sharp question: do the
generation-dominated makespans, the saturated no-speculation verdict,
and the adaptive gain under memory pressure survive contact with the
engine at 10,000 documents, using the same layered comparison as
before.

## Phase B, analytical layer: a null result with a sharp boundary

The premise was that long documents create the contested memory regime
where the LP's retain-or-re-read decisions matter. The map
(results/longdoc_map.csv, experiments/run_longdoc.py) says otherwise
for a single query, and says precisely why. At one hundred documents
of thirty thousand tokens or thirty documents of one hundred thousand
(three million tokens of corpus either way), the pipeline ties the
Bellman optimum everywhere, its predicted makespan sits on the read
floor (37.5 to 38.2 calibrated seconds), task-first still pays 1.7
times, and thinking barely registers because generated tokens per
document are tiny next to the document itself. The exact solver
agrees: at capacity 2.5 footprints the adaptive gain is zero at long
documents, and boundary probes show why the earlier 33 percent win
was a corner: adaptivity needs a footprint lever (thinking length
comparable to document length), a mix space (three or more stages),
and memory near two to three footprints, together. Long documents
alone deliver none of these; blocked admission absorbs the capacity
constraint with zero extra reads.

Where the long document regime does bind is throughput, not makespan:
the sustained-rate program drops from 0.80 to 0.31 documents per
second at one hundred thousand tokens with thinking, because
residency token-seconds per document explode. The LP's genuine trial
is therefore multi-query or streaming-arrival operation, as the plan
originally suspected under "the LP's native habitat," and the
single-query measured phase B reduces to a cheap validation: the
read-floor prediction at document lengths never measured in this
project (the first real test of the cost model's attention terms at
one hundred thousand token contexts).

## Long documents measured: corrections, a race, a quality cliff

The validation run (results/engine/longdoc.json; one hundred documents
of thirty thousand tokens and thirty of one hundred thousand, two
filters, the shipped configuration, YaRN rope scaling for the 100k
contexts) did three things.

First it corrected the analytical layer. The fluid model had priced
document prefill as dense compute only; the 30k cell measured 2.15
times that floor against 2.23 predicted by the quadratic
self-attention term the port had dropped (the paper's own H formula
carries it; the crossover where attention passes dense compute is
d = P/(L w_Q), about 24,000 tokens for this model). With the term
restored the model predicts 83.7 seconds against 79.8 measured at
30k and 191 against 174.6 at 100k, conservative by five to nine
percent, and the short document anchors are unmoved (the term is 1.3
percent there).

Second, lookahead of two or more is pathological at long documents on
the raw request abstraction. The k=2 cell took 158.5 seconds, almost
exactly twice the k=1 read ratio, with cache hits collapsing to half
a percent: both branches of each document prefilled the full thirty
thousand tokens separately, because the second branch is co-scheduled
before the first branch's blocks commit. The in-flight sharing that
reached its theoretical ceiling at three hundred token documents was
queue depth luck, not a guarantee. The principled fixes are the
engine level mechanisms already planned: sequence truncation
serializes a document's questions on one set of notes, and the
cascade kernel shares the read explicitly.

Third, a quality cliff that scheduling cannot fix: answer agreement
falls from 0.92 at three hundred tokens to 0.81 at thirty thousand
and 0.63 at one hundred thousand, barely above chance. The 4B model
under context extension cannot reliably read a fact at the end of a
hundred thousand tokens. Long document products need the larger
model tier (where the persisted notes store also pays best) or
chunked evaluation, and any long context benchmark of this system
must report agreement next to makespan.

## The request toll, profiled and named

Attach profiling is forbidden by the sandbox, so the profiler runs one
workload three ways: engine core in its own process (21.57 seconds),
engine core in-process (21.68 seconds, so cross-process serialization
costs nothing measurable, a surprise), and in-process under a tracing
profiler with the CPU clock. The ranked table for 9,428 requests
(results/engine/profile.json):

- About 22 percent of all CPU is copy.deepcopy of the sampling
  parameters and its helpers, 367,692 calls, thirty nine per request:
  the engine defensively deep-copies the parameters object for every
  request. The single biggest toll.
- About 10 percent is telemetry (histogram, gauge, and mutex counter
  updates) that nothing reads. Disabled from here on with
  disable_log_stats, a free win.
- About 5 percent is per request input processing and validation, and
  about 4 percent asyncio event machinery; both scale with request
  count.
- Prefix hashing, the suspected culprit, is absent from the top
  thirty: under 0.7 percent.

Consequences: the toll is defensive copying and telemetry, not
communication or hashing; sequence truncation attacks it at the root
by cutting requests per document from n to one; and the deepcopy
share is a candidate upstream contribution independent of anything
else we build.

Follow-up, measured: setting the engine's own skip_clone switch on our
parameters object removes the deep copy entirely (absent from the
re-profiled table), and with telemetry also off, total CPU falls from
28.5 to 16.3 seconds on the reference run. The clock gains only about
0.4 seconds because most of that CPU ran alongside the GPU; the value
is headroom for short-step regimes and a clean remaining table, all of
it per-request machinery that sequence truncation removes wholesale.

## Sequence truncation, milestone one: the chain proof

The build under test: one living engine request per document runs the
whole filter chain. The client registers the question token lists and
the yes token ids once; after that the scheduler judges each sampled
answer in-engine, rolls the sequence back to the document boundary
(question and answer notes erased, document notes kept), and appends
the next question through the engine's own session mechanism. One
request per document instead of one per filter call.

The proof run: 50 documents, two filters, selectivity 0.7 per stage,
planted flag outcomes, both modes cold. Every pass criterion met:

- Surviving documents identical: 18 in both modes, the same 18.
- Every shared answer identical: 80 of 80 agree.
- Requests: 50 in chain mode against 80 in request mode (50 stage-one
  calls plus 30 stage-two calls).
- Rewinds: 30, exactly the number of documents that passed stage one.
- Zero heuristic evictions: the strict single-tenant invariant held
  through all the rewind block surgery.
- Wall clock 0.29 seconds in chain mode against 0.32 seconds in
  request mode at this tiny scale (the corpus is 50 short documents;
  the real payoff test is the 10,000-document grid, milestone three).

Milestones two and three followed. Four-filter chains at 50
documents: 11 of 11 survivors match, 92 of 92 answers, 42 rewinds
for 42 continuations, three per surviving document. Then the
10,000-document run, four filters at selectivity 0.8:

- First flight: chain mode 56.8 seconds against request mode's 52.4.
  The step recorder named the entire gap: chain mode scheduled
  4,264,138 tokens against 3,910,484, and the 353,654 extra tokens at
  the 80,000-per-second operating rate are the 4.4 seconds. Each
  continuation was recomputing the questions' 33-token shared
  preamble that request mode reuses from cache, because the rewind
  erased back to the document instead of document plus preamble.
- With the rewind stopping at document plus preamble (identical for
  every question, so no prompt content changes): chain mode 49.9
  seconds against request mode's 52.0. Chain mode now schedules
  105,094 fewer tokens than request mode (the rewind keeps notes at
  exact token positions; cache hits are 16-token-block granular) and
  the target from the plan - high forties against 52.3 - is met.
- Profiling along the way: chain mode halves client-process CPU (7.1
  seconds against 16.8 at 4,000 documents) and cuts engine-core CPU
  about 16 percent (it hashes half the tokens and polls a quarter as
  many inputs), confirming the per-request paperwork story.
- Honest accounting of disagreements at 10,000 documents: two calls
  of 23,902 flip between modes, at the corpus tail, and the flipped
  pair changes when batch packing changes - border-line cases of the
  uncalibrated 8-bit attention, not scheduler state. Direction is
  symmetric: in two flights both errors were request mode's; in one
  flight each mode misread one, costing chain mode one of 3,524
  survivors. Separately, both modes lose about 560 of the 4,088
  planted survivors identically (roughly one percent of flag lookups
  misread by the model regardless of scheduling); that error is
  model-side and cancels in every between-mode comparison.

The long-document half closes the milestone. One hundred documents of
30,000 tokens, two filters: chain mode 79.0 seconds with 100
requests, bit-identical answers and survivors to the one-in-flight
request plan's 79.0 seconds with 150 requests, while the
two-in-flight plan measured 156.9 seconds (its two branches race,
the hit rate collapses to half a percent, and the corpus is read
about twice). The race is eliminated by construction, not by
scheduling care: a document that is one living request cannot race
itself. The 81 percent flag-reading agreement at this length is the
already-measured long-context quality cliff, identical in both modes.

Three bugs found and fixed on the way, all now encoded in the design
note. First and central: the engine captures the finish reason from
the request status before the stopped-request hook runs and sends it
to the client unconditionally, so a mid-chain stage ended the
client's stream even when the engine continued correctly; the fix
judges the answer inside the stop check and erases the stop status
when the chain continues. Second: the memory block straddling the
document boundary stayed cached with the previous question's content;
the rewind now strips that cache entry. Third: a run tag starting
with "r" made every request id suffix parse as a release directive
and silently swallowed all releases, including the end-of-run flush;
the id parser now never reads the last field as a directive.

One fact that simplifies the risk picture, corrected after reading
the engine's overlap machinery to the bottom: the two-deep batch
queue that overlaps scheduling with execution runs even with our
custom scheduler class (it keys only on a config flag). What a
plain-scheduler subclass loses is decode lookahead - scheduling a
request's next token before the current one lands. A request whose
token is in flight is simply skipped for a step, which is exactly
why the rewind never touches an in-flight request. Our filter calls
sample one token at the end of a prefill chunk and have no decode
steps, so the lost lookahead is worth approximately nothing at this
workload; reasoning filters, with real decode work, are the trigger
to adopt the async scheduler base and a placeholder-aware rewind.

## Persisted notes, milestone one: mechanics proven, 4B loses

The engine's tiering store (RAM primary, container-local disk
secondary) ran end to end at 2,000 documents: query one computed
cold and offloaded 53.1 GB of notes; after a full prefix-cache
reset, queries two and three restored from the store instead of
recomputing. Measured on one container: baseline cold 11.0 seconds
and recompute-after-reset 10.5; with the store, first query 29.5
seconds (the write path), restores 20.3 then 18.4. Raw disk write
measured 5.2 GB/s.

This is the break-even table's prediction landing: the corpus is 53
GB of notes but only about 630,000 tokens of compute, the GPU
re-reads at about 80,000 tokens a second, so beating an 8-second
recompute requires moving those bytes at about 7 GB/s - above this
disk before any bookkeeping. At this model size the tier loses
roughly two to one. The value case is unchanged from the plan: the
32B tier, where prefill runs about eight times slower and the bytes
per token barely grow, turns the same arithmetic into a several-fold
win; durable S3 with a disk cache rides the same mechanics.

Caveats: survivor lists across the four runs are not identical -
consistent with the measured borderline-call noise between any two
batch compositions, but per-call diffs were not instrumented here,
so equality-under-restore is an open item for milestone two. The
store ran on the stock scheduler; reconciling it with strict
plan-owned memory is named follow-up work. The sandbox lacks the
store's page pre-fault call, which is worked around by running the
engine core in-process and letting pages fault in lazily.

## The 32B tier, first measurement: chain mode at 1.07 times the floor

One thousand documents (323,541 tokens), four filters at 0.8, on
Qwen3-32B-FP8, where prefill runs at a measured 10,800 tokens per
second (the plan estimated 9,300) and the corpus slightly overflows
the memory pool, so memory policy binds. Four arms, one flight:

- Stage-major gated waves (stock engine): 49.5 seconds, reading 1.68
  times the corpus.
- Naive streaming client (stock engine): 48.1 seconds, 1.68 reads.
- Ranked pinned requests (strict scheduler): 54.5 seconds, 1.25
  reads - fewer reads, more time. Under pool overflow the pins hold
  memory that blocks admission; holding a document resident until
  its chain finishes costs more than the re-reads it saves. The pin
  discipline needs an overflow policy before it ships on this tier.
- Chain mode: 32.0 seconds, 1.14 reads - 1.07 times the 30-second
  read floor, and 33 to 41 percent faster than every other arm. One
  request per document also pays the six-token sampling change only
  once per stage actually needed.

The read-floor arithmetic that decides everything on this tier: a
corpus pass costs 30 seconds of compute but its notes are 41 GB, so
the persisted-notes break-even flips - at the measured 5 GB/s disk,
restore is about 8 seconds against 30 of recompute, the several-fold
win the plan projected.

Two protocol findings. The 32B model does not keep the one-token
answer contract: it restates the flag line (the answer arriving as a
fused "=YES" token in position four) or chatters, so the gate grew a
decisive-token mode - register no-tokens alongside yes-tokens, allow
several tokens per stage, stop at the first decisive one (the 4B
path is unchanged by construction). With it, chain mode decided
2,616 calls with only 9 contradicting the planted flags. But about
ten percent of calls produce no decisive token within six tokens
(the model chatters before deciding), and an indecisive stage kills
the chain: every arm lands about 270 survivors against a realized
truth of 406. The open follow-up is a longer decode budget for the
indecisive tail (with the early stop, decided calls never pay it) -
or accepting that a fixed one-line prompt underdetermines a
reasoning-tuned model's output format.

## Multi-GPU scaling: the planner shards, the floors divide

The declarative layer (plan_query) chooses the physical plan from
the calibrated models; its first measured validation is GPU scaling
on the banked 10,000-document query (49.9 seconds on one H100):

- 2 GPUs: 25.30 seconds, 1.97 times faster. Shards balanced to
  within 38 tokens of 1.58 million; worker walls within 0.09s.
- 4 GPUs: 13.30 seconds, 3.75 times faster. Shards within 39
  tokens; walls within 0.40s.

Documents are independent, so workers share nothing; each computes
the same deterministic plan and runs chain mode on its shard. The
bend from 4.0 to 3.75 is the fixed per-query software residue that
does not shrink with the shard. Survivors: 3,523 and 3,526 against
the single-GPU 3,524 - the known borderline-call noise, a few calls
in 24,000 landing differently in different batch compositions.

## Answer accuracy against planted truth: the noise was the model

Scoring every call against the planted flags at 10,000 documents and
four filters (ideal: 4,096 surviving documents) overturned the
working assumption that 8-bit notes cost about a percent of answers:

- Shipping fp8, scale 1.0: 2,929 of 23,904 calls wrong (12.3
  percent), 3,524 survivors. The most accurate configuration.
- Runtime-calibrated fp8: 3,371 of 22,869 wrong (14.7 percent),
  3,167 survivors. The deprecated calibration path computes scales
  from whatever runs first, which is not representative data.
- Full-precision (bf16) notes: 3,979 of 21,635 wrong (18.4 percent),
  2,615 survivors, and half the memory pool. Moving the attention
  numerics away from what this fp8-quantized checkpoint was tuned
  for hurts rather than helps.

Two facts frame this. First, request mode and chain mode produce
identical wrong sets under every precision (two borderline flips in
roughly 24,000 calls, direction varying run to run): the scheduler
machinery is precision-independent and bit-faithful; the misreads
are the model's reading accuracy on this task, visible since the
first 2,000-document grids as the 0.92 per-call agreement. Second,
the earlier "two misreads in 23,904" statement was mode agreement,
not accuracy - the truth scoring did not exist yet.

Decision: ship fp8 with scale 1.0 - most accurate of the three and
the full 981k-token pool. The remaining accuracy lever is a
checkpoint calibrated offline on representative data (or a stronger
model), a model artifact question, not an engine one.

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
