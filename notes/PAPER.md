# Plan-Governed KV State for Semantic Scans (working draft)

This note is the paper's working skeleton: the formal problem, the
lower bound, the retention theory, the contributions, and the twelve
experiments. Every measured number is taken from notes/RESULTS.md and
notes/SCHEDULER_PLAN.md and the raw result files they name; expected
outcomes are stated in advance and labeled as such.

## 1. Problem statement

A semantic scan applies n natural-language predicates to a corpus of
N documents by prompting a large language model (LLM) once per
required (document, predicate) pair. Predicates come in three
classes. A filter is gated: a document that fails one predicate skips
the rest. A classifier set requires every predicate's answer for
every document. A join predicate is evaluated over pairs of
documents. Prompts are a fixed interface: the system may not reword
them. The device is fixed and characterized by measured constants:
its sustained compute rate, its memory bandwidth, and the size of its
KV pool, where KV is the key-value cache — the per-token attention
state a transformer stores so it can extend a prefix instead of
recomputing it. The objective is to minimize makespan, the wall-clock
time from query start to the last answer, subject to fidelity:
execution must produce token-identical answers to the reference that
issues one isolated model call per (document, predicate) pair.

## 2. Notation

| Symbol | Meaning |
|---|---|
| D, N | the corpus; its number of documents |
| d_j | document j |
| p_1..p_n, n | the predicates; their count |
| C | total token count of the corpus |
| M | the serving model |
| P | parameter count of M |
| W | weight bytes of M |
| kappa | KV bytes produced per token |
| F | device peak floating-point operations per second |
| B | device peak memory bandwidth, bytes per second |
| F', B' | measured sustained rates at our workload shapes |
| T_low | the lower-bound makespan of Proposition 1 |
| s | selectivity of a filter: the fraction of documents that pass |
| k | lookahead depth: predicate prompts issued without waiting for a gate |
| G | mean thinking length: reasoning tokens generated before an answer |
| z_i | size in bytes of KV segment i |
| r_i | recompute time in seconds of KV segment i |
| K | KV pool size in bytes |
| beta | store bandwidth in bytes per second (either direction) |

## 3. Execution model and lower bound

Definition 1 (semantic scan). A semantic scan S = (D, p_1..p_n)
applies predicates p_1..p_n to documents d_1..d_N. Each predicate is
a fixed prompt text; evaluating p_i on d_j means running M on d_j
followed by p_i's prompt text and reading the answer from M's output.

Definition 2 (faithful execution). An execution is faithful if, for
every (document, predicate) pair it evaluates, it produces the same
answer tokens as evaluating that pair alone in a fresh context.

Assumption 1 (exact evaluation). Every pair the query semantics
require is evaluated by M at serving precision. No model cascade
(routing some inputs to a smaller model), no prompt modification, no
input truncation.

Assumption 2 (cold start). No KV for D exists before the query
begins.

Proposition 1 (scan lower bound). Let C be the total token count of
D, P the parameter count of M, W its weight bytes, kappa the KV bytes
per token, and let the device sustain F floating-point operations per
second and B bytes per second of memory bandwidth. Any faithful
execution under Assumptions 1-2 has makespan at least

    T_low = max( 2 P C / F ,  (W + kappa C) / B ).

Proof sketch. A forward pass over one token performs at least 2P
floating-point operations, one multiply and one add per parameter.
Assumption 1 forces every document token through a forward pass at
least once, so total compute is at least 2PC and compute time at
least 2PC/F. By Assumption 2 the corpus KV does not exist at start,
so any execution writes its kappa C bytes at least once, and any
forward pass reads the W weight bytes at least once; together at
least (W + kappa C)/B seconds pass on the memory system. F and B are
device-wide maxima, so the larger term bounds any schedule.

The spec-sheet bound is loose on real silicon, so we also state the
bound with the measured sustained rates F' and B' in place of F and
B, and report the engine against both. On our reference query
(Section 7, E1) the spec bound is 11.3 seconds, the measured read
floor is 39.5 seconds, and the measured work floor — corpus plus
question and answer tokens — is about 47 seconds.

Relaxing an assumption defines a variant problem whose bound follows
by the same argument. Warm start (persisted KV already available)
removes the kappa C write and the forward passes over document
tokens. A model cascade replaces P by the smaller model's parameter
count on the fraction of pairs the smaller model settles. Span
evaluation — answering a predicate from a selected span rather than
the whole document — replaces C by the span total. Each results
section states which assumptions are in force for its claims.

## 4. Thesis: KV as a plan-governed object

Serving engines treat KV as an anonymous cache managed by a recency
rule: evict whatever was used least recently. Our thesis is that for
semantic scans, KV should be a
first-class object whose lifetime the query plan governs. Five
operations on KV generate the physical operator space:

- Fork: start a second branch that reads an existing document KV as
  its shared prefix, instead of recomputing the document.
- Rewind: roll a live sequence back to a marked token position,
  discarding the KV after that position, then append new tokens.
- Fuse: evaluate several forked branch suffixes in one pass over the
  shared document KV, using a shared-prefix attention kernel.
- Spill: move a KV segment to a slower store to free pool space, and
  read it back before its next use.
- Persist: write KV durably at corpus ingest so later queries load
  it instead of recomputing it.

An optimizer with measured device constants chooses among the
resulting operators per query. Two execution primitives cover the
workload family. Rewind execution runs a gated filter chain as one
live sequence per document: append a question, read the answer,
rewind to the document boundary, append the next question. Forked —
and, when priced favorably, fused — branches run all-answers sets,
predicate sets where every answer is required and gating is
impossible. Filters, classifier sets, and joins are nestings of
these two primitives, indexed by the selectivity vector, the
document length, and the mean thinking length G.

## 5. The plan-aware KV retention problem

The plan fixes, before execution, the order in which KV segments are
read; retention is then a well-posed offline problem rather than a
cache heuristic. We state it, settle its decoupled variants, and
leave the coupled variant open.

Problem 1 (plan-aware KV retention). A KV segment is the KV of one
document or shared prefix; segment i has size z_i bytes and
recompute cost r_i seconds, the time to rebuild it by re-running the
forward pass over its tokens. The input is the access sequence
a_1..a_m in which the plan reads segments, the pool size K, and the
store bandwidth beta, the rate at which bytes move between the pool
and a slower store in either direction. Between two consecutive
accesses to a segment — a gap — the schedule chooses one of three
actions: keep (the segment occupies z_i bytes of the pool for the
whole gap, at no time cost), drop (the space is freed now and the
next access pays r_i to recompute), or spill (write z_i bytes out in
z_i/beta seconds, free the space, and read z_i bytes back in
z_i/beta seconds before the next access). Kept segments must fit in
K at every point. Minimize the total time added to the plan's base
compute.

Claim 1 (decoupled prices). Suppose spill traffic is charged as pure
serial delay: transfers overlap nothing and do not share beta with
each other or with the schedule's own reads. Then any non-keep
decision affects only its own gap, so the cheaper alternative is
always taken, and the price of not keeping segment i across a gap is
min(r_i, 2 z_i / beta). The problem reduces to offline caching with
a per-gap eviction price: choose which segments stay resident under
capacity K, paying min(r_i, 2 z_i / beta) each time segment i is not
retained across one of its gaps.

Claim 2 (complexity of the decoupled problem). With uniform segment
sizes, the decoupled problem is offline weighted caching — unit-size
items with item-specific eviction prices — and is solvable in
polynomial time by a minimum-cost flow formulation, a network
optimization solvable exactly in polynomial time; with uniform
prices as well, it collapses to Belady's rule, evicting the segment
whose next access is farthest in the future, so the formulation is a
generalization of Belady's algorithm. With heterogeneous segment
sizes the decoupled problem contains offline caching with item sizes
and prices (general weighted caching), and it inherits that
problem's NP-hardness unchanged, because by Claim 1 the keep, drop,
and spill choices contribute only a fixed per-gap price.

Claim 3 (the bandwidth-coupled variant is open). In the engine,
spill transfers share bandwidth with the schedule's own weight and
KV reads and can overlap compute. The effective price of a spill
then depends on how much bandwidth slack the schedule has while the
transfer runs, which couples every gap decision to the global timing
of the plan. We state this variant — minimize makespan when
transfers may overlap compute and all traffic shares beta — as an
open problem. We claim neither an algorithm nor a hardness result
for it. The engine's spill scheduler is a heuristic for this
variant and is evaluated empirically against a clairvoyant reference
in E9.

## 6. Contributions

1. The engine. Plan-governed KV lifetimes inside a production
   serving engine: admission by KV budget (a document enters only
   when its KV fits the pool), pins for live documents (references
   that make their KV ineligible for eviction), eager frees at
   document death, and a strict mode in which any
   heuristic eviction raises an error — zero occurred across every
   validated run. Rewind execution reaches 1.07 times the read-floor
   lower bound on the 32B tier (32.0 seconds, compared with the
   30-second floor). Two protocol rules the runs forced: the rewind
   boundary rule (rewind to the document plus the questions' shared
   33-token preamble, not to the document alone), and the
   decisive-token gate (register the yes and no token ids, allow a
   few answer tokens, stop at the first decisive one) for models
   that do not keep a one-token answer format.
2. The optimizer. A cost model in which every constant is a measured
   number attributed to a named calibration run; an O(n^2) dynamic
   program over predicate partitions that is exact under the fluid
   model, the relaxation that prices work as long-run rates and
   ignores batch boundaries; a feasibility gate that refuses
   device-model pairs whose bound cannot be met rather than emitting
   a bad plan; and transfer to a new device from two measured
   constants, its sustained token-reading rate and its KV pool size.
3. The theory. The plan-aware KV retention problem of Section 5:
   formal statement, the price decomposition and complexity results
   for the decoupled variants, the bandwidth-coupled variant stated
   as open, and an engine implementation evaluated in E9.
4. The fused-fork operator. For all-answers sets, a shared-prefix
   attention kernel evaluates all branch suffixes in one pass over
   the document KV. It is correctness-gated — enabled only where
   answers are bit-identical to unfused execution — and the planner
   prices it from the measured crossover surface of E5.
5. The method. A floor discipline applied to every claim: state the
   lower bound, solve the rate program, construct the schedule,
   measure the run, and check it with an independent validator — on
   both model tiers, and across GPU counts (measured at one, two,
   and four GPUs; the eight-GPU point is planned).

## 7. Experiment plan

E1 through E4 are complete and are rerun only if a refactor changes
their numbers; E5 through E12 are planned, with expected outcomes
stated in advance. The reference query throughout: 10,000 IMDb
reviews (2,966,000 document tokens), four filters at selectivity
0.8, Qwen3-4B-FP8 on one H100, cold start; its measured read floor
is 39.5 seconds and its work floor about 47 seconds. The 32B tier
is Qwen3-32B-FP8 at 1,000 documents (323,541 tokens), where prefill
— the forward pass that reads prompt tokens and builds their KV —
runs at a measured 10,800 tokens per second and the read floor is
30 seconds.

E1, control ladder (measured). Goal: show how much of the gap to
the lower bound each level of plan control recovers. Setup: the
reference query under five arms — naive with the task text before
the document (no prefix shared), naive with the document first
(prefix reuse possible), the client-side scheduling library, the
in-engine strict scheduler, and rewind execution — plus the same
ladder's endpoints on the 32B tier. Outcome: 122.8, 111.3, 52.5,
52.3, and 49.9 seconds against the 47-second work floor, with
corpus reads falling from 3.03 times the corpus to 1.23; on the 32B
tier rewind execution takes 32.0 seconds at 1.14 corpus reads,
compared with 48.1 to 54.5 seconds for every other arm.

E2, GPU scaling (measured at 1-4, planned at 8). Goal: show that
the planner's sharding divides the floor with negligible residue.
Setup: the reference query sharded over 1, 2, 4, and 8 GPUs with
token-balanced shards, each worker computing the same deterministic
plan. Outcome: 25.30 seconds on two GPUs (1.97 times faster than
the 49.9-second single-GPU run) and 13.30 seconds on four (3.75
times, 94 percent efficiency), shards balanced within 39 tokens;
the eight-GPU point is expected to hold near 94 percent because
shards share nothing.

E3, tier reversals (measured). Goal: show that the same mechanism
flips between winning and losing across model tiers exactly as the
cost model prices it. Setup: the pin discipline and the persisted-KV
restore path, each run on both tiers. Outcome: pins match the
client library at 4B (49.2 versus 48.9 seconds) but lose under 32B
pool overflow (54.5 seconds at 1.25 corpus reads, against 48.1 for
an unpinned streaming client — fewer reads, more time); restore
loses about two
to one at 4B (20.3 and 18.4 seconds against 10.5 of recompute, at
a measured 5.2 GB/s of disk bandwidth) and flips at 32B (about 8
seconds of restore against 30 of recompute).

E4, reasoning grids (analytical layer measured; GPU run planned).
Goal: locate where generation overtakes reading and what that does
to policy gaps. Setup: sweep the mean thinking length G over 0,
32, 128, and 512 tokens through the calibrated cost model, its
dynamic-programming value recurrences, and the exact solver for
small corpora, anchored at G = 0 to the measured grid within about
ten percent (48.7 predicted versus 52.5 measured seconds).
Outcome: at G = 512 the reference query grows from 48.7 to 238
seconds and the task-prompt arm's penalty compresses from 2.6
times to 1.3; speculation — issuing predicate prompts without
waiting for the gate — is strictly worse when the machine is
saturated but wins when starved; adaptive depth mixing
beats every fixed policy by 33 percent only when memory binds; the
planned GPU rerun uses a forced-length instrument to validate all
three on hardware.

E5, fusion gate and crossover (planned). Goal: measure the surface
from which the planner prices the fused-fork operator, and validate
its correctness gate. Setup: six classifiers over documents of 300,
3,000, and 15,000 tokens, with thinking on, under fused, unfused
forked, and sequential execution. Expected outcome: fusion's gain
grows with document length as the shared attention read approaches
the dense-compute crossover (about 24,000 tokens on the 4B model),
and the gate admits only cells with bit-identical answers.

E6, sharing ladder (planned). Goal: price fork and fuse separately.
Setup: one all-answers workload run with forking disabled (every
branch re-reads its document), with forking, and with forking plus
fusion. Expected outcome: a three-row table attributing read
savings to fork and attention-read savings to fuse.

E7, tagging at scale (planned). Goal: demonstrate the all-answers
regime end to end at production width. Setup: 40 predicates over
long documents, executed by the planner's choice of forked and
fused branches. Expected outcome: makespan tracks the read floor
plus the priced branch work, with no per-predicate re-reads.

E8, interactive boundary (planned). Goal: measure the corpus size
at which gating stops paying, and check the planner reproduces it.
Setup: N in {20, 100, 500, 2000}, rewind execution against
ask-everything (evaluate all predicates ungated). Expected outcome:
ungated evaluation wins at small N where the device is starved —
as the exact solver already shows at N = 3 — and gating wins at
large N, with the measured boundary matching the planner's.

E9, retention in practice (planned). Goal: test the Section 5
heuristic where spilling should beat recomputing. Setup: a
length-variance workload with natural thinking lengths on the 32B
tier, comparing the planner's spill schedule, the engine's default
preemption (evict a running request's KV under pressure and
recompute it later), and a clairvoyant reference schedule computed
with all outcomes known in advance. Expected outcome:
the planner lands near the clairvoyant reference and ahead of
preemption, and the gap to the reference bounds what the open
coupled variant could still recover.

E10, planner end to end (planned). Goal: show the optimizer's
choices are the right ones, not just defensible ones. Setup: the
planner drives every operator choice live on the E5 through E9
workloads, and each rejected alternative is also run under
identical conditions. Expected outcome: every emitted plan beats
the alternatives the planner rejected.

E11, wrong-prior robustness (planned). Goal: measure the cost of
bad selectivity estimates and the value of re-planning. Setup:
plans built from selectivities mis-set by controlled factors, with
and without online re-planning from observed pass rates. Expected
outcome: online re-planning recovers most of the gap to the
correctly informed plan.

E12, device transfer (planned). Goal: show the cost model carries
to a new device from two measured constants. Setup: predict the 4B
configuration on an L40S from the L40S's measured token-reading
rate and KV pool size, then run it; also submit the 32B-on-L40S
configuration to the planner. Expected outcome: measurements land
within the model's established five-to-ten-percent envelope, and
the 32B-on-L40S configuration is rejected by the feasibility gate —
reported as a refusal, not a run.

## 8. Non-claims

We claim no new attention kernel: fusion uses existing shared-prefix
primitives (FlashInfer), and the contribution is the operator and
its planning. We claim no prompt rewriting: prompts are a fixed
interface, and fidelity is token-identity with the one-call
reference. Adaptive mid-query recomposition is deferred: its
measured trigger — three or more stages, thinking length near
document length, and a pool near 2.5 per-document footprints, all
at once — is stated, and outside that corner fixed plans tie the
adaptive optimum. Model accuracy is reported, not claimed: at the
shipping precision 12.3 percent of calls disagree with planted
ground truth identically across scheduling modes, and long-context
agreement falls from 0.92 at 300-token documents to 0.63 at
100,000 tokens; scheduling neither causes nor fixes this.

## 9. Related work: positioning checklist

The question asked of each line of work: does it plan engine state
from query semantics against a stated lower bound?

- Sarathi-Serve: engine scheduling (chunked prefill, stall-free
  batching) that is blind to query structure.
- Hydragen and cascade attention primitives: the fusion mechanism,
  with no planner deciding when it pays.
- SGLang RadixAttention and Parrot: reactive prefix reuse and
  inter-request hints discovered at run time, not planned KV
  lifetimes with declared retention.
- LOTUS, Palimpzest, DocETL: semantic-operator optimizers that
  choose calls and models above a black-box serving engine, with no
  control of engine state.
- vAttention, CacheGen, CacheBlend: KV transport, layout, and
  loading mechanisms — complements to spill and persist, with no
  query-level plan.

For each we verify the answer in a reading pass before submission.
