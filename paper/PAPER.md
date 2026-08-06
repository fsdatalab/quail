# Plan-Governed KV State for Semantic Scans (working draft)

STATUS, 2026-08-04. Every measured number in this draft was
calibrated against an achieved read rate of 80,556 tokens per
second (the control arm of results/engine/xengine.json, which
reproduces the speed_limit.json calibration configuration). That
rate was later shown to be an artifact of our Docker image, not a
property of the engine or the hardware: the same vllm==0.26.0
reads 97,220 tokens per second on a CUDA 13 devel base image, and
SGLang reads 98,746 (xengine.json, six arms). The re-baseline flight
(experiments/REBASELINE.md) re-measures the headline cells on the
fixed image. Until it lands, treat every measured number below as a
placeholder at the old anchor. Ratios between arms measured on the
same image should move less than absolute seconds, but that is an
expectation, not a measurement.

This note is the paper's working skeleton: the formal problem, the
lower bound, the optimizer with its algorithms and proofs, the
retention theory, the contributions, and the thirteen experiments.
Every measured number is taken from a named result file in
results/, with notes/RESULTS.md as the ledger; expected outcomes
are stated in advance and labeled as such. Where an algorithm box
and the shipped code differ, the difference is stated next to the
box.

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
| p_1..p_n, n | the predicates, in fixed order; their count |
| C | total token count of the corpus |
| M | the serving model |
| P | parameter count of M |
| W | weight bytes of M |
| kappa | KV bytes produced per cached token |
| F | device peak floating-point operations per second |
| B | device peak memory bandwidth, bytes per second |
| F', B' | achieved sustained rates at our workload shapes |
| T_low | the lower-bound makespan of Proposition 1 |
| s_i | selectivity of predicate i: the fraction of documents that pass it |
| surv(i) | survival after predicates 1..i: s_1 s_2 ... s_i, with surv(0) = 1 |
| G | mean thinking length: reasoning tokens generated before an answer |
| g | GPUs per worker: the model split |
| u | worker count: available GPUs divided by g, rounded down |
| H | token count of the heaviest shard (a shard: the documents one worker owns) |
| L_d | mean document length in tokens |
| l_i | prompt length of predicate i in tokens |
| l_pre | length of the shared question preamble (33 tokens measured) |
| R | achieved sustained token-reading rate, tokens per second |
| R_dec | achieved sustained decode (generation) rate, tokens per second |
| f | per-document KV footprint in bytes: kappa times the document's resident tokens |
| w | width: number of documents resident concurrently |
| W* | saturation width: smallest w that keeps the device fully busy |
| c_req | per-request software overhead in seconds |
| c0 | fixed per-query software overhead in seconds |
| X | document tokens a plan re-reads because retention dropped their KV |
| z_i, r_i | size in bytes and recompute seconds of KV segment i (Definition 3) |
| K | KV pool size in bytes: card memory left after the weights |
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
require is evaluated by M at serving precision. No smaller stand-in
model, no prompt modification, no input truncation.

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
same form with the achieved sustained rates F' and B' in place of F
and B, and report the engine against both. Terminology, fixed here
and used throughout: "floor" is reserved for the proven bound of
Proposition 1. The form built from achieved rates is a read
estimate — or, with question and answer tokens included, a work
estimate. It is an estimate, not a floor, because an achieved rate
can be beaten, and ours was: the 80,556 tokens per second behind
every estimate below rose 21 percent on a better-built image (see
the status block). On our reference query (Section 8, E1) the spec
bound is 11.3 seconds; at the old anchor the read estimate is 39.5
seconds and the work estimate — corpus plus question and answer
tokens — about 47 seconds.

One relaxation is built and measured, and it is why Assumption 2
must be stated: warm start. If the corpus KV was persisted by an
earlier query, execution loads it instead of recomputing it, which
removes the kappa C write and the forward passes over document
tokens, and the bound follows by the same argument with those terms
deleted. E3 measures both sides of this trade: on the 4B tier
loading loses about two to one against recompute, and on the 32B
tier it wins — about 8 seconds of restore, compared with 30 seconds
of recompute. Each results section states which assumptions are in
force for its claims.

## 4. Thesis: KV as a plan-governed object

Serving engines treat KV as an anonymous cache managed by a recency
rule: evict whatever was used least recently. Our thesis is that for
semantic scans, KV should be a first-class object whose lifetime the
query plan governs. Five operations on KV generate the physical
operator space:

- Fork: start a second branch that reads an existing document KV as
  its shared prefix, instead of recomputing the document.
- Rewind: roll a live sequence back to a marked token position,
  discarding the KV after that position, then append new tokens.
- Fuse: evaluate several forked branch suffixes in one pass over the
  shared document KV, using a shared-prefix attention kernel.
- Spill: move a document's KV to a slower store to free pool space,
  and read it back before its next use.
- Persist: write KV durably at corpus ingest so later queries load
  it instead of recomputing it.

The optimizer of Section 5 chooses among the resulting operators per
query, from measured device constants. Two execution primitives
cover the workload family. Rewind execution runs a gated filter
chain as one live sequence per document: append a question, read the
answer, rewind to the document boundary, append the next question.
Forked branches run all-answers sets, predicate sets where every
answer is required and gating is impossible. Fuse belongs to the
operator space but is closed on this stack: we built it, and its
fidelity gate rejected the production kernel on every measured cell
(Section 8, E5). Filters, classifier sets, and joins are nestings of
these two primitives, indexed by the selectivity vector, the
document length, and the mean thinking length G.

## 5. The optimizer: cost estimation, plan enumeration, grouping

The optimizer has three parts: a cost estimator that predicts the
makespan of one candidate plan (Algorithm 1), an enumerator that
filters illegal plans and takes the minimum over the rest
(Algorithm 2), and a dynamic program that groups predicates into
blocks (Algorithm 3). All three work in the fluid model: costs are
computed from long-run rates, work is treated as continuously
divisible, batch boundaries are ignored, and time is charged
additively, so a predicted makespan is a sum of terms. Every device
constant consumed here is a measured number attributed to a named
calibration run; on the reference hardware the calibrated estimator
predicts every measured cell within about five to ten percent.

Assumption 3 (length-independent selectivity). Whether a document
passes a predicate is independent of the document's length, so a
survival fraction applies uniformly to the corpus token mix.

### 5.1 Algorithm 1: the cost estimator

Input: a candidate plan p = (execution mode, predicate blocks, model
split g, shard assignment, access, retention, width w); workload
statistics (N, L_d, l_1..l_n, l_pre, G, s_1..s_n); device constants
(R, R_dec, K, kappa, beta, c0). Output: the predicted makespan of p
in seconds.

```
Algorithm 1  COST(p)

 1  surv(0) <- 1;  surv(i) <- surv(i-1) * s_i        for i = 1..n
       // survival after predicate i (Assumption 3)
 2  A_j <- surv(j-1)  for j = 1..n
       // fraction of documents that issue predicate j when every
       // stage is gated; inside an ungated block of Algorithm 3
       // every member is charged at the block's entering survival
 3  H <- token count of the heaviest shard
 4  T_in <- H / R                 if access = read
        <- kappa * H / beta       if access = restore
       // input pass: read every shard token once, or load the
       // persisted KV bytes from a warm store, whichever the plan
       // selected (Algorithm 2, line 13)
 5  T_quest <- (N / u) * ( A_1 * l_1
                + sum_{j=2..n} A_j * (l_j - l_pre) ) / R
       // stage 1 pays its full question; each later stage pays
       // only the tail past the shared preamble, thinned by
       // selectivity
 6  T_reread <- X / R
       // X: document tokens recomputed because retention dropped
       // their KV; X = 0 in rewind execution, where admission
       // keeps every live document resident (what a plan drops
       // and re-reads is priced by Section 6)
 7  T_dec <- (N / u) * ( sum_{j=1..n} A_j * (G + 1) ) / R_dec
       // decode: thinking plus one answer token per reached
       // stage; up to six tokens under the decisive-token window
       // (Algorithm 2, line 14)
 8  T_mem <- 0                    if w * f <= K
        <- overflow stall time    otherwise
       // nonzero only when the resident footprint exceeds the
       // pool; the overflow is priced by Section 6
 9  return T_in + T_quest + T_reread + T_dec + T_mem + c0
```

The constant c0 is the fixed per-query software overhead, calibrated
once per device; it absorbs the per-request cost c_req (measured
0.55 to 1.12 milliseconds per request, one request per document in
rewind mode) at the plan's request count. The shipped planner
calibrates c0 at 3.2 seconds on the reference H100.

What the shipped code computes. Algorithm 1 lives in one module,
docengine/plan/cost.py, one helper per line. plan_query in
docengine/plan/planner.py consumes it with one question length
shared by every stage, and with X = 0, T_dec = 0, and T_mem = 0:
at G = 0 with one-token answers the decode term is small enough to
sit inside the calibration residue that c0 absorbs, and the
admission budget of Algorithm 2 makes T_mem = 0 by construction.
The full terms are priced by the same module and exercised by the
reasoning layer (docengine/reasoning/model.py), which prices decode
stepwise and memory by footprint, and which anchors to the measured
grid within about ten percent at G = 0 (48.7 predicted, compared
with 52.5 measured seconds; E4). The reading rate R is
derived from one calibrated ratio — the 80,000 tokens per second
achieved at the old anchor (status block) over the 275,000 spec
ceiling at the 4B tier — applied to the spec compute cost 2P per
token; on the 32B tier this predicted about 9,300 tokens per
second, roughly fifteen percent under the achieved 10,800. The
ratio is recalibrated by the re-baseline flight.

### 5.2 Algorithm 2: the plan enumerator

Input: the scan S = (D, p_1..p_n) and the device table V (card
count, per-card memory M_dev, R, R_dec, kappa, W*, and beta with a
warmth flag if a persisted-KV store exists). Output: the cheapest
legal plan, or a refusal naming the violated constraint.

```
Algorithm 2  PLAN(S, V)

 1  candidates <- empty
 2  for g in powers of two with W <= 0.92 * M_dev * g:
       // model split: the weights must fit the g-card group;
       // 0.92 is the shipped memory-utilization fraction
 3    if g > card count:  record "weights need g cards"; break
 4    u <- floor(card count / g)          // worker count
 5    shards <- greedy balance: visit documents in descending
         token order, assign each to the currently lightest shard
 6    H <- token count of the heaviest shard
 7    K <- 0.92 * M_dev * g - W           // pool bytes per worker
 8    if K < W* * f:  record "pool under saturation width";
         continue
       // feasibility gate: the pool left after weights must hold
       // the saturation width times the per-document footprint
 9    mode <- single requests   if n = 1
          <- rewind chain       else if the set is gated (filters)
          <- forked branches    else (every answer is required)
10    budget <- min( 0.80 * K / kappa,
                     max(longest document + l_1, H) )
       // admission budget in tokens: a document enters execution
       // only when its KV fits the remaining budget; 0.80 keeps
       // headroom under the pool
11    pin <- (mode = single requests) and n >= 2
             and H <= K / kappa
       // a pin: a reference that makes KV ineligible for
       // eviction; only for a multi-filter request plan whose
       // shard fits the pool (E3 measured pins losing under
       // overflow)
12    blocks <- GROUP(p_1..p_n)           // Algorithm 3
13    access <- read;  if the store is warm and
         kappa * H / beta < H / R:  access <- restore
       // restore pays only the read: the write was paid at ingest
14    window <- 1 if M keeps the one-token answer format, else 6
       // decisive-token window: register the yes and no token
       // ids, allow up to six answer tokens, stop at the first
       // decisive one (the 32B tier restates or chatters)
15    add plan(mode, g, u, shards, budget, pin, access, blocks,
         window) to candidates
16  if candidates is empty:  return REFUSE with the recorded
       constraint            // reported as a refusal, not a run
17  return argmin over candidates of COST (Algorithm 1)
```

What the shipped code computes, and where it diverges. plan_query
implements lines 3-8, 10-11, 13-14, and 16 as written: the greedy
shard balance is _balanced_shards, the budget and pin lines are
term-for-term the code's rules, and the restore comparison is the
measured break-even that lost two to one at 4B and wins
several-fold at 32B (E3). The refusal path was a stated divergence
until 2026-08-04 and is now implemented: plan_query returns a
Refusal naming the violated constraint — and the needed and
available quantities in one unit — when the weights need more cards
than exist (line 3), when the weights leave no pool at all (the
negative pool is no longer clamped to zero), or when the pool
cannot hold the saturation working set (line 8). W* enters line 8
as a caller-supplied width in documents, default 1 — the honest
minimum, one resident document, until W* is measured per device
(E12; tests/test_planner.py exercises all three refusals). Three
divergences remain, stated so the reader can check them against the
code. First, plan_query evaluates only the smallest split g that
fits the weights, not every legal g; a larger split trades workers
for pool per worker, and pricing that trade is left to the
enumerator form above. Second, line 9's mode choice ships as a rule
rather than an argmin: rewind chains won every measured regime with
two or more filters (E1: 49.9 against 52.3 seconds at 4B; 32.0
against 48.1 to 54.5 at 32B), so the rule and the argmin agree on
every measured cell; the forked-branch arm is priced by the model
and is exercised by E6 and E7. Third, line 11's pin condition is
unreachable under line 9's rule — single-request mode is assigned
only at n = 1, and the pin needs n >= 2 — so the shipped planner
never emits a pinned plan; the pin path is retained for
configurations where rewind execution is unavailable. plan_query
also fixes blocks of size one at line 12: at G = 0 with no stage
barriers, ungated blocks were measured neutral or slightly harmful
(50.4 against 48.9 seconds at two filters), and Algorithm 3's
grouping becomes a real decision only when thinking tokens make a
wasted branch expensive (E4). The saturation width W* never binds
on the reference device — the measured pool holds 981,728 tokens,
several thousand mean-length documents. Where line 8 gets close is
the E12 device: on an L40S the 32B weights leave a pool of 81,329
tokens, and the gate's verdict there turns on the measured W* (see
E12, whose expected outcome this corrected).

### 5.3 Algorithm 3: predicate grouping

Input: predicates p_1..p_n in fixed order with selectivities
s_1..s_n, and BLOCKCOST(i+1..j, S): Algorithm 1's question, decode,
and retention terms restricted to predicates i+1..j when a fraction
S of the corpus enters the block and every member is issued at block
entry (ungated inside the block). Output: the partition of the order
into contiguous blocks minimizing the summed block costs.

```
Algorithm 3  GROUP(p_1..p_n)

 1  best(0) <- 0;  cut(0) <- none
 2  for j = 1..n:
 3    best(j) <- min over 0 <= i < j of
                   best(i) + BLOCKCOST(i+1..j, surv(i))
 4    cut(j)  <- the minimizing i
 5  return the blocks read off cut(n), cut(cut(n)), ...,
      with cost best(n)
```

Lemma 1 (block decomposition). Under the fluid model and
Assumption 3, for any partition of p_1..p_n (order fixed) into
contiguous blocks B_1..B_m: (i) the fraction of documents entering
B_t equals surv(i), where p_i is the last predicate before B_t, and
does not depend on how p_1..p_i were grouped; (ii) the cost of B_t
is a function BLOCKCOST(B_t, S) of the block's members and its
entering survival S alone; (iii) the plan's total cost is the sum of
its block costs.

Proof. (i) A document enters B_t exactly when it passes every
predicate before B_t, and grouping changes only when those
predicates' prompts were issued, never which predicates they are.
The entering fraction is therefore the product s_1 ... s_i =
surv(i), and by Assumption 3 this fraction applies to the full token
mix, so no factor of it mentions the grouping. (ii) A block's charge
in the fluid model is the sum of its members' question, decode, and
retention terms from Algorithm 1, and each term is a function of the
block's members, the entering fraction, and device constants alone.
No KV state crosses a block boundary: the document KV a surviving
document carries out of a block is the same bytes under every
grouping, so no term of one block reads a decision made inside
another. Hence the block's cost is the well-defined number
BLOCKCOST(B_t, S). (iii) The fluid model charges time additively and
distinct blocks charge distinct work, so the plan's total is the sum
over blocks; substituting (i) into (ii) fixes each summand
independently of the grouping of earlier predicates.

Proposition 2 (optimality of Algorithm 3). Algorithm 3 returns the
minimum of the summed block costs over all partitions of the fixed
predicate order into contiguous blocks, using at most n(n+1)/2
BLOCKCOST evaluations, which is O(n^2).

Proof. By induction on j. Base: the empty prefix has cost
best(0) = 0. Step: any partition of p_1..p_j ends in some last block
p_{i+1}..p_j with i < j. By Lemma 1(i) the survival entering that
block is surv(i) regardless of how p_1..p_i are partitioned, so by
Lemma 1(ii) the last block costs the fixed number
BLOCKCOST(i+1..j, surv(i)); by Lemma 1(iii) the rest of the cost is
the cost of a partition of p_1..p_i, which by the induction
hypothesis is at least best(i), with equality achieved by the prefix
optimum. Hence the optimum for the length-j prefix is the minimum
over i of best(i) + BLOCKCOST(i+1..j, surv(i)), which is what line 3
computes, and best(n) with the recorded cuts is the optimum. The
double loop evaluates BLOCKCOST once per pair (i, j), at most
n(n+1)/2 times.

Implementation and cross-check. Algorithm 3 is group_stages in
docengine/reasoning/model.py. The test suite checks it exhaustively:
for every n <= 6 it enumerates all 2^(n-1) contiguous partitions,
sums the same additive block cost over each, and confirms the
dynamic program returns the same optimum (tests/test_reasoning.py).
One scoping note, which is also a divergence to record: Lemma 1(iii)
needs an additive cost. The reasoning layer also carries a second,
certified value — the maximum of total compute seconds, total
bandwidth seconds, and one block's dependency path — and a maximum
is not additive across blocks, so Proposition 2 does not apply to
it. The implementation minimizes that envelope by enumerating all
2^(n-1) partitions directly (best_composition; eight partitions at
the workload's n = 4), and uses the dynamic program where the
additive fluid cost is the objective.

## 6. The plan-aware KV retention problem

The plan fixes, before execution, the order in which KV is read;
retention is then a well-posed offline problem rather than a cache
heuristic. We state it, settle its decoupled variants, and leave the
coupled variant open.

Definition 3 (KV segment). A KV segment is the KV produced for the
tokens of one document, or of one shared prefix, managed as an
indivisible unit: it is kept, dropped, or spilled whole. The unit is
forced by the semantics — evaluating a predicate on a document
attends over the KV of every one of that document's tokens, so KV
missing any part of its segment cannot answer a predicate and must
be rebuilt in full before use. Segment i has size z_i bytes and
recompute cost r_i seconds, the time to rebuild it by re-running the
forward pass over its tokens.

Problem 1 (plan-aware KV retention). The input is the access
sequence a_1..a_m in which the plan reads segments, the pool size K,
and the store bandwidth beta, the rate at which bytes move between
the pool and a slower store in either direction. Between two
consecutive accesses to a segment — a gap — the schedule chooses one
of three actions: keep (the segment occupies z_i bytes of the pool
for the whole gap, at no time cost), drop (the space is freed now
and the next access pays r_i to recompute), or spill (write z_i
bytes out in z_i/beta seconds, free the space, and read z_i bytes
back in z_i/beta seconds before the next access). Kept segments must
fit in K at every point. Minimize the total time added to the plan's
base compute.

Lemma 2 (per-gap price decoupling). In the decoupled cost model —
every transfer is charged as pure serial delay, overlapping no
compute and sharing beta with no other transfer — some optimal
schedule charges every non-kept gap of segment i exactly
min(r_i, 2 z_i / beta), and the only remaining decision is which
gaps to keep.

Proof. Fix any feasible schedule and any gap in which segment i is
not kept. Drop frees z_i bytes for the entire gap and adds r_i of
delay at the gap's end; spill frees the same z_i bytes over the same
interval and adds z_i/beta of delay near the gap's start and
z_i/beta before its end. In the decoupled model all delays are
serial constants, so where they fall does not matter, and neither
action's charge appears in any other gap's charge. The two actions
also occupy identical pool space over identical intervals, so
exchanging one for the other preserves feasibility of every other
decision unchanged. Replacing the dearer action by the cheaper one
therefore lowers total cost with no side effects; applying this
exchange at every non-kept gap yields an optimal schedule in which
each non-kept gap of segment i pays min(r_i, 2 z_i / beta).

Proposition 3 (complexity of the decoupled problem).

(a) Reduction to offline caching. Build a caching instance: one item
per segment, item i of size z_i with fault price
q_i = min(r_i, 2 z_i / beta); the request sequence is a_1..a_m; the
cache capacity is K. Map schedules both ways by "segment i kept
across a gap if and only if item i stays in cache between the two
consecutive requests." By Lemma 2 an optimal retention schedule pays
q_i exactly on each non-kept gap, which is exactly the fault cost of
the corresponding caching schedule, so the two optima coincide.

(b) Equal sizes, an idealization. With z_i equal for all i, the
instance is offline weighted caching — equal-size items with
item-specific fault prices — which is solvable in polynomial time by
the classical minimum-cost flow formulation [1] (minimum-cost flow
is a network optimization problem solvable exactly in polynomial
time). With equal prices as well, the optimum is Belady's rule —
evict the item whose next request is farthest in the future [2] — so
the flow formulation is a generalization of Belady's algorithm. We
state this case as the idealization it is: it locates the complexity
boundary, and it does not describe our corpus, whose document
lengths run from roughly 50 to 1,000 tokens, so segment sizes vary
by more than an order of magnitude.

(c) Heterogeneous sizes. Offline caching with arbitrary item sizes
is NP-hard even in the fault model, where every fault price is
equal [3]. The hardness transfers to our problem by reduction from
that caching problem: given such an instance, set z_i to the item
sizes, set r_i to the common fault price, and choose beta small
enough that 2 z_i / beta >= r_i for every i, so the per-gap price of
part (a) is exactly r_i. By (a) the decoupled retention optimum then
equals the caching optimum, so deciding it is NP-hard.

What the engine does in practice. The heterogeneous problem is not
solved exactly in the engine. The engine's spill scheduler (the E9
arm) is a greedy price-per-byte rule: when the pool must shed bytes,
it sheds the segment whose per-gap price per byte,
min(r_i, 2 z_i / beta) / z_i, is smallest, spilling when
2 z_i / beta < r_i and dropping otherwise. Its gap to a lower bound
is measured, not proven: E9 runs the rule against a clairvoyant
reference — a schedule computed with all outcomes known in advance.

Open Problem 1 (bandwidth-coupled retention). In the engine, spill
transfers share bandwidth with the schedule's own weight and KV
reads and can overlap compute. The effective price of a spill then
depends on how much bandwidth slack the schedule has while the
transfer runs, which couples every gap decision to the global timing
of the plan. We state this variant — minimize makespan when
transfers may overlap compute and all traffic shares beta — as an
open problem. We claim neither an algorithm nor a hardness result
for it. The greedy rule above is a heuristic for this variant, and
E9 measures its distance from the clairvoyant reference.

## 7. Contributions

The engine. We build the five operations of Section 4 into a
production serving engine (vLLM [4]) and make the plan the sole
owner of query KV: admission by KV budget (a document enters only
when its KV fits the pool), pins for live documents, eager frees at
document death, and a strict mode in which any heuristic eviction
raises an error rather than silently substituting for the plan —
zero such evictions occurred across every validated run. Rewind
execution reaches 1.07 times the read estimate at the old anchor on
the 32B tier (32.0 seconds, compared with the 30-second estimate;
E1). Two protocol rules the measurements forced are part of the
contribution: the rewind boundary rule, which rewinds to the
document plus the questions' shared 33-token preamble rather than to
the document alone, and the decisive-token window of Algorithm 2,
line 14, for models that do not keep a one-token answer format.

The optimizer. We give a calibrated cost estimator (Algorithm 1) in
which every constant is a measured number attributed to a named
calibration run, a plan enumerator with an explicit legality filter
and feasibility gate (Algorithm 2), and a predicate-grouping dynamic
program (Algorithm 3) that is exact for the additive fluid cost by
Lemma 1 and Proposition 2, at O(n^2) block-cost evaluations,
cross-checked against exhaustive enumeration of all 2^(n-1)
partitions for every n <= 6 in the test suite. The estimator
predicts every measured cell on the reference hardware within about
five to ten percent, and the same machinery transfers to a new
device from two measured constants — its sustained token-reading
rate and its KV pool size — which E12 tests, including the
feasibility gate's verdict at the measured saturation width:
refusal or throttled plan, whichever the arithmetic says.

The theory. We formalize retention as the plan-aware KV retention
problem (Problem 1) over the segments of Definition 3, prove the
per-gap price decoupling (Lemma 2), reduce the decoupled variant to
offline caching with per-gap eviction prices and settle its
complexity — polynomial-time by minimum-cost flow at equal sizes,
generalizing Belady's algorithm, and NP-hard at heterogeneous sizes
(Proposition 3) — and we state the bandwidth-coupled variant as Open
Problem 1 without claiming an algorithm or a hardness result for it;
the engine's greedy heuristic for the coupled variant is evaluated
in E9.

The fusion negative, and the gate that caught it. We specified the
fuse operation of Section 4, built it on the production
shared-prefix kernel path, and closed it. The fidelity gate —
bit-identity of answers against unfused execution (Definition 2),
plus a probability-gap discriminator that separates confident flips
from borderline noise (a flip is confident when the unfused answer
token led its runner-up by more than 0.2 nats, natural-log units of
probability) — rejected every measured cell. The cascade kernel
flips answers confidently on both KV formats: 29, 16, and 43
confident flips per 300-answer cell at 300, 3,000, and 15,000
document tokens on fp8 KV, and 20 and 12 on the bf16 isolation arm,
whose third cell silently fell back to unfused execution, so its
zero flips proves nothing (E5). We report this as a contribution in
two parts. The two-format evidence chain is an upstream kernel bug
report: the flips survive a KV-format change, which points at the
kernel path, not at quantization noise. And the gate itself joins
the paper's fidelity method: a gate that catches a broken kernel
before any answer ships is the fidelity discipline working, not
failing.

The method. Every claim in the paper follows one discipline: state
the lower bound (Proposition 1), solve the rate program of the fluid
model, construct the schedule, measure the run, and check it with an
independent validator that replays every batch. The discipline is
applied on both model tiers and across GPU counts — measured at
one, two, four, and eight GPUs (E2) — and it is what turned two
silent accounting gaps (the rewind boundary rule and the pin
coverage of the shared question preamble) into stated rules rather
than lingering luck. The same discipline closed fusion (E5) and is
holding the shared-scan speedup at the door until its fidelity
check passes (E13).

## 8. Experiment plan

Roster status. E1 through E4 are measured, E5 is complete as a
negative result, and E13 is measured with its fidelity check
pending; E6 through E12 are planned, with expected outcomes stated
in advance. Every measured cell predates the re-baseline flight and
will be restated against its numbers (status block). The reference
query throughout: 10,000 IMDb reviews (2,966,000 document tokens),
four filters at selectivity 0.8, Qwen3-4B-FP8 on one H100, cold
start; at the old anchor its read estimate is 39.5 seconds and its
work estimate about 47 seconds. The 32B tier is Qwen3-32B-FP8 at
1,000 documents (323,541 tokens), where prefill — the forward pass
that reads prompt tokens and builds their KV — achieved 10,800
tokens per second and the read estimate is 30 seconds.

E1, control ladder (measured). Goal: show how much of the gap to
the lower bound each level of plan control recovers. Setup: the
reference query under five arms — naive with the task text before
the document (no prefix shared), naive with the document first
(prefix reuse possible), the client-side scheduling library, the
in-engine strict scheduler, and rewind execution — plus the same
ladder's endpoints on the 32B tier. Outcome: 122.8, 111.3, 52.5,
52.3, and 49.9 seconds against the 47-second work estimate, with
corpus reads falling from 3.03 times the corpus to 1.23; on the 32B
tier rewind execution takes 32.0 seconds at 1.14 corpus reads,
compared with 48.1 to 54.5 seconds for every other arm.

E2, GPU scaling (measured through eight GPUs). Goal: show that the
planner's sharding divides the single-GPU makespan with negligible
residue. Setup: the reference query sharded over 1, 2, 4, and 8
GPUs with token-balanced shards, each worker computing the same
deterministic plan. Outcome: 25.30 seconds on two GPUs (1.97 times
faster than the 49.9-second single-GPU run), 13.30 seconds on four
(3.75 times), and 6.66 seconds on eight (7.50 times, compared with
the perfect 8.00) — 94 percent efficiency at four and at eight
GPUs, where efficiency is the measured speedup divided by the GPU
count. Shards balance within 39 tokens of each other at every
width, and the eight worker walls span 6.40 to 6.66 seconds, so
nothing is shared and nothing straggles
(results/engine/multigpu2.json, multigpu4.json, multigpu8.json).

E3, tier reversals (measured). Goal: show that the same mechanism
flips between winning and losing across model tiers exactly as the
cost model prices it. Setup: the pin discipline and the persisted-KV
restore path, each run on both tiers. Outcome: pins match the
client library at 4B (49.2 versus 48.9 seconds) but lose under 32B
pool overflow (54.5 seconds at 1.25 corpus reads, against 48.1 for
an unpinned streaming client — fewer reads, more time); restore
loses about two to one at 4B (20.3 and 18.4 seconds against 10.5 of
recompute, at a measured 5.2 GB/s of disk bandwidth) and flips at
32B (about 8 seconds of restore against 30 of recompute).

E4, reasoning grids (analytical layer and hardware anchor
measured). Goal: locate where generation overtakes reading and
what that does to policy gaps. Setup, model: sweep the mean
thinking length G over 0, 32, 128, and 512 tokens through the
calibrated cost model, its value recurrences, and the exact solver
for small corpora, anchored at G = 0 to the measured grid within
about ten percent (48.7 predicted versus 52.5 measured seconds);
at G = 512 the model grows the reference query from 48.7 to 238
seconds. Setup, hardware: force the thinking length with a
forced-length instrument — 2,000 documents, four filters at
selectivity 0.8, four policies
(results/engine/reason_grid.json) — plus an admission-budget sweep
at G = 128 (results/engine/width_scaling.json). Outcome: the
predictions held. Walls in seconds at G = 0 / 32 / 128 / 512:
pipeline (gated, document KV held across stages) 10.9 / 32.6 /
79.7 / 275.3; lookahead-2 (the next two questions issued ungated)
20.2 / 49.4 / 101.2 / 328.5; full speculation (all four questions
at once) 22.2 / 58.5 / 135.0 / 461.3; waves (gated stage waves
that re-read surviving documents at each stage) 23.8 / 44.5 /
89.6 / 283.9. Three confirmations. The waves-to-pipeline gap
compresses from 2.2 times at G = 0 to 1.03 at G = 512: re-reading
stops mattering once decode dominates. Speculation loses at every
G on this corpus, which keeps the device saturated — full
speculation pays 186 seconds over the pipeline's 275.3 at
G = 512 — because a wasted branch now carries G thinking tokens.
And the interior lookahead never beats the pipeline here, exactly
as the model prices it: lookahead pays only on a starved machine,
and the starved half of the prediction stays with E8. The
admission sweep saturates near a 100,000-token budget: 95.1
seconds at a 50,000-token budget, 78.1 at 100,000, then flat
within about 0.2 seconds out to 700,000 — the pool never binds at
this scale.

Correction, 2026-08-05: the speculation and lookahead walls above
carry an instrument artifact. The client launched a document's
speculative questions simultaneously, and the prefix cache reuses
only committed blocks, so the branches each prefilled the document
(results/engine/reason_race.json: with the launch staggered, full
speculation at g=0 costs 10.15 seconds against the pipeline's
9.27, not the 2x this grid shows; without the stagger the race
reproduces, 19.51 seconds at 2.41 corpus reads). The pipeline and
waves rows are unaffected. The grid's result file was retracted
(deleted from results/; in git history) so the artifact cells
cannot be quoted; the grid re-fly with the fixed client re-banks
it. The g>0 speculation penalty (a wasted branch carries its
thinking trace) is real but smaller than the rows above state.

E5, fusion gate and crossover (complete; negative result). Goal as
stated in advance: measure the surface from which the planner
prices the fused-fork operator, and validate its correctness gate.
The expected outcome was that fusion's gain grows with document
length toward a dense-compute crossover near 24,000 tokens on the
4B model, and that the gate admits only cells with bit-identical
answers. Setup: six classifiers over 50 documents at 300, 3,000,
and 15,000 tokens, fused against unfused forked execution, on fp8
KV and again on bf16 KV (results/engine/fusegate.json,
fusegate_auto.json). Actual outcome: the gate admitted nothing. On
fp8 KV the cascade kernel flips 29, 16, and 43 answers per
300-answer cell past the 0.2-nat confidence gap (41, 23, and 52
flips before the gap filter). On bf16 KV it still flips 20 and 12
in the first two cells; the 15,000-token cell recorded zero flips,
but only 1 of its 47 batched steps actually ran the cascade
kernel — the other 46 silently fell back to unfused execution — so
the zero proves nothing. The speed case never appeared either: the
fused question pass took 7.8 seconds, compared with 0.3 unfused,
at 300 tokens, and 3.7 against 3.8 at 15,000. Fusion on this stack
is closed. The two-format evidence chain is the upstream kernel
bug report, and the gate's catch is the experiment's product: no
fused answer ever shipped.

E6, sharing ladder (planned; reframed fork-only after E5). Goal:
price fork. Setup: one all-answers workload run with forking
disabled (every branch re-reads its document) and with forking.
Expected outcome: a two-row table attributing the read savings to
fork. The fused third row is gone: E5 closed fusion on this stack.

E7, tagging at scale (planned; reframed fork-only after E5). Goal:
demonstrate the all-answers regime end to end at production width.
Setup: 40 predicates over long documents, executed as forked
branches. Expected outcome: makespan tracks the read estimate plus
the priced branch work, with no per-predicate re-reads.

E8, interactive boundary (planned). Goal: measure the corpus size
at which gating stops paying, and check the planner reproduces it.
Setup: N in {20, 100, 500, 2000}, rewind execution against
ask-everything (evaluate all predicates ungated). Expected outcome:
ungated evaluation wins at small N where the device is starved —
as the exact solver already shows at N = 3 — and gating wins at
large N, with the measured boundary matching the planner's.

E9, retention in practice (planned). Goal: test the Section 6
greedy rule where spilling should beat recomputing. Setup: a
length-variance workload with natural thinking lengths on the 32B
tier, comparing the planner's spill schedule, the engine's default
preemption (evict a running request's KV under pressure and
recompute it later), and the clairvoyant reference schedule.
Expected outcome: the planner lands near the clairvoyant reference
and ahead of preemption, and the gap to the reference bounds what
the open coupled variant could still recover.

E10, planner end to end (planned). Goal: show the optimizer's
choices are the right ones, not just defensible ones. Setup: the
planner drives every operator choice live on the E6 through E9
workloads, and each rejected alternative is also run under
identical conditions. Expected outcome: every emitted plan beats
the alternatives the planner rejected.

E11, wrong-prior robustness (planned). Goal: measure the cost of
bad selectivity estimates and the value of re-planning. Setup:
plans built from selectivities mis-set by controlled factors, with
and without online re-planning from observed pass rates. Expected
outcome: online re-planning recovers most of the gap to the
correctly informed plan.

E12, device transfer and refusal (planned; expected outcome
corrected 2026-08-04). Goal: show the cost model carries to a new
device from two measured constants, and that the feasibility gate
tells the truth in both directions. Setup: predict the 4B
configuration on an L40S from the L40S's measured token-reading
rate and KV pool size, then run it; submit the 32B-on-L40S
configuration to the planner; and measure the L40S's saturation
width W*. Expected outcome, prediction half: measurements land
within the model's established five-to-ten-percent envelope.
Refusal half, corrected: this draft used to expect line 8 to
refuse 32B-on-L40S, and the shipped gate's arithmetic says
otherwise. The 32B weights fit one L40S — 33.5 GB, compared with
the 44.2 GB provisioned at 0.92 of the card's 48 GB — leaving a
pool of 81,329 KV tokens, far above one working set of 1,668
tokens (the 1,000-document corpus's largest document, 1,622
tokens, plus one 46-token question). So at the shipped minimum
W* = 1 the planner emits a plan, throttled by admission: a
65,063-token budget (80 percent of the pool), pins off under
overflow (tests/test_planner.py, the corrected E12 test). The gate
refuses once the pool cannot hold W* working sets — past a
W* of about 48 resident documents, which is 81,329 pool
tokens over the 1,668-token working set. The H100 admission sweep
(E4) saturated near 100,000 tokens, about 300 mean-length
documents, so a comparable L40S width would refuse. E12 measures
W* on the L40S and reports the gate's verdict at the measured
value, either way. The card-count refusal of line 3 is exercised
directly: weights that need more cards than exist are refused by
name, not planned around.

E13, shared corpus scans (measured; fidelity check pending). Goal:
amortize one corpus read across q concurrent queries that share
document KV. Setup: 2,000 documents, four filters per query, q in
{1, 2, 4, 8} concurrent queries, shared execution against running
the q queries separately (results/engine/shared2000.json.gz).
Outcome, speed: shared execution takes 14.1 / 15.1 / 17.2 / 22.3
seconds at q = 1 / 2 / 4 / 8, compared with 14.1 / 28.0 / 56.3 /
113.7 separate — 1.00, 1.86, 3.27, and 5.09 times faster — and the
corpus-read multiplier at q = 8 falls from 8.82 (separate) to 1.42
(shared). Outcome, fidelity: unverified, and the speed claim is
gated on it. At q = 8 the shared and separate survivor sets
diverge in 6 of the 8 queries, and shared mode scores 123 more
wrong answers (6,519 wrong of 34,027 scored calls, compared with
6,396 of 34,251 separate). That gap is too large to wave off as
noise. The discriminator run — the E5 gate's 0.2-nat
probability-gap test applied to every disagreement — decides:
near-ties ship with a measured tolerance; confident flips mean a
routing bug to find before the 5.09 number appears anywhere.

## 9. Non-claims

We claim no new attention kernel. We claim no fused execution
operator: fusion was specified on existing shared-prefix primitives
(FlashInfer [5]), built, and closed by its own fidelity gate —
confident answer flips on both KV formats (E5; Section 7). What we
claim there is the gate and the documented negative, not the
operator. We claim no prompt rewriting: prompts are a fixed
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

## 10. Related work: positioning checklist

The question asked of each line of work: does it plan engine state
from query semantics against a stated lower bound?

- Sarathi-Serve [6]: engine scheduling (chunked prefill, stall-free
  batching) that is blind to query structure.
- Hydragen [7] and cascade attention primitives [5]: the fusion
  mechanism, with no planner deciding when it pays.
- SGLang RadixAttention [8] and Parrot [9]: reactive prefix reuse
  and inter-request hints discovered at run time, not planned KV
  lifetimes with declared retention.
- LOTUS [10], Palimpzest [11], DocETL [12]: semantic-operator
  optimizers that choose calls and models above a black-box serving
  engine, with no control of engine state.
- vAttention [13], CacheGen [14], CacheBlend [15]: KV transport,
  layout, and loading mechanisms — complements to spill and
  persist, with no query-level plan.

For each we verify the answer in a reading pass before submission.

## 11. References

[1] M. Chrobak, H. Karloff, T. Payne, S. Vishwanathan. New results
    on server problems. SIAM Journal on Discrete Mathematics 4(2),
    1991.
[2] L. A. Belady. A study of replacement algorithms for a
    virtual-storage computer. IBM Systems Journal 5(2), 1966.
[3] M. Chrobak, G. J. Woeginger, K. Makino, H. Xu. Caching is hard
    — even in the fault model. Algorithmica 63(4), 2012.
[4] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu,
    J. Gonzalez, H. Zhang, I. Stoica. Efficient memory management
    for large language model serving with PagedAttention. SOSP
    2023.
[5] Z. Ye et al. FlashInfer: efficient and customizable attention
    engine for LLM inference serving. MLSys 2025.
[6] A. Agrawal et al. Taming throughput-latency tradeoff in LLM
    inference with Sarathi-Serve. OSDI 2024.
[7] J. Juravsky, B. Brown, R. Ehrlich, D. Y. Fu, C. Re,
    A. Mirhoseini. Hydragen: high-throughput LLM inference with
    shared prefixes. 2024.
[8] L. Zheng et al. SGLang: efficient execution of structured
    language model programs. NeurIPS 2024.
[9] C. Lin et al. Parrot: efficient serving of LLM-based
    applications with semantic variable. OSDI 2024.
[10] L. Patel, S. Jha, C. Guestrin, M. Zaharia. Semantic operators:
    a declarative model for rich, AI-based analytics over text data
    (LOTUS). 2024.
[11] C. Liu et al. Palimpzest: optimizing AI-powered analytics with
    declarative query processing. CIDR 2025.
[12] S. Shankar, A. G. Parameswaran, E. Wu. DocETL: agentic query
    rewriting and evaluation for complex document processing. 2024.
[13] R. Prabhu et al. vAttention: dynamic memory management for
    serving LLMs without PagedAttention. ASPLOS 2025.
[14] Y. Liu et al. CacheGen: KV cache compression and streaming for
    fast large language model serving. SIGCOMM 2024.
[15] J. Yao et al. CacheBlend: fast large language model serving
    for RAG with cached knowledge fusion. EuroSys 2025.
