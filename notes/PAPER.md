# Paper skeleton: formal statement, contributions, experiment plan

## Problem statement

A semantic scan query applies n natural-language predicates
(filters: gated; classifiers: all answers needed; join predicates:
over pairs) to a corpus of N documents using an LLM, on a device
with measured compute rate, bandwidth, and KV pool. Prompts are a
fixed interface. Minimize makespan subject to answer fidelity:
executions must produce token-identical answers to the naive
one-call-per-predicate reference.

## The contract and its floor

Under the contract (every document token processed once, by this
model, at this precision, cold), the floor is arithmetic:
T >= max(2 P C / R_flops, (W + kappa C) / BW) for corpus tokens C.
Every relaxation of a contract clause (persist: not cold; cascade:
not this model; span reading: not every token) yields its own floor
by the same arithmetic. All claims are stated against the contract
in force.

## Thesis

Attention state (KV) should be a first-class, plan-governed object.
Five verbs - fork, rewind, fuse, spill, persist - generate the
physical operator space for semantic scans, and an optimizer with
measured constants chooses among them per query. Two execution
primitives cover the workload family: rewind executes gated chains,
forked (optionally fused) branches execute everything-needed sets;
filters, classifiers, and joins are nestings of the two, indexed by
(selectivity vector, document length, thinking budget).

## Contributions

1. The state algebra and engine: plan-governed KV lifetimes inside
   a production engine (rewind execution at 1.07x the contract
   floor; zero heuristic evictions across every run; the suffix
   protocol; the decisive-token gate).
2. The optimizer: calibrated cost model (every constant a measured
   number with a named source flight); O(n^2) partition DP for
   composition, exact under the fluid model; feasibility gating;
   device transfer from two constants.
3. Theorem candidate: plan-aware spill scheduling - given the
   plan's access sequence, choose evict-recompute vs spill-restore
   per victim under shared bandwidth to minimize makespan. Formal
   statement, hardness or exact algorithm, engine implementation.
4. The fused-fork operator for everything-needed sets, correctness
   gated (bit-identical answers), with the measured crossover
   surface the planner prices it from.
5. The floor discipline itself: bound -> program -> constructed
   schedule -> measured run, validator-checked, on both model tiers
   and across 1-8 GPUs.

## Experiment plan

Banked (rerun only if the refactor changes numbers):
E1 ladder: naive -> client -> in-engine -> chain, 4B and 32B.
E2 scaling: 1/2/4/8 GPUs, balanced shards, 94 percent at 8.
E3 reversals: pins and restore flip between tiers.
E4 reasoning grids: 16 cells x 2 tiers; gap compression; width W*.

To fly:
E5 fusion gate + crossover: 6 classifiers x {0.3k, 3k, 15k} doc
   lengths x fused/unfused/sequential, thinking on; the planner's
   fusion surface. (Gate script in progress on a side thread.)
E6 no-sharing baseline: fork disabled vs fork vs fork+fuse - the
   three-rung table that prices each verb.
E7 tagging at scale: 40 predicates x long documents, the
   everything-needed regime end to end.
E8 interactive boundary: N in {20, 100, 500, 2000}, rewind vs
   ask-everything - the measured mode boundary the planner must
   reproduce.
E9 spill theorem validation: length-variance workload (natural
   thinking lengths), planner spill schedule vs vLLM preemption vs
   oracle, 32B where spill wins.
E10 planner end-to-end: plan_query drives every choice live on
   E5-E9 workloads; each emitted plan beats the alternatives it
   rejected.
E11 wrong-prior robustness: selectivity misestimates, online
   re-plan recovery.
E12 device transfer: 4B on L40S from constants alone (32B on L40S
   rejected by the feasibility gate - shown as a refusal, not a
   run).

## Non-claims

No new attention kernel (FlashInfer primitives, used correctly).
No prompt rewriting. Adaptive mid-query recomposition parked with
its measured trigger. Accuracy of the models themselves is
reported, not claimed.

## Related-work positioning to verify (reading pass)

Sarathi-Serve (workload-blind engine scheduling); Hydragen /
cascade primitives (mechanism without a planner); SGLang
RadixAttention, Parrot (reactive reuse, not planned lifetimes);
LOTUS, Palimpzest, DocETL (call-granularity optimizers over a
black-box engine); vAttention, CacheGen/CacheBlend (KV transport
and layout). Question for each: does it plan engine state from
query semantics against a stated floor?
