# Plan for solving the schedules

> Historical analytical plan. The current runtime boundary and physical plans
> are in [PHYSICAL_PLANS.md](PHYSICAL_PLANS.md).

## What the project is

The file paper.md in this repo is a working paper about a database query
that runs a chain of yes or no language model filters over a table of
documents. A document that fails one filter skips the rest, so the amount of
model work depends on how many documents pass each stage. All model calls
run on one graphics card, and the goal is to finish every required filter
decision as fast as possible. The paper fixes the filter order, defines a
cost model built from the card's compute speed, memory bandwidth, and memory
capacity, and compares three ways to schedule the work:

- **Task-first** processes every document under filter 1, then reprocesses
  the surviving documents under filter 2, and so on. The card rereads each
  surviving document at every stage, but the short filter prompt is shared
  across all documents at a stage.
- **Pipeline**, which the paper also calls document-first, reads each
  document once, keeps the model's stored memory of the document (the KV
  cache) on the card, and runs each filter as a prompt of about 50 tokens on
  top of that stored memory, waiting for each answer before running the next
  prompt.
- **Speculation** runs one or more future filter prompts on a document
  without waiting for the earlier answers. The later prompts are wasted
  whenever an early filter fails. A lookahead of k means k prompts run at
  once; a lookahead of 1 is the pipeline, and a lookahead equal to the
  number of filters is full speculation.

The paper asks for solvers that find the best schedule under its cost model,
lower bounds that no schedule can beat, and an experiment program on two
models (Qwen3-4B-FP8 and Qwen3-32B-FP8) and two cards (NVIDIA H100 and
L40S). The plan below covers all of that with no runs on real hardware.
Hardware measurement is a separate later phase.

## The data

The workload is real. The script scripts/build_workload.py samples 10,000
reviews without replacement from the 50,000 labeled reviews of the standard
IMDb movie review dataset, using seed 20260731, and counts each review's
tokens with the Qwen3 tokenizer. A token is a piece of text the model reads,
usually a word or part of a word. Both released model checkpoints ship
tokenizer files that are identical byte for byte, so one length list serves
both models. The result is workloads/documents.parquet.

The 10,000 sampled reviews contain 2,966,000 document tokens in total. The
median review is 223 tokens, the average is 297, the 99th percentile is
1,142, and the longest is 2,924. Review length in tokens is about 1.28 times
the length in words.

Memory capacity shapes the whole problem. Keeping a document's stored state
on the card costs 72 kilobytes per token for the 4B model and 128 for the
32B model. After subtracting the model weights and a reserve, the card can
hold the stored state of only part of the corpus at once: about 34 percent
of it for the 4B model on the H100, 19 percent on the L40S, 12 percent for
the 32B model on the H100, and 3.3 percent on the L40S. So the schedulers
face the tradeoff the paper is about, which is whether to keep a document's
stored state, delete it and recompute it later, or avoid the question by
speculating.

## Decisions that pin down open points

The review in notes/REVIEW.md found places where the paper has an error or
where two implementers could read a definition differently. The solver
adopts the following conventions, labeled C1 to C12, and records them so any
result can be traced to them.

- **C1, attention width.** Attention compute is counted with the width
  n_q·d_h (number of query heads times width per head), not the hidden
  width h that the paper's equation 20 uses. For Qwen3 the two differ by 1.6
  times. See finding 1 in the review.
- **C2, gate timing.** No batch may contain work that needs a filter answer
  still unresolved when the batch starts, and the rule applies even to the
  offline solver that knows all outcomes in advance. Section 3.4 of the
  paper states the rule; knowing the future helps only with packing,
  retention, and deletion choices.
- **C3, what counts as written and stored.** The revised paper's
  write-through rule is adopted exactly: every new token whose stored state
  a later token or branch reads counts as written, and only a final
  position with no descendant is skipped. So document tokens are written
  (except the decision position of a task-first call), the first 49 tokens
  of every 50 token filter prompt are written, and shared prompt blocks are
  written. During a batch, all of the batch's new tokens count toward peak
  memory. Reads are counted only for blocks already on the card when the
  batch started, and a block shared by several operations in one batch is
  counted once.
- **C4, algorithms.** The offline solver is Dijkstra's algorithm over
  scheduler states, which stays correct when deletion and recomputation
  create loops. The online case, where answers arrive only as batches
  finish, is solved as a stochastic shortest path problem (a shortest path
  problem in which each move has random outcomes) by value iteration.
- **C5, weight sizes.** The per batch weight traffic is the byte size of the
  repeated transformer blocks in the released checkpoint, and the memory
  footprint is the full loaded checkpoint. Until measured from the released
  files, the solver uses estimates: 3.6 and 4.5 gigabytes for the 4B model,
  and 31.2 and 33.5 for the 32B model.
- **C6, reserve.** The ideal model runs with a 2 gigabyte memory reserve for
  the runtime, recorded as such.
- **C7, attention speed ceiling.** The primary runs set the attention speed
  ceiling equal to the dense compute ceiling, which is optimistic and
  labeled. A sensitivity run uses half of it.
- **C8, minimum batch count for the lower bound.** Every new document
  token's stored state must sit on the card during its batch, so the number
  of batches is at least the total document tokens divided by the card's
  free capacity, rounded up. If a token cap per batch is configured, the
  bound also uses it.
- **C9, chunk quantum.** A document may be split into chunks. Exact runs on
  small instances allow any split (quantum 1). The 10,000 document runs
  restrict chunk boundaries to multiples of 256, which can only make the
  reported schedule slower than the unrestricted best, and is recorded.
- **C10, workload details.** The document pool is the 50,000 labeled
  reviews, with a document id of the form split/row. 418 reviews are exact
  duplicates of another review and are kept as separate rows; the scheduler
  does not share stored state between identical texts. Every filter prompt
  is exactly 50 tokens. Stored state uses 1 byte per element, with 2 bytes
  as a sensitivity case.
- **C11, startup.** There is no separate startup term. Shared prompt blocks
  are loaded inside the first batch that needs them and are never counted
  twice.
- **C12, what is being estimated.** The online value is the expected finish
  time of the best policy that never uses an unrevealed answer. Offline
  values are averaged over random outcome tables, and the same outcome
  table is reused across policies so comparisons are fair.

## How the code is organized

```
docengine/
  configs.py             model and device numbers
  instance.py            a problem instance: lengths, prompts, pass rates, limits
  costmodel.py           batch statistics and the ideal batch cost
  lb.py                  certified lower ledgers and the resource bound
  manifest.py            writes a schedule out as one record per batch
  optimizer/
    state_actions.py     length types, cache states, and batch actions per policy
    steady_state_lp.py   the expected-flow linear program and its residuals
    queue_augmented.py   the exact bounded-queue program for small checks
  runtime/replay.py      converts program rates into real validated batches
  reference/             the exact small-instance solvers (validation role)
    engine.py            states, legal batches, transitions
    offline.py           Dijkstra over states, returns the best schedule
    online.py            value iteration for the online case
  sched/blockwise.py     direct schedule builders (also the replay fallback)
  validator/check.py     independent checker that replays a schedule
experiments/
  run_lp.py              program plus replay on the 10,000 document runs
  run_n10k.py            the direct-builder 10,000 document runs
  run_smallN.py          exact runs on small instances with real hardware numbers
scripts/build_workload.py
tests/                   19 checks
```

The checker in validator/check.py deliberately shares no cost code with the
solver. It replays each schedule record, recomputes every quantity from the
paper's formulas on its own, checks memory limits and answer timing, and
reports errors per batch. The paper's appendix B requires the separation.

## The exact optimizer

A scheduler state records, per document, the next filter whose answer is
still needed and how many of the document's tokens have stored state on the
card, plus which shared prompt blocks are loaded. An action is one batch
(document chunks, filter prompts, and prompt block loads) followed by a
deletion choice. The offline solver runs Dijkstra's algorithm over these
states and rebuilds the best schedule from the visited states. The online
solver first lists every state reachable from the start, then runs value
iteration until the values stop changing.

The tests check the solver against hand worked answers and against the
paper's own claims. On instances small enough to enumerate by hand, the
solver returns the hand computed schedule. Allowing chunks never makes the
best schedule slower, and a case built like the paper's section 6.6 example
shows a strict win. Forced full speculation costs the same for every
outcome table, and its online value equals its offline value. The average
offline value never exceeds the online value, checked over all 16 outcome
tables of a two document, two filter instance. When memory is too small to
keep both documents, the solver deletes and recomputes, and the checker
accepts the result. The checker rejects schedules with a corrupted count or
with a prompt that uses an answer not yet revealed.

Exact solving is limited to small instances, which matches the paper's own
hardness claim. With single token chunks and deletion choices, the offline
solver handles 4 documents of about 7 tokens in roughly 146 seconds, and
the online solver handles 3 documents. With whole document chunks at real
document lengths, 4 documents solve in seconds. The boundary is recorded
rather than hidden.

## Schedules for 10,000 documents

Exact search cannot reach 10,000 documents, so the plan follows the paper:
build a good feasible schedule, compute the lower bound, and report both
with the gap. Two builders in sched/blockwise.py cover the three policies.
The task-first builder processes documents in waves, one wave per filter,
packing each batch up to the card's free memory and splitting the document
that straddles a batch boundary. The blockwise builder covers pipeline and
speculation with a lookahead setting. It reads new documents up to capacity,
runs their first block of filter prompts in the same batch, keeps the
survivors' stored state for exactly one batch, and runs their next prompts
in the following batch alongside the next group of documents. Failed
documents are deleted at the batch boundary. If high pass rates leave no
room to make progress, the builder deletes part of the backlog and
recomputes those documents later, which is the paper's deletion and
recomputation path, not a failure.

Both builders only use answers that earlier batches revealed, so a real
scheduler could follow them without knowing the future. Their schedules are
therefore valid upper bounds for the offline and the online problem at
once.

Planned but not yet built: a local search pass that merges, splits, and
moves work between batches to close any remaining gap, and a builder that
holds documents back the way the exact optimizer does (see notes/RESULTS.md
on why the current builders lose 12 to 24 percent on small instances).

## Order of work and status

1. Cost model, lower bound, manifest, checker, tests. Done.
2. Exact offline and online solvers, with the verification battery. Done.
3. The 10,000 document runs on all four model and device pairs, pass rates
   0.1 to 0.9, with coupled outcome tables. Done; results and gaps are in
   notes/RESULTS.md and results/.
4. Small instance study on real hardware numbers, and the four filter
   lookahead study. Done.
5. The revised paper's program layer: state and action generators per
   policy, the expected-flow linear program, the exact bounded-queue
   program for small checks, and the replay runtime with fill, core,
   repair, and drain phases. Done; results in notes/RESULTS.md.
6. Remaining: measure the true weight bytes from the released checkpoints,
   run the pass rate grid at finer steps with more repetitions, teach the
   replay and builders to hold documents back, extend the pipeline program
   past two filters, produce the paper's plots, and only then calibrate
   against real hardware.
