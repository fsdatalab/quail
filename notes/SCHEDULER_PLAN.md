# Plan: where the scheduler work stands and what comes next

This document replaces the earlier five step plan, most of which is now
executed. It records what the measurements established, the decisions
they force, and the next phases in priority order. Terms are defined in
notes/RESULTS.md, which also holds all numbers cited here.

## What is done and what it established

- The engine reads at about 80,000 tokens per second and that is the
  machine, not the software. Kernel benchmarks beneath the stack put
  the serving tax at ten percent or less. 80,000 is the calibration
  constant that turns the paper's formula into wall clock predictions.
- The per request overhead is 1.12 milliseconds, half ours (fixed with
  pre-tokenization and streaming, no engine changes) and 0.55
  milliseconds engine internal.
- At 10,000 documents (3.4 times the card's note capacity) naive
  execution silently re-reads the corpus until the document-first
  advantage vanishes. Execution order is the decisive variable.
- The streaming client library (docengine/runtime/engine_client.py)
  recovers it all: 2.3 to 2.7 times faster than naive use of the same
  engine, reads within a few percent of once per document, and lands
  at 0.96 to 1.02 times the calibrated prediction. The query runs at
  the machine's speed limit with zero engine modifications.
- Speculation's measured value was compensation for stage barriers.
  Under streaming it is neutral. Its remaining rationale at this
  operating point is filling the final tail.
- The LP's role at this operating point is certification, not control:
  with short documents and short residence times, greedy admission
  provably ties the optimum. Its decisions become real when memory is
  genuinely contested.
- Under recency based eviction, submission order within a batch is
  itself a scheduling decision (consumers of resident notes before
  producers of new ones). An in-engine scheduler would pin instead.

Decision that follows: modifying the engine's internals is deferred.
At the current operating point it can win 5 to 15 percent plus
robustness, and both frontier phases below will re-price it before we
commit.

## Phase A. Reasoning filters (requested; changes the design space)

Today's filters answer in one token, which is why reading dominates
everything. Letting a filter think first (the model writes out
reasoning, then answers) is expected to raise accuracy, and it changes
the physics enough that the whole policy analysis must be redone
rather than patched. The arithmetic that says so: with 2,500 documents
in flight and 300 thinking tokens per filter call, generation costs
roughly as much wall clock as reading the whole corpus once, so the
bottleneck moves from reading to writing. Worse, thinking occupies
note memory: 300 tokens across 2,500 in-flight documents is about
55 gigabytes, which alone exceeds the card's pool, so admission must
charge each document for its expected thinking, roughly halving
concurrency. And a wasted speculative branch now wastes an entire
reasoning trace, not 25 tokens, so the gate or speculate crossover
moves hard toward gating. None of the measured conclusions can be
assumed to survive this; they must be re-derived and re-measured.

Work items, in order.

1. A harder planted task. The current flags line is too easy to show
   any accuracy benefit from thinking. Plant indirect facts instead,
   for example a line stating "the screening was at home" with the
   filter asking whether the reviewer saw the film in a theater, so
   selectivity stays controlled by us while the question needs an
   actual inference step. Shorten the question templates at the same
   time (a five to ten percent read saving already identified).
2. Accuracy measurement. Thinking on against thinking off, same
   planted truth, on the harder task. This prices the accuracy side of
   the trade before any scheduling work.
3. Cost model extension. Add a generation phase to the instance and
   cost model: a thinking length distribution per filter, decode steps
   priced by weight reads plus per token note reads over the growing
   context, and note memory that grows during a call and is freed
   after it. Then re-derive the policy comparison analytically:
   task-first against document-first against lookahead, now as a
   function of thinking length. Publish the crossover map before
   burning GPU hours.
4. Client library update. Admission budget charges expected thinking
   tokens; free a document's thinking notes eagerly after its answer;
   re-evaluate the lookahead dial (expected default: off).
5. Measured grid. The 10,000 document experiment repeated with
   reasoning filters at two or three thinking budgets, against the
   extended model's predictions, with the same layered comparison.
6. Re-price fusion. With reasoning, every generated token re-reads the
   document's notes, so k filters sharing one pass over the notes
   (cascade attention, which exists as a kernel primitive in the
   ecosystem) may become first order. Measure the attention read share
   in the reasoning runs; if it is material, co-scheduling a
   document's filters plus cascade becomes the first concrete
   in-engine work item.

## Phase B. Long documents: where contested memory and the LP get real

Every remaining open question converges on the regime where a
document's notes are large relative to memory. Document length
variance only creates real decisions when a sacrifice must be made,
and then the LP's length weighted retain-or-re-read choices are
exactly the tool; at 300 token documents no sacrifice ever happens,
which is why greedy tied the optimum. At 30,000 to 300,000 token
documents (contracts, transcripts), the card holds a handful, chains
cannot always finish while resident, re-reading a discarded document
costs seconds, and admission, retention, and chunked reading become
genuine choices.

Work items: build a long document workload with the same planted
selectivity trick; solve the LP against greedy on the model first and
publish the predicted gap (a null result here would itself be worth
knowing before hardware); extend the client for partial document
admission (the paper's chunking machinery, until now theoretical,
becomes operational); then the hardware comparison, greedy client
against LP driven schedule. This is the step where the LP either earns
its place as a controller or is demoted, with a measurement either
way.

## Phase C. The engine skeleton, run in parallel with phase A

The in-engine work starts now rather than after phases A and B, for
three reasons. The project's thesis is that the engine should
understand queries, and the client library demonstrates that thesis
only by workaround, winning through accidental alignments (recency
luck, ordering races). Phase B's fair trial of the LP requires real
retention control, because the client cannot faithfully execute a
retention decision when eviction belongs to the engine's recency rule.
And learning the engine's internals is the longest lead item on the
board, so it should overlap the other phases, not follow them.

The skeleton has two parts. The engine's scheduler is officially
replaceable by a custom class, so admission, ordering, and later
co-scheduling move inside it. The memory manager is not replaceable,
so a small contained patch to its block pool adds two per request
hooks: pin and unpin, which exclude a document's notes from eviction
until released, and free now, which drops notes immediately for
failed documents and, later, thinking tokens.

The acceptance test is also the demonstration that guarantees beat
luck. First reproduce the client library's 10,000 document numbers
with pinning in place of ordering tricks, no regression allowed. Then
inject an adversarial co-tenant, a background stream of unrelated
requests hammering the cache, and show that the client library on the
stock engine degrades while the pinned scheduler holds its makespan.

## Phase D. Gated items

- Cascade fused evaluation of a document's filters. Trigger: phase A
  item 6. Known worthless for one token filters at any document
  length (about two milliseconds per question on a 100,000 token
  document against seconds of prefill).
- Sequence truncation, promoted to the top engineering candidate: one
  engine-level sequence per document that appends a question, reads
  the answer, truncates the question and answer notes back to the
  document prefix, and appends the next question. Prompts are
  unchanged and each question sees exactly the context it sees today;
  what changes is that a document costs one request instead of n, and
  the per-request software floor (about 0.55 milliseconds times
  24,000 calls in the four filter query, the bulk of the roughly five
  second residual above the work floor) collapses. Prompt fusion
  (asking all questions in one call) was considered and rejected: it
  computes the same document reads plus extra cross-question
  attention, changes what each question conditions on, and its only
  real edge over speculation with note reuse is the same request
  count reduction that truncation achieves without touching prompts.
- Multiple queries sharing one card: out of scope by product decision
  (each tenant runs one query on a dedicated GPU). Shared scans stay
  on the roadmap: queued queries over the same corpus can be merged
  so the corpus is read once for all of them, a throughput multiplier
  that needs no engine changes.
- 100,000 document demonstration run with the client (about ten
  minutes of GPU). Any time a headline is wanted.
- 32 billion parameter model cell. Fold into phase B if wanted; it
  shrinks the note budget several fold and slows reading about eight
  fold, so it also raises the generation share.

## Costs

Phase A: items 1 and 2 are about two days and tens of dollars of GPU
time; item 3 is two to four days of modeling; items 4 and 5 about a
week with two or three GPU hours; item 6 rides along free. Phase B is
roughly a week, dominated by the workload build and the client's
chunked admission. Phase C items are re-priced by A and B before any
commitment.
