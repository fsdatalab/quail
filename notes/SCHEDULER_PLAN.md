# Plan: teach the serving engine to understand our queries

This document proposes the next phase of the project: building a
scheduler for the model serving engine that understands the query it is
running. It is written to be read without prior context. It defines
every term it uses, states what our measurements showed, then lays out
the work in five steps, each with a cost, a deliverable, and a decision
gate.

Scope decisions reflected in this revision: we build on vLLM only and do
not evaluate other engines; we do not pursue keeping documents in memory
between queries, because at realistic corpus sizes that is impossible
and every query must start from scratch; and the experiments move from
2,000 documents to 10,000, which is deliberately more than the card can
hold at once.

## The terms this document uses

- A language model reads and writes text in units called tokens. A token
  is roughly a short word. Our test documents are about 330 tokens each.
- Reading and writing are different kinds of work. Reading (the technical
  term is prefill) means the model processes a prompt's tokens. Writing
  (decode) means it produces answer tokens one at a time. Each of our
  requests writes exactly one token, YES or NO, so almost all the work
  is reading.
- When the model reads text, it builds internal notes about that text,
  called the KV cache. Think of it as working memory. While a document's
  notes are on the card, the model can answer another question about
  that document without re-reading it. The notes are large, about 72
  kilobytes per token for our small model, so what to keep and what to
  discard is a real decision.
- The serving engine is the open source software that runs the model on
  the graphics card. We use vLLM. You hand it requests and it returns
  completions, packing many requests together to keep the card busy.
- The scheduler is the part of the engine that decides, at every moment,
  which requests to work on next and which notes to keep or discard.
  Today's scheduler is general purpose. It serves requests in arrival
  order and keeps whichever notes were used most recently. It knows
  nothing about our queries.
- Cold means the working memory holds nothing useful when a query
  starts, so the query must read every document from scratch. We call
  that first read-everything pass the cold first wave. At realistic
  corpus sizes every query is cold, so cold is the only regime this plan
  optimizes. (Our earlier measurements included warm runs, where a
  previous run had left notes in memory. They remain useful as a
  diagnostic, because they isolate the engine's per request overhead,
  but warmness is not something we can rely on at scale.)
- Our queries are chains of filters. Each filter asks a yes or no
  question about each document, and a document that fails one filter
  skips the rest. The pass rate of a filter is the fraction of documents
  that pass it. The scheduling choices are when to ask each filter's
  question (wait for each answer, or ask ahead and accept waste when a
  document fails) and in what order to move documents through the
  filters.
- The ideal calculator is our cost formula from the paper. It prices
  only the unavoidable arithmetic at the card's maximum speed. No real
  system reaches it. Its value is as a fixed floor for measuring how far
  from perfect any real run is.

## What the measurements showed, and what changes at 10,000 documents

We rented one H100 card by the hour and ran 33 controlled query runs
over 2,000 movie reviews with the correct answers planted inside the
documents, so pass rates are under our control. The findings that drive
this plan:

1. Reading each document once and asking the filters on top of the kept
   notes beat re-reading documents for every filter in every
   configuration, by 1.2 to 2.6 times.
2. The choice between waiting and asking ahead followed the pass rate
   the way our theory predicts.
3. Every run landed 3.3 to 4.1 times above the ideal calculator. The
   engine read 58,000 to 78,000 tokens per second against a theoretical
   card maximum of about 275,000.
4. Even with nothing to read, the engine spent 1.0 to 1.4 milliseconds
   of wall clock per request on software overhead. At four filters that
   is 5 to 6 seconds wrapped around roughly 0.3 seconds of model work.

The 2,000 document setting had one property that made life artificially
easy: all 2,000 documents' notes fit on the card at once, about 45 of
the roughly 60 gigabytes available for notes. Nothing ever had to be
evicted, so almost any execution order got the full benefit of keeping
notes.

At 10,000 documents that is no longer true. The notes for the whole
corpus would need about 230 gigabytes, and the card holds about a
quarter of that, roughly 2,700 documents' worth at a time. Now execution
order decides everything. If we naively send all 10,000 first-stage
questions, then all the second-stage questions, the notes of the early
documents are long gone by the time their second question arrives, and
the engine silently re-reads them. The measured advantage of keeping
notes should collapse toward the re-read-everything cost. The fix is to
move documents through the query in blocks sized to the card: admit
about 2,500 documents, run them through all the filters while their
notes are hot, discard, admit the next block. Our analytical schedule
builders already produce exactly these blocked schedules; at 2,000
documents they made no measurable difference because nothing needed
blocking, and at 10,000 they should be the difference between keeping
the 2.6 times advantage and losing it. That claim is testable and is
the centerpiece experiment of this plan.

## Where we can win, corrected

An earlier version of this plan claimed the cold first wave cannot be
sped up. That was too strong. The honest split has three parts.

- Truly fixed: the arithmetic efficiency of the math kernels on this
  model at these document lengths. Whatever tokens per second the
  kernels can deliver on a well fed card is a floor a scheduler cannot
  move. We do not yet know that number; today we only know the ceiling
  (275,000) and today's achieved rate (58,000 to 78,000).
- Recoverable during the first read: everything between today's rate
  and the kernel floor. Candidates include how many tokens the engine
  packs into each internal step, how many requests it keeps in flight,
  per step bookkeeping, and the cost of converting text to tokens,
  which we currently pay repeatedly for the same document. Measuring how
  much of the 3.7 times gap this explains is step one.
- Recoverable after the first read: the per request software overhead
  (measured directly by the warm diagnostic runs), the waiting caused by
  our batch-and-wait rounds, wasted ask-ahead work, and, at 10,000
  documents, the re-reading caused by naive execution order.

## The plan, in five steps

### Step 1. Measure the engine's true speed limit and the overhead split

Two micro-measurements, no engine changes, on the existing harness.

First, the reading speed limit. Feed the engine one giant reading-only
job with pre-converted token numbers and sweep its batching settings
(tokens per internal step, concurrent requests). The best rate we
observe is the empirical kernel floor for this model on this card. This
tells us exactly how much of the 3.7 times gap is recoverable at all,
and how much of it the engine's default settings were leaving on the
table.

Second, the overhead split. Re-run a small grid with two client fixes:
hand the engine pre-converted token numbers instead of raw text, and
stream answers so each document advances the moment its own answer
arrives instead of waiting for the whole round. Whatever per request
overhead survives is genuinely internal to the engine and is the prize
for step 4.

About two to three days of work and under ten dollars of GPU time.

### Step 2. The 10,000 document experiment: show that order decides

Scale the grid to 10,000 documents, all cold. Run each configuration two
ways: the naive order (every document through stage one, then every
survivor through stage two, and so on) and the blocked order produced by
our analytical schedule builders, driven through the engine batch by
batch exactly as we already do in manifest mode. Record the engine's own
counters for how many tokens were re-read.

Expected result, stated as a falsifiable prediction: naive
document-first execution loses most of its advantage over
re-read-everything, and the blocked schedule keeps it. If the prediction
holds, this is the core demonstration that query-aware scheduling is
necessary at scale, not just helpful. If it fails, the engine's memory
management is better than we think and the plan shrinks. Runtime per
full pass over the corpus is about 45 seconds of pure reading, so a
twenty-run grid fits in roughly two GPU hours, under ten dollars.

### Step 3. Build the blocked, streaming scheduler as a client library

Combine what steps 1 and 2 validated into a small library that anyone
can use in front of an unmodified engine: it takes the filters, their
pass rates, and a document list; sizes blocks to the card's memory;
admits a block, streams each document through all its filters while its
notes are hot, frees failed documents' slots by letting them finish,
admits the next block as capacity frees; and asks ahead only when the
card would otherwise sit idle, with the pass rates setting the dial.
This is the practical near-term deliverable, useful on day one without
touching the engine. Success criteria: at 10,000 documents, beat naive
execution in every cell, and land within the step 1 kernel floor's
implied budget rather than today's 3.7 times. Roughly one to two weeks.

### Step 4. Move the scheduler inside the engine

vLLM allows a replacement scheduler to be plugged in as a class, without
maintaining a fork. Inside the engine we can do what the client library
cannot: advance a document to its next filter in the same internal step
its answer token is produced; pin the current block's live documents so
the recency rule cannot evict what the plan knows is needed; free a
document's notes the instant it fails a filter, so the next block
starts sooner; and prototype running each document as one ongoing
conversation that appends the next question after each answer, so later
filters stop paying the per request cost entirely. That last change
fights the engine's one-request-one-answer design the most, so it stays
on a branch until proven. Two to four weeks depending on how much of
this survives contact with the engine's internals.

### Step 5. Put the optimizer's numbers in charge

At 10,000 documents the interesting decisions (how many documents to
admit, what to keep versus re-read, how far ahead to ask) are exactly
the quantities our paper's optimizer computes best values for. The final
step wires those computed rates into the step 4 scheduler as its
admission and retention policy and measures how close the result gets to
the calculator's floor. One hero run at 100,000 documents makes the
headline demonstration. This closes the loop the paper opens: plan a
query on a formula, execute it on real hardware, and land where the
formula said you would.

## Costs and risks

GPU rental stays trivial. Steps 1 and 2 are days of work and tens of
dollars. Step 3 is one to two weeks. Steps 4 and 5 are two to four
weeks. The engineering risks: the engine's internals change quickly
between versions, so we pin the version we build against (0.26) and
stay behind its official plug-in seam where one exists; and the
conversation-per-document design may be rejected by the engine's
architecture, which is why it is staged last. The step 2 prediction
could also simply be wrong, which would be cheap to learn and would
redirect effort toward steps 1 and 4.

## What we can show when this is done

One demo, same corpus, same filters, all cold, side by side: the engine
used the way most people use it against the informed scheduler. At
10,000 documents the naive run re-reads most of the corpus at every
filter and the informed run does not, every filter stage after the first
read costs close to its real arithmetic, and the measured times sit
within the step 1 kernel floor's distance of the paper's calculator.
The claim the project rests on is that these systems are predictable
enough to plan for, and this demo is that claim running on hardware.
