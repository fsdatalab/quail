# Plan: teach the serving engine to understand our queries

This document proposes the next phase of the project: building a
scheduler for the model serving engine that understands the query it is
running. It is written to be read without any prior context. It first
defines every term it uses, then states what our measurements showed,
then lays out the work in five steps, each with a cost, a deliverable,
and a decision gate.

## The terms this document uses

- A language model reads and writes text in units called tokens. A token
  is roughly a short word. Our test documents are about 300 tokens each.
- Reading and writing are different kinds of work. Reading (the technical
  term is prefill) means the model processes a prompt's tokens to
  understand them. Writing (the technical term is decode) means the model
  produces answer tokens one at a time. In our workload each request
  writes exactly one token, YES or NO, so almost all the work is reading.
- When the model reads text, it builds internal notes about that text,
  called the KV cache. Think of it as the model's working memory. If
  those notes are kept on the graphics card, the model can answer a new
  question about the same text without re-reading it. The notes are
  large, about 72 kilobytes per token for our small model, so keeping
  them is a real storage decision, not a free trick.
- The serving engine is the open source software that runs the model on
  the graphics card. We use one called vLLM. You hand it requests, each
  being a prompt, and it returns completions. Inside, it packs many
  requests together to keep the card busy.
- The scheduler is the part of the engine that decides, at every moment,
  which requests to work on next and which working memory to keep or
  discard. Today's scheduler is general purpose. It serves requests in
  arrival order and keeps whatever memory was used most recently. It
  knows nothing about our queries.
- Cold and warm describe the state of that working memory when a query
  starts. Cold means the memory holds nothing useful, so the first thing
  the query must do is read every document from scratch. We call that
  first pass the cold first wave, and it is the single most expensive
  part of a query. Warm means an earlier query over the same documents
  already left their notes in memory, so reading is skipped entirely.
- Our queries are chains of filters. Each filter asks a yes or no
  question about each document, and a document that fails one filter
  skips the rest. The pass rate of a filter is the fraction of documents
  that pass it. The scheduling question is when to ask each filter: wait
  for each answer before asking the next question (the pipeline), ask
  several questions ahead without waiting (speculation, where questions
  asked about a document that then fails are wasted), or re-read the
  surviving documents for every filter (the naive way most people use
  these engines today).
- The ideal calculator is our cost formula from the paper. It prices
  only the unavoidable arithmetic, at the card's maximum speed. No real
  system can reach it. Its value is as a fixed floor, so we can say how
  far from perfect any real run is.

## What the measurements showed

We rented one H100 graphics card by the hour and ran a controlled grid
of 33 query runs over 2,000 movie reviews, with the correct answers
planted inside the documents so we control the pass rates exactly. Five
findings matter for this plan.

1. Reading each document once and asking the filters on top of the kept
   notes beats re-reading documents for every filter, in every
   configuration, by 1.2 to 2.6 times. The advantage grows with more
   filters and higher pass rates.
2. The choice between waiting and asking ahead follows the pass rate the
   way our theory says it should. Waiting wins when filters reject most
   documents. Asking ahead wins when most documents survive.
3. Every run lands 3.3 to 4.1 times slower than the ideal calculator,
   and one number explains almost all of it. The engine reads about
   58,000 to 78,000 tokens per second on our short documents, against a
   theoretical card maximum of about 275,000. That gap lives in the math
   kernels, not in scheduling, and no scheduler will close it.
4. The warm runs are the striking ones. When a query runs over documents
   whose notes are already in memory, the pipeline finishes in 3.3
   seconds instead of 10.2 at two filters, and 4.7 instead of 12.4 at
   four. Warm queries get within 1.2 to 1.5 times of the ideal floor. A
   document store answers many queries over the same corpus, so this is
   the regime that matters commercially, and today it happens only by
   accident.
5. Even when there is nothing left to read, the engine spends about 1.0
   to 1.4 milliseconds of wall clock per request on software overhead,
   things like converting text to tokens, bookkeeping, and picking the
   next answer token. At four filters and a high pass rate, the filter
   stages after the first read involve about 4,400 requests, which is 5
   to 6 seconds of overhead wrapped around roughly 0.3 seconds of
   actual model work.

Two more facts set up the plan. First, today's engine threw away the
entire corpus memory in one of our runs because its keep-the-most-recent
rule had no idea the corpus would be needed again seconds later. Second,
our harness currently moves documents through filters in rounds: send a
batch of questions, wait for every answer, send the next batch. The
fastest document always waits for the slowest, and that waiting is our
code's fault, not the engine's.

## What a better scheduler can and cannot win

It cannot speed up the cold first wave. Reading two thousand documents
for the first time runs at whatever speed the math kernels deliver.

It can win almost everything after that: the 5 to 6 seconds of software
overhead in later filter stages, the waiting caused by our
batch-and-wait rounds, the wasted work from asking ahead when waiting
would have been fine, and, largest of all, the difference between cold
and warm across repeated queries, worth about 3 times, by keeping the
corpus notes in memory on purpose instead of by luck.

## The plan, in five steps

### Step 1. Find out exactly where the 1.4 milliseconds goes

Before building anything, we change our own test harness in two ways
that require no engine changes, and measure what each is worth. First,
hand the engine pre-converted token numbers instead of raw text, because
today the engine re-converts the same document text at every filter
stage. Second, replace the batch-and-wait rounds with streaming, where
each document advances to its next filter the instant its own answer
arrives. The engine already offers a streaming interface.

The deliverable is a table splitting the per request overhead into our
share and the engine's share. The decision gate: whatever overhead
survives both fixes is the honest size of the prize for a custom
scheduler. If the floor collapses to a fraction of a millisecond, the
scheduler case rests mostly on memory control (step 4) and we can skip
parts of step 3. About two days of work and under ten dollars of GPU
time.

### Step 2. Run the same grid on the competing engine

There is a second open source engine, SGLang, whose working memory is
organized as a tree of shared prefixes. Our workload is exactly tree
shaped, one document trunk with several question branches, so some of
what we plan to build may come for free there. We run the identical grid
and compare. The decision gate: build on vLLM or on SGLang, whichever
starts closer to where we want to end. About two days and under ten
dollars.

### Step 3. Prototype a scheduler that knows the query plan

vLLM allows a replacement scheduler to be plugged in as a class, without
maintaining a private fork of the whole project. We build one that is
told the query plan: the filters, their order, and their pass rates.

It does two things differently. It advances each document the moment
that document's answer token is produced, so nothing waits for
stragglers. And it treats asking ahead as a dial rather than a policy:
it asks the next question early only when the card would otherwise sit
idle, using the known pass rates to decide how much asking ahead is
worth the waste. Success criteria: recover at least half of the
later-stage overhead measured in step 1, and make the informed scheduler
match or beat both fixed policies in every cell of the grid. Roughly two
to three weeks of engineering.

### Step 4. Control the working memory deliberately

This is the deeper change, touching how the engine keeps and discards
notes. Three behaviors, in order of increasing difficulty. Keep the
notes of documents that are still alive in the query, because they will
be needed at the next filter. Discard the notes of a document the moment
it fails a filter, because they are dead weight. Protect the corpus
notes across queries, so the second query over the same corpus starts
warm by design, which our measurements price at about 3 times.

A fourth behavior is worth prototyping here: run each document as one
ongoing conversation instead of a fresh request per filter, appending
the next question to the same conversation after reading the answer.
Later filters then stop paying the per request cost at all. This fights
the engine's one-request-one-answer design more than anything else in
this plan, so it stays on a branch until proven. Roughly three to four
weeks.

### Step 5. Put the optimizer in charge when memory is scarce

Everything above runs in a regime where all the documents' notes fit in
the card's memory at once. With a bigger model or more documents they do
not fit, and then real decisions appear: how many documents to admit at
a time, and which notes to keep versus re-read later. Our paper's
optimizer computes the best rates for exactly these decisions, and this
scheduler is the first place those computed rates can be enforced on
real hardware. The experiment: the 32 billion parameter model, or ten
times the documents, where the naive engine degrades and the informed
scheduler should not. This step is the research payoff and the paper's
closing demonstration.

## Costs and risks

GPU rental is noise. The card costs a few dollars per hour and no
experiment here needs more than two hours. The real cost is engineering
time: steps 1 and 2 are days, steps 3 and 4 are two to four weeks each.

Risks. The engine's internals change quickly between versions, so we pin
the version we build against (0.26) and keep our changes behind its
official plug-in seam where one exists. The conversation-per-document
design in step 4 is the most invasive and could be rejected by the
engine's architecture; it is staged last among the engine changes for
that reason. Separately, our tiny test questions are answered correctly
82 to 99.5 percent of the time depending on wording, which is an
accuracy matter, not a scheduling one, and can be improved independently
at any time.

## What we can show when this is done

One demo, same corpus, same filters, side by side. Today's engine used
the way most people use it, against the informed scheduler. The first
query is somewhat faster. Every filter stage after the first read is
several times faster. The second query over the same corpus is about
three times faster, because the scheduler kept the corpus in memory on
purpose. And the measured times sit within about 15 percent of what the
paper's calculator predicts, which is the claim the whole project is
built on: that these systems are predictable enough to plan for.
