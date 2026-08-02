# DocEngine: the plan from zero

The objective, everywhere in this document: minimize end to end
makespan, the wall clock from the moment a query starts to the moment
the last document's last answer exists. One query, one dedicated H100,
single tenant, prompts fixed. The reference query used for every
number below: four yes or no filters, pass rate 0.8 each, over 10,000
movie reviews, about 3.1 million tokens of documents in total, model
Qwen3-4B-FP8. A document that fails a filter skips the rest.

## The ceiling, derived

Reading dominates this workload, so the ceiling is a reading rate.
Four layers, each derived or measured, each below the last:

1. Spec sheet arithmetic: reading one token through a P parameter
   model costs about 2P arithmetic operations, here 2 x 3.6 billion =
   7.2 GFLOP per token. The H100's advertised dense FP8 rate is
   1.979e15 operations per second. Dividing gives 275,000 tokens per
   second, or 11.3 seconds for the corpus. Unreachable: it assumes
   the multiply units never stall and prices everything except
   multiplies at zero.
2. Measured silicon: the model's four matrix multiply shapes,
   benchmarked alone in FP8, sustain 1.35e15 operations per second
   (68 percent of spec), equivalent to 186,000 tokens per second.
3. Measured model: a transformer is only half multiplies by time.
   The bare model floor, bracketed by benchmarks with no serving
   stack, is 62,000 to 89,000 tokens per second.
4. Measured engine: vLLM reads at 80,000 tokens per second, flat
   across every setting, input form, and sequence length we tried.
   Its serving tax is at most ten percent and possibly near zero.

So the operating floor is 80,000 tokens per second. For the reference
query that means: corpus once, 39.5 seconds; plus the question tokens
and answers, about 7.5 seconds; a work floor near 47 seconds. Any
schedule that reads every document exactly once and does nothing else
wasteful lands there. Reading each document at least once is
unavoidable within a query, so nothing goes below 39.5 seconds except
the one idea that bypasses reading entirely (priority four below).

## The ladder: what we built and measured, from vanilla vLLM up

Every rung is measured on the reference query, cold start.

Rung 0, vanilla vLLM, task prompts. How nearly everyone uses these
engines: one prompt per document per filter, the task text in front.
122.8 seconds. The engine silently re-reads surviving documents at
every filter (3.03 times the corpus read in total), because nothing
tells it the requests are related.

Rung 1, vanilla vLLM, cache friendly prompts. Put the document first
so the engine's prefix cache could reuse it. 111.3 seconds, reading
2.74 times the corpus. The advantage almost vanishes at scale: the
cache holds about a quarter of the corpus, and by the time a
document's next filter arrives its notes are evicted. Lesson: a good
template without execution order control is worth almost nothing.

Rung 2, precomputed blocked schedules. Our analytical builder laid
out every batch in advance (possible only with outcomes planted) and
drove the engine batch by batch. 74.0 seconds, reading 1.27 times the
corpus. Lesson: order control recovers the reads, but closed batches
idle the machine while stragglers finish, and even a clairvoyant
timetable loses to reactive rules. Scheduling structure belongs up
front; timing belongs to the run.

Rung 3, the client library. Streaming with budget admission: a
document enters only when its notes fit, runs all its filters
back to back the moment each answer arrives, then releases its
budget; prompts pre-tokenized. Zero engine changes. 52.5 seconds,
reading 1.23 times the corpus, within a few percent of the calibrated
prediction. Along the way we measured the per request software floor:
1.12 milliseconds, half of it ours (fixed), 0.55 milliseconds
engine internal.

Rung 4, the in-engine scheduler, the shipping configuration. The
plan's decisions move inside vLLM through its official scheduler
seam: live documents' notes pinned, dead documents' notes freed the
instant the plan learns of death, request order assigned from
verified plan tags, unplanned requests refused, and the recency rule
left with zero authority (its firing is a crash, and it fired zero
times across the validated grid). 52.3 seconds alone, identical
answers, and the property no client side code can provide: under a
hostile co-tenant demanding more than the card's whole read
bandwidth, the query holds 52.0 seconds while stock vLLM never
finished inside an hour.

The analytical stack stands beside the measured one: a cost model
calibrated by the single measured constant that predicts every
measured cell within about five to ten percent; exact small instance
solvers, Bellman recurrences, and throughput programs for both answer
only and reasoning filters; the reasoning map (generation dominates
above roughly one hundred thinking tokens, speculation is punished
when saturated, favored when starved); the long document null result
(pipeline ties the optimum on the read floor; the contested regime
needs thinking comparable to document length, three or more stages,
and memory near 2.5 footprints together); and the finding that
clairvoyance is worth under one percent, so scheduling's value is
orchestration, not prediction.

Position today: 52.3 seconds, against a 47 second work floor, a 39.5
second read floor, and an 11.3 second spec sheet fiction.

## The physical operator catalog

A query plan composes one operator from each family, per document
batch. This is the database view of everything above and below.

Access operators, how a document's notes get on the card:
  A1 text scan: read the text, compute notes (2P per token at the
     operating rate). The default.
  A2 resident read: notes already on the card and pinned; near free.
  A3 notes load: stream persisted notes from the tier (priority
     four); cost is bytes over store bandwidth.
  A4 chunked scan: partial document admission for documents larger
     than memory (exists in the model, unused at current lengths).

Evaluation operators, how a question is asked:
  E1 task call: task text plus document plus cue, full re-read per
     call. The naive operator, rung zero.
  E2 branch call: document first request per stage, notes reused
     through the cache or pins. Shipped today.
  E3 truncated continuation: one living sequence per document,
     append question, read answer, roll back, append next. Priority
     two; kills the per request toll.
  E4 cascade call: k question tails evaluated concurrently over one
     pass through the notes (the Hydragen or tokasaurus style
     kernel). Semantics preserving, unlike prompt fusion, because
     each tail still attends over only the document and itself.
     Parked until reasoning filters make note reads material.
  E5 shared scan call: tails from several queued queries evaluated
     while the notes are hot. Priority five; the cross query twin of
     E4.

Gating operators, when calls are issued:
  G1 stage wave: barrier per filter across all documents. Rungs zero
     through two.
  G2 streaming gate: per document, answer driven, no barriers.
     Shipped.
  G3 lookahead k: a block of k questions ungated, waste on failure.
  G4 adaptive mix: per document k chosen against the memory budget;
     wins 33 percent in the proven corner, ties elsewhere.

Memory operators:
  M1 recency rule: the vanilla engine's default. Demoted to
     unclaimed memory only, which in strict mode means nothing.
  M2 pin and eager free: plan claimed retention. Shipped.
  M3 spill and recompute: the retain some, re-read some mix the LP
     prices; relevant only in the contested corner.
  M4 persist and restore: ingest time notes writing, query time
     loading; the memory side of A3.

The measured ladder in these terms: rung zero is (A1 per call, E1,
G1, M1); the shipped configuration is (A1 once, E2, G2, M2); next is
(A1, E3, G2, M2); the 32B tier is (A3, E3, G2, M2 plus M4); the
reasoning corner is (A1, E2 or E4, G4, M2 or M3). The analytical
layers are the cost based optimizer choosing among exactly these,
given filter count, pass rates, document lengths, thinking lengths,
and the hardware constants.

## What is left, in priority order

Priority one, the profile (about fifteen GPU minutes). Split the 0.55
millisecond per request bundle into named parts (serialization,
prefix hashing, detokenizer setup, scheduler bookkeeping, sampling,
client wakeups) with a profiler on both processes during a reference
run. We have bounded this bundle end to end but never split it; the
split ranks what truncation must keep cheap on the once-per-document
path and whether anything else is worth precomputing (block hashes
are precomputable in principle; we suspect they are tens of
microseconds, and the profile will say).

Priority two, sequence truncation (days; the last big makespan item).
One living engine sequence per document: append a question's tokens,
decode the answer, roll the sequence back to the document boundary,
append the next question; the gate runs inside the scheduler, which
knows the yes and no token ids. Prompts unchanged, each question sees
exactly today's context, and a document costs one request instead of
n, collapsing the per request residual (about five seconds of the
reference query). Target: high forties, essentially the work floor.
vLLM's existing streaming input sessions provide the append
machinery; we add rollback and the gate. Milestones: rewind proved
correct against separate request answers, in-scheduler gate, the 10k
grid.

Priority three, long document validation (an afternoon). Run thirty
100,000 token documents and one hundred 30,000 token documents
through the shipped configuration. The model predicts the read floor
with policy differences compressed; this is the first measurement of
100,000 token contexts and of pins that large, and it closes the
single query story across document lengths.

Priority four, the persisted notes tier (about a week; the only path
below the read floor). At corpus ingest, read every document once and
persist its notes; at query time, stream notes into the card instead
of re-reading text. The break even rule: the store's bandwidth must
exceed kappa times the prefill rate, because that is the rate at
which recomputation produces notes. The table:

  model   notes/token   prefill rate   break even    NVMe (7-14 GB/s)   S3
  4B      72 KB         80,000/s       5.9 GB/s      yes, comfortably   only with a 100 Gbps
                                                                        class network path
  32B     128 KB        ~9,300/s       1.2 GB/s      trivially          yes, even on common
                                                                        10-25 Gbps paths

So the right architecture is a tier, not a choice: S3 (or any object
store) as the durable, cheap layer (roughly 4 times cheaper per byte
than attached NVMe, elastic, survives machines), NVMe as a read
through cache in front of it, and the loader prefetching along the
admission order the scheduler already computes, so notes arrive just
before their document is admitted. For the 32B tier S3 alone clears
the bar; for 4B a cold S3 read is roughly a wash with recompute on a
common network and a win once the NVMe cache is warm. KV compression
(two to four times, decoded on the card) would put S3 above the bar
for 4B as well and is the known refinement.

On zero copy: mostly yes, with one boundary. If notes are serialized
at ingest in the engine's native paged block layout (fp8, per layer,
block aligned), loading needs no format conversion at all, and from
local NVMe the transfer can bypass host memory entirely via GPUDirect
Storage, landing in the paged cache with one batched scatter to the
block slots assigned at load time (the scatter is needed because
block ids differ between ingest and load). From S3 the bytes must
transit host memory once (object stores speak TCP, not DMA), but
still with zero serialization work. vLLM's KV connector interface is
the integration point either way.

Priority five, shared scans (days, client side only). Queued queries
over the same corpus merge so the corpus is read once for all of
them. With one GPU per query as the product shape, this is the
throughput multiplier for the common case, and it needs no engine
work.

Priority six, the reasoning measured grid. When reasoning filters
matter, validate the phase A map on hardware using the forced length
instrument (generate exactly G tokens, answer first), which needs no
artificially hard task. Decode pricing, the saturated no speculation
verdict, and truncation rolling back through thinking all get tested
at once.

Parked, with explicit triggers: the cascade or Hydragen style shared
prefix kernel (trigger: measured reasoning filters with a material
attention read share; complementary to truncation, since cascade
saves note reads within a speculated block while truncation saves
software cost between gated stages); the adaptive mix scheduler
(trigger: workloads in the proven corner of three plus stages,
thinking near document length, memory near 2.5 footprints); multi
query arbitration (out of scope by product decision); the 100,000
document demonstration (whenever a headline is wanted); the 32B
model tier (bundle with the disk KV tier, where it shines).

Standing decisions: prompts are a fixed interface; single tenant;
one GPU per query; the artificially hard filter accuracy study is
skipped; strict mode is the shipping default.
