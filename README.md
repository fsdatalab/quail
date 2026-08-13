# Quail

Quail runs SQL-style filter queries over documents with an LLM, and
tries to make them fast by changing how the inference engine manages
memory.

This is a research repo. It holds one experiment, run six ways, and
the code that produced it.

## The problem

Say you have 10,000 movie reviews and a query with five filters:

    WHERE is_positive(review)
      AND mentions_acting(review)
      AND ...

Each filter is a question the LLM answers about the document. Five
filters, 10,000 documents, so up to 50,000 questions.

The expensive part is not the answer. Each answer is one token
("YES" or "NO"). The expensive part is reading the document — about
320 tokens of it, five times over, once per filter.

An inference engine like vLLM caches what it has already read, so in
principle the second filter reuses the first filter's work. In
practice it often cannot. The cache is sized for one model, and our
corpus is 3.4 times bigger than it. So the engine drops documents to
make room, and when the next filter asks about a dropped document, it
reads the whole thing again.

The engine drops by recency, because it does not know what is coming.
A query does know. That gap is what Quail is about.

## What we tried

**Pipelining.** Send a document to filter 2 as soon as filter 1
answers, instead of pushing all 10,000 documents through filter 1
first. Obvious, and a normal client can do it.

**Token-based admission.** Let a document into the engine only when
its tokens fit a budget computed from the cache size. The usual
alternative is to cap the number of concurrent requests, which cannot
work well here: our documents range from 44 to 2,947 tokens, so a
request count tells you nothing about how much memory they need.

**KV rewind.** Keep one request alive per document for the whole
chain of filters. After each answer, erase the question from the
document's cached state and append the next question in its place.
Because the request never ends, the engine cannot drop the document
between filters. This needs changes inside the engine — no client can
do it through the public API.

## What we found

On one H100 with Qwen3-4B (fp8), 10,000 documents, five filters:

| | wall time | times the corpus is read |
|---|---|---|
| stock vLLM | 42.9 s | 1.23x |
| KV rewind | 39.8 s | 1.14x |

**1.08x faster.** Modest, and that is the honest number.

We first measured 2x, then found the baseline was misconfigured. It
was running 4,096 concurrent requests — the engine's default limit —
which needs 1.6 times more memory than the cache has. The engine
thrashed and read the corpus 2.40 times. Setting its concurrency from
the cache size instead (2,048 documents) took it from 80.1 s to
42.9 s. Most of the "win" was our baseline being badly set up.

What survives is smaller and has a specific cause. The engine matches
its cache in fixed 16-token blocks, so the block where the document
ends and the question begins never matches and gets recomputed on
every filter. A rewind cuts at the exact token instead, so it does
not pay that.

**Reusing work across queries is where the real win is.** If a second
query arrives for the same documents, we can save their cached state
to CPU memory and load it back rather than re-reading the documents:
2.33 s instead of 4.54 s, a 2x speedup. That holds whenever the
transfer is faster than re-reading, which for this model means faster
than 7.2 GB/s. Bigger models make it easier, since they read more
slowly.

## Is the hardware even the limit?

Before optimizing, we checked what the H100 can do at all.

The GPU is busy 99.5% of the time during these queries, so there is
no idle time to reclaim. But it only reaches 97,000 tokens per second
against a theoretical 275,000. We looked at where the other two
thirds go.

Not slow matrix multiplies: Nsight Compute puts those at 91-93% of
the hardware ceiling. The time goes to everything around them.
Per token:

| | time | share |
|---|---|---|
| matrix multiplies | 6.5 µs | 61% |
| fp8 quantize/scale | 2.0 µs | 19% |
| normalization | 1.35 µs | 13% |
| activation functions | 0.74 µs | 7% |
| attention | 0.45 µs | 4% |

So 39% of every token is spent on small memory-bound operations
between the matrix multiplies. Making the multiplies faster would
barely help. This is a property of the model and the engine, not of
our query.

We also fitted a simple model of step cost by sweeping the batch
size:

    time per step = 10.7 µs × tokens + 3.1 ms

The fixed 3.1 ms is per-step overhead — kernel launches, the
scheduler, Python. It stops mattering past roughly 300 tokens per
step, which matches where the analytical roofline says the matrix
multiplies saturate (~400). Our batches are 25,305 tokens, far past
both.

## Try it

```bash
pip install -e ".[dev]"

python -m pytest tests/ -q     # 45 tests, no GPU needed
python -m quail.roofline       # what the hardware allows, from specs alone
```

The engine experiments need a GPU and run on Modal:

```bash
modal run experiments/modal_filters.py                # the main result
modal run experiments/modal_profiling.py::batchsweep  # the step cost model
modal run experiments/modal_profiling.py::torchprof   # where the time goes
modal run experiments/modal_profiling.py::ncubench    # are the kernels good?
modal run experiments/modal_persist.py --stage baseline
modal run experiments/modal_persist.py --stage store  # reuse across queries

python plots/make_figures.py                          # redraw the figures
```

There is no local GPU path. The engine code reaches into vLLM
internals that are pinned to version 0.26.0.

## The experiment

Five yes/no filters over IMDB movie reviews. Each review has a line
added to the end:

    [FLAGS] FLAG_1=YES FLAG_2=YES FLAG_3=NO FLAG_4=YES FLAG_5=NO

Filter *j* asks what `FLAG_j` says. A document that fails a filter
skips the rest.

The answer is written in the document, so this measures how fast the
system runs, not how well the model thinks. Pass rates are 0.9, 0.9,
0.9, 0.8, 0.8, so about 47% of documents survive all five filters and
the survivor count drops as the query goes — which is what makes
skipping worth doing.

Answers are restricted to the YES and NO tokens with a one-token
limit, so each answer comes out of reading the document and no
generation step ever runs.

## Where things are

```
quail/
  roofline.py             what the hardware allows, from specs alone
  plan/planner.py         describe a query, get a plan
  plan/cost.py            every measured constant, in one table
  runtime/engine_client.py   the two ways to run a query
  engineext/scheduler.py     the vLLM scheduler we substitute in
  engineext/chainlogic.py    its decision logic, pure and testable

experiments/
  workload.py             the documents and questions everything shares
  modal_filters.py        rewind vs stock
  modal_profiling.py      batch sweep, profiler, Nsight Compute
  modal_persist.py        reuse across queries
  modal_pinprobe.py       how fast the CPU-GPU link actually is

plots/make_figures.py     redraws every figure in results/plots
```

`chainlogic.py` imports no vLLM, torch, or numpy. Every decision the
scheduler makes is computed there on plain values and tested with no
engine installed, so a vLLM upgrade can break the glue without
breaking the rules.

## Scope

Filters only, Qwen3-4B, one H100. Open-ended text generation,
classification, larger models, and multi-GPU are all out for now — we
want the simplest case understood before adding any of it back. Git
history has the removed code.

## Known problems

- Saving to CPU memory runs at 10.2 GB/s through vLLM's connector,
  but the same memory copies at 55.4 GB/s directly. Five sixths of
  the link disappears somewhere inside, and we do not know where.
- A batch size of 1,024 is reproducibly slower than 512. No
  explanation; we left it out of the fit rather than hide it.
- The per-token cost is not really constant. Matrix multiplies cost
  4.65 µs per token at small batches and 6.51 µs at large ones,
  probably cache behaviour. The model averages over it.
- This model gets about 27% of the flag values wrong, even though
  they are written in the text. Both methods get them equally wrong,
  so comparisons are unaffected, but do not use this setup to say
  anything about accuracy.
- Text generation is untouched. Everything here is about reading.
