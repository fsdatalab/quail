# Physical plans

## Logical query

The query is an ordered conjunction of filters. A document is returned only when
every filter returns true. Online execution may stop a document after its first
false answer.

Ground truth belongs to the evaluator. The runtime receives document tokens,
filter tokens, and model answers that have already returned. It does not receive
future answers.

## Task first

Task first runs filter 1 over every live document, then filter 2 over the
survivors, and so on. It creates a barrier between stages. A document that
finishes early waits for the slowest document in its stage.

This is a stock vLLM baseline.

## Streaming document first

Streaming document first admits documents continuously. A document advances to
its next filter as soon as its current answer returns. Full document KV blocks
remain in HBM while the document is live.

Stock vLLM owns scheduling and KV allocation in the baseline. DocEngine owns both
in the custom implementation.

## One request per document

The first request contains the document and filter 1. It may be split into exact
token chunks across engine steps. Later filters reuse full document KV pages and
recompute only the partial page at the document boundary.

The custom runtime constructs vLLM model-runner inputs directly. It does not call
the vLLM scheduler or KV cache manager.

## Fused k

Fused k evaluates k independent filter tails over one document prefix. Every tail
attends to the document and its own earlier tail tokens. It never attends to a
different filter tail.

The verified runtime gives one document a shared FP8 prefix page list and each
tail its own FP8 pages. It runs one prefix group per forward.

FlashInfer accepts several document groups in one forward, and the direct kernel
matches ordinary attention closely. In the full model, however, large
multi-group batches changed Boolean answers. Multi-group cascade is therefore an
explicit experiment, not a deployable physical plan.

k is a physical choice, not part of the logical query. Speculative answers after
an earlier false answer are wasted work.

## Finite-query planner

The exact planner enumerates legal actions and possible answers for small
queries. The measured planner builds exact variable-length candidate batches and
ranks them with the versioned cost catalog.

The steady-state throughput LP remains a lower reference. It is not the runtime
scheduler.

## KV ownership

DocEngine keeps an explicit free list and reference set for every FP8 page.
Allocate, share, extend, truncate, and free are explicit operations. An
allocation failure stops the run. There is no recency eviction.

Full invariant scans run at query boundaries. The hot path checks only pages
touched by the current operation.

## Current evidence

See [REBUILD_RESULTS.md](REBUILD_RESULTS.md) for measured results and gate status.
See [EXPERIMENTS.md](EXPERIMENTS.md) for immutable run IDs and revisions.

Answer-aware schedules are oracle references only. They are not listed as
deployable physical plans.
