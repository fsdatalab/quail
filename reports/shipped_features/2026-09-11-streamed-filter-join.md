# Filter survivors stream into the join anchored on them

Date: 2026-09-11

## What changed

- The first join group's anchor filter no longer runs to completion
  before the join starts. When that anchor alias has a filter chain, the
  planner marks the edge between the `PackedFilter` and the
  `AnchoredJoin` as streamed (`AnchoredJoin.stream_anchor`). The generic
  runner (`runtime/runner.py`) does not execute a streamed producer on
  its own: the consumer receives it as a `StreamedInput`, runs it, and
  returns its result under `NodeResult.produced`, so both nodes keep
  their own answers and metrics.
- `executor/loop.py` gained `FilterStream`, the filter chain as one
  chunk per `next()` call. With `hold_survivors`, a document that passes
  its last stage keeps its arena pages pinned and comes back from
  `next()` as a (key, prefix) pair. `run_filter` is now a loop over that
  stream and behaves as before.
- `run_join` accepts an `anchor_source`. It pulls from the stream until
  its own chunk can fill (`JoinAdmission.buildable_tokens`), runs one
  join chunk, and pulls again. The stream reports when fresh admission
  is short of arena pages; the join then drains what it holds before
  pulling more. Anchors are admitted incrementally through
  `JoinAdmission.admit`. Each streamed anchor is resident, so it packs
  the frame and its partner suffixes only.
- A held document's pages cover the join's largest frame, so the join
  never claims a page for a streamed anchor. The chain re-reads the
  arena's free page count before every chunk, because the join frees
  and claims pages between chunks (`FilterAdmission.free_pages`).
- Each operator keeps its own attention path: the chain sets `unified`
  and the join sets `merge_quant` before each of its chunks.
- On several GPUs the anchor's chain runs inside the join round on each
  GPU, over that GPU's filter shard of every anchor document. Partner
  chains still finish and gather first. Nothing new crosses GPUs.
- The planner removes the streamed alias from the retention pool's
  initial uses, credits every streamed anchor as resident, and turns
  arena writes on for a single-stage chain that streams. The pool now
  serves the other aliases only. `explain()` shows "survivors stream
  into the join with KV pinned" on the chain and "KV: anchor=streamed
  from its filter" on the join.

## Why

- Under operator-at-a-time execution every filter survivor's KV had to
  be held for the rest of the filter phase, and the retention pool (the
  arena minus two chunk budgets) held about 470 reviews on IMDB. The
  rest were recomputed at the join: 1,217,732 recomputed KV tokens on
  IMDB-3, 32.2% of its fresh tokens. Streaming bounds the pinned set to
  what the join has not answered yet, so a survivor's KV is read once
  and freed.
- This is the filter-to-join half of issue #23. An anchor switch
  (`Exchange`) stays a barrier: the new anchor's documents went through
  the GPU as partner suffixes, and suffix KV is never written.

## Numbers

- IMDB-3: 32.58 to 22.47 seconds (31.0% less), recomputed KV tokens
  1,217,171 to 0. IMDB-4: 20.32 to 17.35. IMDB-5: 18.16 to 16.60.
  BIO-3: 89.97 to 80.65. IMDB-10: unchanged at 59.0 in that run, because
  its first join group anchors on an unfiltered alias and the filtered
  alias anchors a later group; the
  [planner follow-up](2026-09-11-unlimited-kv-planning.md) then deferred
  that chain to its group. Answer tables identical on all five.
- Details, the prediction, and the plot are in
  [the report](../2026-09-11-streamed-filter-join.md).
