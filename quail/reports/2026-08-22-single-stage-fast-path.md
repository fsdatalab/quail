# Single-stage filters skip the arena (issue #6, on the unified path)

Port of the single-stage fast path (PR #13) onto the attention-path
split (PR #35, issue #24). Stacked on the `attention-path-selection`
branch.

## What changed

A single-stage filter query (one question, no joins, no KV store) has
no later reader of any document's KV. On the unified filter path,
every document still paid:

1. `arena.alloc` / `arena.free_key` page accounting, with capacity
   pages reserved for the question tail,
2. the per-layer scatter of every current token (document AND
   question) into arena pages (`kv_row_scatter`), on all 36 layers,
3. the paged-KV indirection inside the one causal FlashAttention
   call (block table reads instead of contiguous rows).

The fast path skips all of it when nothing will read the KV:

- The planner decides. `plan_query` puts an `arena_writes` field on
  every FilterChain operator: False exactly when the chain has one
  stage and no store, True otherwise. `explain()` shows the field on
  the FilterChain line, and the plan carries a remark saying why
  when writes are off. The payload forwards the decision per alias
  (`filter_arena_writes`) and the worker passes it to `run_filter`.
- `run_filter(arena_writes=...)` executes the decision. `None`
  (direct callers: warmups, ablation cells, old payloads) derives
  the same rule from its arguments. `True` forces the arena path
  (the baseline in this report). `False` with multiple stages or a
  store raises, because `store.save` and later stages read the
  arena - a wrong planner call fails loudly.
- `pack_chunk`: a fresh group whose key owns no arena pages packs
  `[document | question]` as ONE causal segment. Under unified
  attention a chunk is either all paged or all unpaged; mixing
  raises.
- `attention_unified`: a chunk with no arena pages runs the plain
  varlen causal call. No scatter, no paged read, no merge.
- `FilterAdmission(arena_pages=None)` removes the page bin:
  admission is the token budget alone.

This is less work per token than stock vLLM does for the same
prompt, not just a match. Stock vLLM (the pinned vllm==0.26.0, v1
engine) writes every prompt token's KV into its paged cache with
`reshape_and_cache_flash` and its one causal FA3 call reads K and V
back through the block table - quail's unified path copies exactly
that pattern. Stock cannot skip the write: after prefill it decodes,
and the generated tokens read the prompt's KV. A quail filter never
decodes - the TRUE/FALSE answer is read from the final position's
hidden state in the same forward pass - so the KV has no reader and
both the write and the paged read can go.

The join path is untouched: `run_join` still allocates pages for
every anchor - a join group's many suffixes share the anchor's KV
through the arena by design.

Two defects in the `attention-path-selection` base, both hit by any
single-stage filter, are fixed in the first commit of this stack:
`attention.py` defined `attention_merge_quant` and
`attention_unified` twice (the second copies silently overrode the
first), and `run_filter`'s unified-mode `temp_tail` expression
raised `TypeError` on single-stage queries - `calibrate()` and
`tests/gpu/limit_gate.py` were both broken by it.

## Why the answer story is simpler than PR #13's

The original PR #13 was written against the retired split attention
path, where the arena route computed suffix attention as two calls
plus a softmax-state (LSE) merge. That merge reorders floating-point
operations relative to one softmax, so the two routes flipped
629/10,000 near-tie answers and the cell needed a stock-vLLM referee
to show the flips were symmetric rounding, not a bug.

The unified path removed that difference. The kernel-parity cells
(`results/attention_parity.json`, `results/attention_parity_64h.json`)
measured the unified paged causal call bit-identical to a contiguous
causal call (max_abs 0.0 on every edge case), and the fast path IS
that contiguous call over the same `[document | question]` rows at
the same positions. The gate is therefore exact: 0 answer flips.
The referee and the re-chunked noise-floor control are gone with it.

## Setup

- One H100, Qwen3 4B fp8, bf16 KV, the `quail-milestone1` Modal app.
- The 10,000-document IMDB corpus with one planted-flag question
  (`build_corpus(n_filters=1)`), ~3.20M corpus tokens.
- A/B in one cell (`tests/gpu/milestone1.py::run_filter1`):
  `arena_writes=True` (the arena path on unified attention, the
  configured baseline) vs `arena_writes=False` (the fast path). Same
  corpus, same 110,376-token chunk budget, both paths warmed by
  `warm_kernels`, 2 repetitions each.
- Correctness gates ran first in the probe cell
  (`tests/gpu/milestone1.py::run_probe`).

## Prediction (stated before the run)

- ~3.23M fresh tokens. The unified filter path ran 8.45 us/token at
  the five-stage geometry (`results/attention_paths.json`), which
  puts the arena run near 27-28 s.
- The fast path removes the scatter, the paged indirection, and the
  page accounting: a 2-5% cut. Smaller than the 6.3% the same
  switch bought on the split path in PR #13, because unified
  already dropped call B and the LSE merge - the two largest things
  the old fast path skipped.
- 0 answer flips between the paths.
- Probe: every parity gate at 0 disagreements, both construction
  raises fire.

## Measured

Not yet measured. The session that authored this stack cannot reach
Modal: its egress proxy does not carry gRPC, which the Modal client
requires. Two commands settle it (from `quail/`, on this branch):

    uv run modal run tests/gpu/milestone1.py::run_probe   2>&1 | tee results/m1_probe.log
    uv run modal run tests/gpu/milestone1.py::run_filter1 2>&1 | tee results/m1_filter1.log

then

    uv run --with matplotlib python reports/make_single_stage_fast_path_plots.py

and fill this section from `results/m1_probe.json` and
`results/m1_filter1.json`. The probe must pass every gate before the
filter1 walls count.

Figure (produced by the plot script after the run):
plots/single_stage_fast_path.png

## What would falsify the prediction

- Any nonzero `flips` in m1_filter1: the fast path's contiguous call
  is then NOT bit-equivalent to the paged call at this geometry,
  contradicting `results/attention_parity.json` - stop and
  investigate before trusting either path.
- A fast-path wall above the arena wall: the plain varlen call would
  have to be slower than scatter plus the paged call, which no
  measured constant supports.

## Files

- `tests/gpu/milestone1.py::run_probe` - correctness gates,
  including the five new fast-path gates.
- `tests/gpu/milestone1.py::run_filter1` - the A/B cell.
- `reports/make_single_stage_fast_path_plots.py` - the figure.
