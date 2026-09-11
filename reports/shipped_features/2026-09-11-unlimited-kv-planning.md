# The join search prices KV reuse as unlimited

Date: 2026-09-11

## What changed

- The join search (`planner/joins.py`) now assumes a document prefix
  computed once is free at every later anchor use, the same assumption
  the speed-of-light estimate makes. A filtered alias's first anchor use
  is priced as "filter" resident; any later anchor use, consecutive or
  not, as "kept". The capacity-based credit is gone: `retention.allocate`,
  the `resident_*` fields of `AliasStats`, `with_resident_fraction`,
  `fit_resident_documents`, and the `keep_min_doc_tokens` and
  `keep_resident_fraction` fields of `PackedFilter` were removed, along
  with the "shared KV on" plan remarks.
- Plan emission places a filtered alias's chain right before the first
  group anchored on it, after any barrier, when no earlier group uses
  that alias as a partner, and marks that group's edge streamed. Before,
  every filter ran before every join and only the first group could
  stream.
- The executor is unchanged. It keeps as much KV as the arena holds,
  pinned across a streamed edge and pooled across a materialized one,
  and `regret_tokens` reports what it could not keep.

## Why

- The old credit modeled the old hand-off: survivors waited in a capped
  pool, so about 11.6% of IMDB-10's filtered reviews were priced as
  resident wherever their join sat in the order. The search picked the
  order that saved a few thousand tiny tuples and lost 1.2 million
  prefix tokens to recompute. With streaming, the truthful residency at
  a first anchor use is 100%, and the plan should be chosen on that.
- Over-crediting where the executor cannot keep everything is
  acceptable: the order that would be best with unlimited KV is the one
  to aim the executor at, and the recompute it leaves is measured, not
  hidden.

## Numbers

- On IMDB-10 the search still puts the `r2` group first (16.64 estimated
  seconds against 16.82 for `r1` first, both now with `r1` fully
  resident), and `r1`'s chain now runs after the barrier and streams
  into its group. The measured effect is in
  [the streamed filter-join report](../2026-09-11-streamed-filter-join.md).
