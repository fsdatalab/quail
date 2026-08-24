# Multiple pairwise joins, first slice: two joins sharing one anchor

Date: 2026-08-24. First slice of issue #38 (Order steps 0 and 1),
plus the standalone LIMIT bug #39. CPU-side code and CPU tests only;
no engine run, so no measurements in this note.

## What changed

A query can now hold several pairwise join predicates, as long as
they all share one table. `reviews x products` then
`products x recalls` runs as two stages anchored on products, instead
of one three-way cross product. For three 1,000-row tables that is
2 x 10^6 pairs to score instead of 10^9 triples. The single-call
n-way join is unchanged and still available.

By layer:

- **LIMIT fix (#39).** LIMIT counts output rows. The engine enforced
  it by cutting filter-survivor document lists, which is wrong for
  join queries: one document appears in zero or many output rows, so
  a query with a join and a LIMIT could silently return fewer rows
  than exist. New rule, in `runtime/coordinator.py`
  (`filter_round_limit`): the filter round gets the limit only for
  filter-only payloads; with joins it gets None everywhere (the
  single-GPU path, each GPU child's sub-payload, and the multi-GPU
  merge), and `_assemble` caps the final output rows. Filter-only
  queries keep the early stop - there one survivor is one row, so
  stopping early is correct and saves work.
- **Builder** (`builder.py`). The one-full-join guard is removed.
  Chained `ai_join` calls add one pairwise join each. Each call's
  prompt must reference every table that call joins and at least one
  table already in the query, so the joins connect.
- **SQL front end** (`sqlfront/compile.py`). Each multi-table
  `AI_FILTER(PROMPT(...))` - on a JOIN's ON or as a WHERE term - is
  its own join predicate. New coverage rule: every JOINed table must
  appear in at least one join predicate, and the predicates' tables
  must form one connected graph with the FROM table.
- **Planner** (`planner/decide.py`). With more than one full join,
  `choose_shared_anchor` anchors every stage on the table they all
  share (the cheapest one by summed stage tokens when several are
  shared). Joins that share no table, or forced anchors that would
  split the stages, raise NotImplementedError: the between-group
  re-shard is a later slice.
- **Worker/executor.** No change needed. `_stage_groups` already runs
  consecutive same-anchor full stages as one gated `run_join` call,
  and `run_join` already drops an anchor document with no stage-1
  match before stage 2 (valid pruning: it can appear in no output
  tuple).
- **Recombination** (`runtime/session.py`, `_assemble`). New code.
  With k full stages sharing one anchor, output tuples come from
  matching the stages' surviving pairs on the anchor's document ids:
  for each surviving anchor document, one stage's matches extend the
  tuple, and a partner table two stages share must carry the same
  document in both. Id matching only, no model calls.
  `executor/pack.py`'s `assemble` is the two-stage reference
  semantics and is checked against in tests. The per-member check
  against each table's final survivor set and the exists/anti gate
  semantics are unchanged. LIMIT applies to the final tuples only.

## Why

Issue #38: composing pairwise joins avoids the cross-product blowup
of forcing every multi-join condition into one n-way prompt. This
slice is the smallest correct piece: two (or more) full joins on one
shared anchor, single group, no re-shard. #39 had to land first
because any upstream LIMIT cut compounds across two pair sets.

## Not in this slice

Later slices of #38: the joint (order x anchor) planner enumeration
with barrier costs, gate/dedup/replay across group boundaries, the
between-group re-shard with barrier-time decisions, and benchmarks.
Stages anchored on different tables are refused with a plain error
at planning, in the coordinator, and in `_assemble`.

## Tests

`uv run pytest tests/ -q` from `quail/`: 176 passed. New coverage:
the limit rule (coordinator split, merge, and payload), chained
builder and SQL compilation to two JoinSpecs, connected-graph
rejection, shared-anchor planning and both NotImplementedError
paths, two-stage split and merge across simulated workers, worker
stage grouping, and recombination checked against `pack.assemble`
with survivor-set filtering and LIMIT on final triples.
