# 2026-09-11: a ponytail pass over the pair-join, Foreign, and plan-edit code

The `ponytail-review` skill now checked in under `.claude/skills/` asks
one question of each piece of code: does it need to exist, and if so,
can it be shorter. This pass ran it over every line the branch added
under `quail/` for joins over pairs, the Foreign operator, and plan
edits. Nothing here changes a result, a report key, a plan shape, a
node id, or a public call.

Before: 20,422 lines under `quail/`. After: 20,338 lines, 84 fewer.
The diff is 67 lines added and 151 removed across 17 files. The test
suite stays at 112 passed.

## Deleted

- `possible_anchor_aliases` and `_length_stats` in
  `quail/planner/decide.py`: no caller once the old retention
  allocator went.
- `AliasStats.histogram` in `quail/planner/joins.py`: computed for
  every alias and read by nothing after that same allocator went.
- `FilterAdmission.sync_free_pages` and its guards: one caller, one
  assignment; the join loop now sets `free_pages` directly.
- `FilterStream.budget`, the `held` entry in the stream holder, and the
  `anchor_batch: None` input key the worker sent: written, never read.
- The barrier-reads-a-stream guard in `ForeignRuntime`: `validate_streams`
  refuses that graph before any runtime sees it.
- `StreamedPairs.stream`, `anchor`, `partner`, `written_pos`: set at
  construction, read by nothing.
- The `pa.Array` to `pa.chunked_array` conversion in `pair_table`:
  `pa.table` and `pc.cast` take a plain array.
- `_REMOVABLE` in `quail/planner/plan.py`: one use, and its members were
  already spelled out in the error message next to it.

## Shrunk

- One `pair_partner` in `quail/runtime/pairs.py`. The request backend
  had its own copy of the same rule, and the coordinator rebuilt the
  per-anchor partner rows that `partner_maps` already builds.
- The join search's own records now carry each stage's work, so the
  planner no longer walks the chosen sequence a second time to get it.
- The planner groups the join sequence once with
  `retention.group_sequence` instead of three times with three copies
  of the merge rule.
- The per-alias filter work is computed once and summed, instead of once
  for the total and again per alias.
- `_filter_work`, `_where_terms`, and `BoundBuilder.apply_table` were
  wrappers with one line of their own; the callers say that line.
- `explain` uses `PhysicalGraph.node` instead of building its own index.

## Found and left alone

- `join_applies`, `pair_fraction`, `JoinStage.pair_fraction`,
  `resident_docs`, and the `group_ids` default in `retention.schedule`
  have no production caller but are asserted or imported by tests.
- Duplicate argument checks in `Query.apply`, `Query.apply_table`, and
  `_ai_join_pairs` repeat what `Apply.validate` checks later; their
  messages are the ones a user sees first, so they stay.
- The `if self.done` return in `FilterStream.next` and the `anchor_done
  is None` branch in the empty-stage drain are reachable from outside
  callers and stay.
- The "node seconds do not add up to the plan estimate" note in
  `explain()` is user-facing and asserted by a test.
- `_stream_reaches_join` has one caller but its docstring carries the
  trial-graph rule; inlining it saves five lines and the explanation.
- `quail/bench/quailb.py`'s `join.on` branch could fold into one call;
  left for a change that touches the benchmark runner.
