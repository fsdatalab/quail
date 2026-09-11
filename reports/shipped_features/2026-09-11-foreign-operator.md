# The Foreign operator: a user function between two GPU operators

Date: 2026-09-11

## What changed

- Builder: `.apply(fn, columns=[...])` and `.apply_table(fn,
  columns=[...])` put a Python function in the plan. The function gets
  one Arrow table per alias (row indices under the alias name plus the
  listed columns) and returns the ids to keep, every id
  (`ids="preserve"`), or, after `join(other)`, the pairs the next AI
  predicate is asked about. It never invents an id; the runtime checks.
- Logical node `Apply` and physical node `Foreign` (`apply:<name>`,
  type `quail.foreign`). Its `kind` is `per_batch` or `barrier`, its
  `ids` is `preserve`, `drop`, or `pairs`. A pairs node feeds the join
  on a new `pairs:<written position>` port (value type `PAIRS`).
- `per_batch` keeps the stream: the node appends the function to the
  survivor stream's `transforms` (or returns `StreamedPairs`), and the
  join runs it on each batch before admission, freeing the KV of any
  survivor it drops. `barrier` makes the planner materialize the
  anchor's chain; the function runs once over every survivor.
- `validate_streams` refuses a barrier or any other consumer on a
  pinned survivor stream; it runs on every plan and request. The
  planner refuses per-batch functions on several GPUs and the request
  backends refuse `apply`.
- The session registers the functions (`registry.functions`) and
  ships the columns they read as `columns:<alias>` request relations.
  `explain()` shows the node and marks a join stage `over pairs from
  <name>`.

## Why

- A caller who owns the data often knows which pairs matter or which
  documents to drop, in code the engine cannot see. Letting that code
  sit between two GPU operators, without a barrier when it works per
  batch, keeps the pipelining and the pinned KV that streaming
  survivors won.

## Numbers

- See [the Foreign operator report](../2026-09-11-foreign-operator.md).
