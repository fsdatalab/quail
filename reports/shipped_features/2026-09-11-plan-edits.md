# Per-node estimates, readable node ids, and plan edits

Date: 2026-09-11

## What changed

- Node ids name the operator and what it works on: `scan:c1`,
  `ai_filter:c1`, `ai_join:e1` (`ai_join:e1:2` for a second group on
  the same anchor), `barrier:e2` (the next anchor), `exchange:e1`,
  `apply:same_page`, `recombine`, `project`, `limit`. The old
  `input:`, `filter:`, `group:<n>`, `barrier:<n>`, and `sink` ids are
  gone, in the plans, the retention settings, and the docs.
- `PhysicalPlan.estimates` prices each model node's own work in
  seconds through the speed of light model; `explain()` shows it on
  the node line, with a note that the parts do not add up to the plan
  total because chunk packing shares forward passes. A filter chain a
  join anchors on also shows the recompute it would pay if its KV were
  released instead of pinned (survivors past the retention pool cap
  times prefix cost). The join search never reads that figure.
- `PhysicalPlan.insert(node, between=(producer, consumer))`,
  `remove(node_id)`, and `move(node_id, between=...)` edit the DAG one
  edge at a time and return a new plan. An illegal edit raises
  `PlanEditError` at the call with the rule it broke. After an edit
  the plan re-derives `pin_survivors`, `keep_kv`, and `hold_tokens`
  from the new shape (a `Barrier` on a pinned edge turns the pin off)
  and re-estimates every node. `query.run(plan=edited)` executes it.

## Why

- A caller who reads a plan should be able to see where the time and
  the KV risk are, and move a function or a barrier to a different
  edge without learning the planner's internals. The edits are the
  ordinary graph edits; the invariants the planner relies on are
  re-derived rather than trusted.

## Numbers

- See [the plan edits report](../2026-09-11-plan-edits.md).
