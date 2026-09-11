# Joins over pairs: an equality chooses what the model sees

Date: 2026-09-11

## What changed

- A new logical node `Join(left, right, on)` holds ordinary column
  equalities; `on=()` is a cross join. The AI predicate, still a
  `SemanticJoin`, sits on top of it and is asked only of the pairs the
  equalities allow. Both front ends build this shape; `ai_join` is the
  shorthand for a join over every pair.
- Builder: `.join(other, on=col("c.url") == col("e.id"))` followed by
  `.ai_filter(prompt over both tables)`. SQL: `JOIN evidence e ON
  c.evidence_wiki_url = e.id AND AI_FILTER(...)`, or the equality in
  `ON` and the AI predicate in `WHERE`. A `JOIN` with an equality and
  no AI predicate is refused.
- The session builds each join's pair table with an Arrow hash join
  over the key columns (`quail/runtime/pairs.py`) and ships it with
  the request as `PhysicalRequest.relations`. The projection pushdown
  loads the key columns as values.
- The planner prices a join with conditions by its pair count (the
  pair table's rows as a fraction of the cross product) instead of
  the cross product. `JoinStage` carries `equalities` and
  `pair_fraction`; `explain()` shows `on c.url = e.id` on the stage.
- The executor's `JoinAdmission` streams a partner list per anchor
  per stage (`anchor_partners`), so an anchor packs only its own
  pairs. An anchor with no pair at a stage settles without a chunk.
  The join output carries `anchor_partners`, and every consumer of
  answer rows (the Arrow answer tables, the coordinator's thinning
  and merging) reads bits through it.
- The multi-GPU path ships each worker its anchors' partner rows; the
  request backends submit only the listed pairs; the speed of light
  estimate counts and prices only the allowed pairs.
- QUAIL-B gains FEV-10: FEV-5 with `ON c.evidence_wiki_url = e.id`.
  Evidence ids are FEVER page names, so no corpus change was needed.
  quail-bench `JoinSpec.on` and the scorer apply the condition to the
  expected rows.

## Why

- A join predicate rarely needs the whole cross product. When the
  data already says which documents belong together, asking the
  model about the other pairs is wasted work and noise in the output.
  The condition is ordinary relational work, so it belongs in the
  query, priced by the planner, and enforced before any pair is
  packed.

## Numbers

- See [the pair join report](../2026-09-11-pair-join.md).
