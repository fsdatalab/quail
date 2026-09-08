# Readable query plans

`query.explain()` prints logical and physical operator trees with predicates,
projected columns, selected anchors, and estimated output rows. Physical filter
predicates appear in execution order. Shared inputs use numbered references.
Missing row estimates are marked unknown. Join evaluation counts are not
reported as output rows.

Quail plans show token budgets, KV rewind, retention for joins, anchor KV
reuse, and estimated join evaluations by default.

`verbose=True` includes node ids, ports, all runtime settings, and stage details.
`result.explain()` uses the same tree with measured output rows and time.
Its verbose form includes every recorded metric. Missing metrics are labeled
unavailable.

Planning and execution are unchanged. CPU tests cover estimates, limits,
predicate order, shared inputs, extensions, missing metrics, and refusal output.
