# Post filter join planning and KV retention

Quail now runs every filter chain before it fixes the left deep join
plan. The join DP uses the actual filter survivors, document lengths,
and current KV contents. It searches join order and anchor choice
together. Execution does not change that plan after a join starts.

Passing filter documents can remain in GPU KV for a later join. Active
KV is pinned. Retained KV is evictable. Failed documents and documents
past their last planned use are freed immediately. There is no CPU spill
or offload path.

When active work needs more pages, the arena runs an exact small DP. It
chooses the retained document set with the least counted recomputation
work that frees enough pages. The value uses model dimensions and the
published H100! limits. It does not use fitted calibration constants.

The one H100! check retained all 7 filter survivors and reused all 7 at
the join with 0 evictions. See
`reports/2026-08-26-filter-join-kv-retention.md`.
