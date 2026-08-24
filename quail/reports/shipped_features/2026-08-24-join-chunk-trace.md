# Per-chunk trace for run_join

`run_join` now takes the same optional `trace` list `run_filter`
already had: one dict per launched chunk with the chunk's stage,
token count, and its (anchor, start, end, carried) groups, appended
in launch order so it stays aligned with the CUDA-event spans even
when a gate prefetch builds chunks out of order. `run_filter`'s trace
dicts also gained a `pieces` field with the (doc, stage, fresh)
groups.

Why: the packing sweep (issue #25, report
`2026-08-24-packing-sweep.md`) fits cost constants per chunk, which
needs each chunk's exact composition next to its GPU time. Before
this, join chunk composition was not observable from outside the
loop.

No behavior change when `trace` is not passed; no measured cost when
it is (a list append per chunk).
