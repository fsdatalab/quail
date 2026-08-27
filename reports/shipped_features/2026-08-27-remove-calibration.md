# Removed: calibration (the measured constants a, a2)

## What changed

The calibration machinery is gone. It measured two constants per
(model, device) pair — `a`, wall seconds per fresh token, and `a2`,
the quadratic attention coefficient — but nothing in the engine ever
consumed them:

- Every planner decision (filter order, join order, anchor choice,
  sharding) compares token counts at the same serving rate, so the
  rate cancels out. No decision needed a wall-clock constant.
- `plan_query` loaded the constants only to stamp a
  `calibration_source` string into the plan. `explain()` and the
  session report echoed that string and nothing else read it.
- `derived_table` used `a` only for a serving-rate row that one
  test asserted on.
- Nothing at all read `calibration/channels.json`.

Removed:

- `quail/planner/calibration.py`: the `Calibration` struct, loading,
  spec-ratio scaling from the anchor pair, the affine fit, and the
  commit path.
- `quail/planner/calibrate.py`: the GPU length-sweep measure step,
  the host copy channel probe, and `resolve_pair`.
- `quail/runtime/calibrate.py`: the Modal entry.
- `quail/calibration/`: the two committed pair files and
  `channels.json`.
- `results/calibrate.json`.
- The `calibration_source` field on `PhysicalPlan`, the
  `calibration` argument of `plan_query`, the calibration line in
  `explain()`, and the `calibration` key in the session report.
- The `quail/calibration` image mounts in four GPU cells. The engine
  no longer ships any data file into the Modal image.
- The calibration tests. `derived_table` stays, without the
  serving-rate row.

## Why

The constants were measured, stored, and displayed, but no code path
priced anything with them. The planner was built so its decisions
survive without a wall-clock rate, and it says so in its docstring:
"No wall prediction." Keeping a measurement pipeline whose output is
only a provenance string is maintenance cost with no return.

## Before/after numbers

- No behavior change: plans, payloads, and execution are identical.
  Only the provenance string is gone from `explain()` and the
  session report.
- 599 lines removed, 32 added, across 19 files.
