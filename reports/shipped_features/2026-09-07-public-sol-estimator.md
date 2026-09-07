# Public SoL estimator, engine-side regret accounting, price in specs

Date: 2026-09-07.

First step of moving QUAIL-B out of the engine repository. Everything
the benchmark used to reach into the engine for is now either reported
by the engine or exposed as public API.

## What changed

- `quail.speed_of_light_estimate(query, answer)` is the public speed of
  light estimator. It takes a query built through a session and an
  `answer(prompt, assignment)` callable giving the exact answer of one
  prompt for one row assignment, and returns seconds, fresh tokens,
  dollars, per stage records, and its assumptions. The filter pass,
  the exact left deep join search, and the survivor computation moved
  from `reports/make_sol_quailb.py` into `quail/planner/estimate.py`
  and `quail/planner/live_rows.py` (formerly `quail/bench/sol_dp.py`).
  The script now only reads the corpora and labels, wraps the labels as
  the `answer` callable, and writes the same JSON layout as before. Its
  unused production planner simulation and the `anchor_tokens_both_ways`
  field went away.
- Every `QueryResult.report` carries `shared_prefix_tokens`,
  `cross_row_cached_tokens`, and `regret_distinct_tokens` next to
  `regret_tokens`. `Query.finish()` computes them from the token stores
  it already holds (`quail/runtime/prefixes.py`). The benchmark copies
  the numbers; `add_prefix_metrics` is gone.
- `BenchmarkEvaluator.evaluate()` scores the answer relations the engine
  returned instead of walking plan nodes to find them. It no longer
  imports `PackedFilter`, `RequestExecution`, or `expected_join_stages`.
  `BenchmarkEvaluator.answer(prompt, assignment)` is public and is the
  `answer` callable the estimator wants.
- `H100_USD_PER_HOUR` and `H100_PRICE_SOURCE` live in `quail.specs`,
  next to the device they price. `DeviceSpec` has `usd_per_hour` and
  `price_source` fields.
- `quail` exports `ColumnRef`, `SHARED_PRE`, `bind_prompt`,
  `bind_join_prompt`, `render_join_prompt_text`, and `true_false_ids`.
  The labeling pass uses these to ask the reference model the exact
  question the engine asks. `true_false_ids` moved to `quail.logical`.
- `Query.token_inputs()` tokenizes the scanned columns once; planning
  and the estimate share the stores.

## Why

The benchmark imported plan node classes, the executor's answer token
lookup, and a report script's private search. None of that survives a
move to another repository. After this change `quail/bench/` uses the
public API plus `collect_operators`, the result relation helpers, and
the worker image builder, and the SoL line on the plots is Quail's estimate, called through one
function any other engine could replace with its own.

## Before and after

| | Before | After |
| --- | --- | --- |
| Engine modules imported by `quail/bench/` beyond the public API | 12, including `physical`, `executor.loop`, `backends.quail`, `runtime.tokens` | 4: `planner` (`collect_operators`), `runtime.result`, `runtime.worker`, `specs` |
| SoL search code | 1,155 line script | 566 line planner module, 449 line script |
| CPU tests collected | 259 | 264 |

No GPU run. The estimator is the same arithmetic as the script it
replaces; the unit tests check the filter work by hand on a three
document corpus and the join search against the anchor choices. The
saved estimates on the volume were not regenerated.

```sh
uv run ruff check quail tests experiments reports tools
uv run python tools/check_long_strings.py
uv run vulture
uv run pytest -q
```
