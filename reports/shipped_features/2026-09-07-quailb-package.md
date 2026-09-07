# QUAIL-B split into its own package, with Quail as one runner

Date: 2026-09-07.

Second step of moving QUAIL-B out of the engine repository. The
benchmark is now a top-level package, `quailb/`, that runs no engine.
`quail/bench/` is Quail's runner for it.

## What changed

- `quailb/` holds the benchmark: `data.py` (document sets, pinned
  sources, sampling, corpus identity), `prompts.py`, `queries.py`,
  `labels.py` (saved reference labels and collections), `judge_pass.py`
  (the labeling pass on Modal), `scoring.py`, and the README.
- The 32 queries are data. `quailb.queries.QUERIES` is a tuple of
  `QuerySpec` records: aliases with their tables, text columns, and
  filter templates in written order; binary joins with their aliases
  in placeholder order; and the select list. The two PrivacyPolicies
  queries are `PRIVACY_QUERIES`. Before, the queries existed only as
  closures over a Quail session.
- Scoring takes a `RunOutput`: the engine's filter answers keyed by
  (alias, written position), its join answers keyed by written
  position, and its final rows, all in the benchmark's own ids. The
  expected rows come from the labels through pyarrow joins. Nothing in
  `quailb/` imports `quail`, except the labeling pass, which renders
  the exact prompt text the engine sends through Quail's public prompt
  helpers.
- `quail/bench/quailb.py` is the runner. `build_query(session, spec)`
  turns one spec into a Quail query, `run_output(result, spec, corpus)`
  turns a `QueryResult` into a `RunOutput`, and `answer_oracle`
  wraps the labels as the callable `quail.speed_of_light_estimate`
  takes. `queries(session)` keeps the old id to (description, builder)
  shape for the experiment scripts. `run_suite` and the same-GPU Modal
  runner are unchanged in what they measure and save.
- `rows_from_answers` derives final rows from saved answers, for runs
  that kept their answers and not their rows; the shared KV retention
  scorer uses it.
- Packaging, ruff, vulture, the long string check, and CI cover
  `quailb` alongside `quail`. Every Modal image that runs the benchmark
  ships `quailb` next to `quail`: the same-GPU runner, the labeling
  pass, and the experiment cells that import it.
- The runner's `--report` option looked for the plot script under
  `quail/reports/`; it now looks under the repository's `reports/`.

## Why

A benchmark like TPC-H or ClickBench is a specification plus data plus
scoring, and each system brings its own runner. QUAIL-B's definition
was tangled with Quail's session API and result object, so nothing
could move. Now the definition is data and ids, the scoring is
pyarrow, and Quail's part is a runner in the engine repository.

## Before and after

| | Before | After |
| --- | --- | --- |
| Where the benchmark lives | `quail/bench/` | `quailb/` (definition), `quail/bench/` (Quail runner) |
| `quail` imports inside the benchmark definition | session, catalog, planner, result relations, plan nodes | none, except public prompt helpers in the labeling pass |
| Query definitions | closures over a session | `QuerySpec` records |
| CPU tests collected | 264 | 268 |

No GPU run. The runner produces the same `run_suite` JSON as before.
The scoring reproduces the previous evaluator's numbers on the unit
tests, including the FEV-9 expected rows over a three claim corpus.

```sh
uv run ruff check quail quailb tests experiments reports tools
uv run python tools/check_long_strings.py
uv run vulture
uv run pytest -q
```
