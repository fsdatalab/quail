# QUAIL-B split into its own repository, with Quail as one runner

Date: 2026-09-07.

Second step of moving QUAIL-B out of the engine repository. The
benchmark is now its own private repository,
https://github.com/fsdatalab/quail-bench, installed here as the
`quailb` package and pinned to a commit in `pyproject.toml`. It runs
no engine. `quail/bench/` is Quail's runner for it.

## What changed

- The `quailb` package holds the benchmark: `data.py` (document sets, pinned
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
  `quailb/` imports `quail`. The exact prompt text a predicate asks is
  `quailb/rendering.py`; the labels answer that text, and a test in
  the Quail runner checks that Quail sends the same text for every
  predicate.
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
- `quailb` is a dev dependency from the private repository. CI reads
  it with the `QUAILB_TOKEN` secret, a token with read access to that
  repository. Every Modal image that runs the benchmark ships the
  installed `quailb` next to `quail`: the same-GPU runner and the
  experiment cells that import it. The labeling pass runs from the
  benchmark repository.
- The benchmark repository carries the same `AGENTS.md` as this one,
  with `CLAUDE.md` a symlink to it, so the writing and check rules are
  shared.
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
| Where the benchmark lives | `quail/bench/` | `fsdatalab/quail-bench` (definition), `quail/bench/` (Quail runner) |
| `quail` imports inside the benchmark definition | session, catalog, planner, result relations, plan nodes | none |
| Query definitions | closures over a session | `QuerySpec` records |
| CPU tests collected here | 264 | 256, plus 33 in the benchmark repository |

No GPU run. The runner produces the same `run_suite` JSON as before.
The scoring reproduces the previous evaluator's numbers on the unit
tests, including the FEV-9 expected rows over a three claim corpus.

```sh
uv run ruff check quail quailb tests experiments reports tools
uv run python tools/check_long_strings.py
uv run vulture
uv run pytest -q
```
