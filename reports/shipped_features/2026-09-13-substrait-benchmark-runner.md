# The QUAIL-B runner reads Substrait plans

Date: 2026-09-13.

quail-bench 0.4.0 defines every benchmark query as a Substrait plan.
Substrait is a standard protocol buffer format for relational query
plans. The alias-based `QuerySpec` fields Quail's runner read
(`aliases`, `joins`, `select`, the per-template selectivity methods)
are gone, and the answers an adapter returns are keyed by the plan's
operator ids (`filter-1`, `join-1`) instead of by alias and written
position. The pin in `pyproject.toml` moves from `63ff9b9` to
`4c356d8`, and `quail/bench/` is rewritten as the adapter that
contract asks for.

## What changed

- `quail/bench/substrait.py` is new. `read_plan(plan)` walks the
  Substrait relation tree: `ReadRel` scans with an alias hint,
  `FilterRel` calls to `ai_filter`, inner `JoinRel` calls to `ai_join`
  with ordinary `equal` conditions beside them, and the `ProjectRel`
  of id columns under the root. It resolves function anchors through
  the plan's extension declarations and checks that the AI functions
  come from the `org.fsdatalab.quail_b` extension. The result is a
  `QueryPlan`: relations in scan order, filters and joins in operator
  id order, and the projection. `build_query(session, plan,
  selectivity, order)` turns that into a Quail query with the builder.
- `quail/bench/quailb.py` is shorter. `build_query(session, spec)`
  reads the spec's plan and applies quail-bench's fixed selectivity
  estimates by prompt; a query with an estimate for every predicate is
  ordered by cost, any other in written order, as before.
  `run_output` and `prompt_pieces` key filter answers, join answers,
  and token pieces by operator id. `register_tables(session, dir)`
  registers every Parquet file in a directory; it replaces
  `register_sets` and `register_privacy_sets`, and `queries(session)`
  lists the queries whose tables are registered, so the PRIV queries
  appear when their tables do. `run_query`, `run_suite`, and the
  command line are unchanged in behavior.
- `quail/bench/restate.py` is deleted. It added prompt pieces to runs
  saved before Quail reported them. Every run since reports them, and
  quail-bench 0.4.0 saves runs in a new format (operator id keys,
  `schema_version` 2) that `quail-b report` requires, so the old runs
  it targeted cannot be rescored by quail-bench either way. The
  comparison plot script reads those runs' saved `measurements.parquet`
  directly and still works.
- `quail/bench/quailb_parallel.py`: the Quail-and-vLLM family function
  and the SGLang family function share one body, `_run_family`, and
  both write their family result under `families/` with the same
  keys. `ensure_data` finds a query's tables through `read_plan`.
- `reports/make_quailb_eval_plots.py` is deleted: no note referenced
  it, and it read a result format no runner writes.
- Three one-off comparison cells are deleted:
  `experiments/cells/fixed_join_plan.py`, `shared_kv_retention.py`, and
  `join_continuous_batching.py`. Each ran a baseline checkout against the
  current one for a past note and saved answer tables in the old
  alias-and-position format. `reports/score_shared_kv_retention.py`,
  which read that format, goes with them. The notes they produced keep
  their numbers and volume paths and now name the commit the scripts
  live at in the history.
- `reports/make_quailb_comparison_plots.py` no longer reads the saved
  September 5 suite (the runner's format before quail-bench). That
  path filled pipelined vLLM's FEV-1 to FEV-9, which the September 12
  run's FEVER container did not produce. Those 9 of 99 cells are now
  marked missing: an x below the axis in the plots and a "not run" row
  in the tables. The "Quail before and after this branch" section,
  which existed only through that suite, is gone with it.
  `reports/quailb-comparison.md`, `plots/quailb_main.pdf`, and
  `plots/quailb_fev.pdf` are regenerated from the same volume files;
  the other dataset figures do not change.
- Two per-pair Python loops on Quail's side of scoring are now Arrow
  and numpy work: the runner's row-index-to-id mapping (`pc.take`) and
  the Quail backend's join answer table assembly in
  `quail/execution.py`, which now visits anchors in Python and pairs in
  numpy. Timed on a synthetic 2,000,000-pair join: 0.7 s to 0.1 s and
  0.9 s to 0.1 s. The September 12 run evaluated 3.15 million pairs
  in total at scale 0.1, so this matters at scale 1.0, where one join
  has 20.7 million pairs. The remaining per-pair loops (answer
  agreement, the KV minimum) and the label dictionaries are in
  quail-bench.
- The SoL script and the comparison plot script read query structure
  from `read_plan`. Experiments call `register_tables`.

## Checks

No GPU run was made. The change is to the adapter's input format, not
to what Quail executes, and that was checked on the CPU:

- Prediction: the Quail plan of every query is the same whether it is
  built from the old alias-based spec or from the Substrait plan.
- Result: for all 35 queries (33 published plus PRIV-1 and PRIV-2), on
  the `quail` and `stock_vllm` backends, `query.explain()` and
  `query.plan()` on the stand-in tables of `tests/test_quailb.py` are
  byte for byte identical before and after the change, apart from a
  note about background tokenization timing. 70 plans compared, 0
  differences.
- `tests/test_bench_runner.py` decodes IMDB-4 and FEV-10 (its
  equality condition) into the expected relations and operators,
  decodes every published plan, rejects an `ai_filter` from another
  extension, checks the answer tables and prompt pieces come back
  keyed `filter-1` and `join-1`, scores them with quail-bench's
  `evaluate` and `token_metrics`, and checks FEV-9's prompts and labels
  as before. `tests/test_quailb.py` builds and plans all 35 queries on
  four backends.
- `ruff`, `tools/check_long_strings.py`, `vulture`, and `pytest` pass.

The first GPU run on the new quail-bench will write its run directory
in the new format; the numbers in `reports/quailb-comparison.md` stay
current because the executed plans did not change.
