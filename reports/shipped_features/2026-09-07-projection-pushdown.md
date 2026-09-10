# Projection pushdown as a logical rule

- `Scan` has a new `columns` field: the source columns kept as values
  beside the tokenized document column. Its `output_schema()` now
  lists those columns, so filter and join schemas above it shrink
  with it.
- `quail/logical_rules.py` adds `ProjectionPushdown`, the first built
  in logical rule. It fires at the root Project, walks the tree for
  the columns the SELECT list returns, and rewrites each Scan with
  that set. The document column is kept as a value only when the
  query returns it; prompts read it as tokens. The rule is idempotent
  and leaves every other node unchanged.
- `Query.plan()` reads the pruned Scans to decide what each token
  store holds. Before, it recomputed the set from the root Project on
  its own. The same in-process path handles sessions inside Modal functions.
- The result projection now raises a clear `CompileError` when a
  column was not loaded, instead of a bare `KeyError`.
- `Scan.explain_fields()` shows the kept columns.
- The session token cache is keyed on the document column alone. Value
  columns live in their own memory mapped files (`ColumnStore`), keyed
  by provider content and column name. A query that returns columns
  not yet stored scans only those columns from the provider, in one
  pass, and reuses the token file. Before, the cache key included the
  value column set, so two queries with different SELECT lists over the
  same documents tokenized the corpus twice. On the first query the
  tokens and the value columns are written in the same pass over the
  source. A later scan for new value columns must return rows in the
  same order as the first; the session checks the row count and raises
  if it differs.

Two renames ride along:

- The physical `HashJoin` node is now `Recombine` (type name
  `quail.recombine`). It combines every join stage's true pairs with
  the survivor ids into result tuples. Acero still runs a hash join
  underneath, but the node's job in the graph is recombination, and
  the planner already named it `recombine`.
- `ExecutionLocation.CLIENT` is gone. `Project` and `Limit` are
  `COORDINATOR` nodes like `DocumentInput`, `Exchange`, and
  `Recombine`. The runner only ever separated `GPU_EXECUTOR` nodes
  from the rest, and the Modal worker ran every node, so the client
  label described nothing. Extension nodes that used `CLIENT` must
  switch to `COORDINATOR`.

`Recombine` is also skipped when it would do nothing. Filters always
run before the joins their table feeds, so with one full join and no
gate the join's true pairs are the result rows. Both planners now wire
that join's `join_answers` port straight into `Project`, and the
result projection drops the false rows itself. A query with several
full joins, or with an `EXISTS` or `NOT EXISTS` gate, still plans a
`Recombine`. `tests/test_recombine.py` checks both shapes on the Quail
and request backends and runs a single-join query end to end.

Nothing changes in the Acero join itself. It already joined int32 document
index columns only, and values were attached at the end with a take
against the memory mapped token store. This change makes that column
set a planner decision instead of a session detail.

Validation: `tests/test_projection_pushdown.py` covers the rule on a
filter plus join plan, the document column case, idempotence, the
explain fields, and three end-to-end session queries (`SELECT r.stars,
r.id` loads exactly those two columns; `SELECT r.review` and
`SELECT *` still return the right values; a second query with a
different SELECT list tokenizes no document again and adds only its
new column files). No GPU run; this is a
planning change with no effect on model work.

```sh
uv run ruff check quail tests experiments reports tools
uv run python tools/check_long_strings.py
uv run vulture
uv run pytest -q
```
