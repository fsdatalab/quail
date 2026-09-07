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
  its own. `_modal_request` does the same when it copies local Arrow
  columns to the worker, so both paths use one definition.
- The result projection now raises a clear `CompileError` when a
  column was not loaded, instead of a bare `KeyError`.
- `Scan.explain_fields()` shows the kept columns.

Nothing changes in the Acero join. It already joined int32 document
index columns only, and values were attached at the end with a take
against the memory mapped token store. This change makes that column
set a planner decision instead of a session detail.

Validation: `tests/test_projection_pushdown.py` covers the rule on a
filter plus join plan, the document column case, idempotence, the
explain fields, and two end-to-end session queries (`SELECT r.stars,
r.id` loads exactly those two columns; `SELECT r.review` and
`SELECT *` still return the right values). No GPU run; this is a
planning change with no effect on model work.

```sh
uv run ruff check quail tests experiments reports tools
uv run python tools/check_long_strings.py
uv run vulture
uv run pytest -q
```
