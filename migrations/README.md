# Migrations

One-shot scripts that rewrite data already on the `quail-results`
Modal volume, so a change to identity or layout does not force
regenerating it.

Add one when a change moves a content-addressed id without changing
the data behind it. The usual case is a field leaving `JUDGE_SPEC` or
`predicate_payload`: those feed `label_set_id`, which is both a
directory name and a stored column on every label row, so the id moves
even though every answer stays the same.

Rules:

- Report first, write on `--apply`. A migration that cannot be run
  read-only is not finished.
- Copy, never move. Leave the old directories on the volume; a
  superseded collection stays readable under its own id.
- Verify before writing the manifest. Compare row counts, TRUE counts
  and source counts against the manifest you read from, and fail loudly
  on any difference.
- Keep the script after it runs. It records what happened to the data
  and is the starting point for the next one.

Run from the repository root, and keep the function call id in a tee
file:

    uv run modal run migrations/<name>.py 2>&1 | tee results/<name>.log

| script | ran | what it did |
|---|---|---|
| `rehash_judge_identity.py` | 2026-08-26 | Moved 15 filter label sets, 17,057 labels, to new `label_set_id`s after the scheduler capacity knobs left `JUDGE_SPEC`. No model calls. |
| `join_relabel_diff.py` | 2026-08-26 | Checked what the 8 join predicates' forced relabel actually changed: 379 of 233,644 answers, 0.16%. Reads two collections, writes nothing to them. |

A migration that only reads and reports, like `join_relabel_diff.py`,
belongs here too: it answers whether a forced relabel was worth its
cost, which is a question about the volume's data, not about the
engine.
