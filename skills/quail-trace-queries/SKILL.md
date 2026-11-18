---
name: quail-trace-queries
description: Write accurate, low-cost Quail SQL queries over agent traces. Use when modeling sessions, messages, tool calls, and tool results for Quail or writing queries with AI.IF, AI.SCORE, AI.CLASSIFY, AI.EXTRACT, or AI.MAP.
---

# Quail queries over agent traces

Quail runs relational SQL on the CPU in DataFusion and sends only the remaining
rows and columns to a GPU model. GPU time scales with the rows reaching the
model, the input tokens per row, and the output tokens generated per row.
Preview every query plan with `query.explain()`, cut rows and text with SQL
first, and pick the cheapest AI function able to answer the question.

## 1. Choose an accurate model for the GPU

Start with Qwen3.8 27B, Quail's most accurate general-purpose model, unless
your instructions request another model. Select the Qwen3.8 27B checkpoint
matching your GPU, or `qwen3-32b-fp8` on an L40S:

| GPU | `device` | `model` | Weight precision |
| --- | --- | --- | --- |
| NVIDIA B200 | `b200` | `qwen3.8-27b-nvfp4` | 4-bit floating-point (NVFP4) and 8-bit floating-point (FP8) |
| NVIDIA H100 SXM | `h100-sxm` | `qwen3.8-27b-fp8` | FP8 |
| NVIDIA RTX PRO 6000 Blackwell | `rtx-pro-6000-blackwell-server` | `qwen3.8-27b-fp8` | FP8 |
| NVIDIA L40S | `l40s` | `qwen3-32b-fp8` | FP8 (Qwen3.8 kernels do not support L40S) |

```python
config = quail.EngineConfig(model="qwen3.8-27b-nvfp4", device="b200")
```

Small 4B models (`qwen3-4b-fp8`, `qwen3.5-4b-fp8`) run faster per token yet
miss subtle trace failures such as reward hacking, silent test skipping, or
unspoken frustration. Switch to a 4B model only after verifying its accuracy
against Qwen3.8 27B or human labels on at least 100 rows (section 8).

## 2. Rank functions by GPU cost

AI functions differ widely in GPU cost per row. The table ranks operations from
cheapest to most expensive, with planner estimates on 1,000 rows of 165 tokens
on one H100 running `qwen3.8-27b-fp8`:

| Operation | GPU work per row | H100 estimate (1,000 rows) | Use |
| --- | --- | --- | --- |
| Plain SQL (`WHERE`, `JOIN`, `GROUP BY`, `LAG`, `regexp_count`) | 0 tokens (CPU only) | 0 s | Filter columns, match log text, deduplicate rows, and build turn views before AI calls. |
| `AI.IF`, `AI.SCORE` | 1 prefill pass (reads input once) + 1 token | 6.4 s | Filter on a yes-or-no question or rank rows by a 0-to-1 score. |
| `AI.CLASSIFY` | 1 prefill pass + constrained label scoring | 7.0 s | Assign each row to one label from a fixed list of at least two categories. |
| `AI.MAP(col, instruction, 20)` | 1 prefill pass + up to 20 decode steps (1 output token per step) | 8.7 s | Write short free text on filtered rows with an explicit `max_tokens` cap. |
| `AI.EXTRACT`, `AI.EXTRACT_SPAN` | 1 prefill pass + verification rounds per field (up to 64 tokens) | 9.9 s (1 field) | Copy exact substrings from text on filtered rows. |
| `AI.MAP(col, instruction)` | 1 prefill pass + up to 256 decode steps (default `max_tokens`) | 19.1 s | Avoid leaving `max_tokens` at 256 when a short phrase answers the question. |
| AI join (`JOIN ... ON AI.IF(...)`) | Evaluates candidate row pairs (`left_rows * right_rows` without filters) | 39.1 s (`1,000 x 50`) | Filter both tables first and include equality keys in `ON` whenever possible. |

Combine AI filters, classifications, and generation over one table inside a
single SQL query rather than splitting work across separate Python queries.
Quail groups adjacent AI calls into one model region, runs filters first, and
reuses KV, the model's cached attention state, so later operators do not read
the document again. Guard `AI.MAP` with `CASE WHEN` so generation runs only on
matching rows while reusing KV from `AI.CLASSIFY`:

```sql
SELECT u.turn_id,
       AI.CLASSIFY(u.turn_text, ['correction', 'no_issue', 'other_issue']) AS kind,
       CASE WHEN AI.CLASSIFY(u.turn_text, ['correction', 'no_issue', 'other_issue']) = 'other_issue'
            THEN AI.MAP(u.turn_text, 'Write a three-word failure label.', 16)
       END AS new_label
FROM user_replies u;
```

Quail evaluates the repeated `AI.CLASSIFY` call once when both calls match
character for character. Pass `{'selectivity': 0.02}` on a selective `AI.IF`
filter when far fewer than 20% of rows pass, because the planner assumes a 20%
pass rate by default when ordering filters.

## 3. Preview the plan before running the GPU

Call `query.explain()` before `query.run()` or `query.collect()`.
`query.explain()` tokenizes the input on the CPU and prints input token counts,
expected row counts, pass-rate assumptions, KV reuse, and estimated GPU time
without loading the model. Check `tokens=`, `est. rows` on `AI.EXTRACT` and
`AI.MAP`, and any `Cross Join` before spending GPU time. Call `query.run()`
when tuning a query so `result.explain()` and `result.report` record measured
pass rates, fresh tokens, cached tokens, and GPU cost (`usd_per_query`).

## 4. Split traces into narrow tables and single-column views

Store trace records in four tables so each query reads only the columns needed
for its question:

- `sessions`: one row per session (`session_id`, `agent`, `model`, `outcome`, `created_at`).
- `messages`: one row per user prompt or agent reply (`turn_id`, `session_id`, `turn_number`, `speaker`, `content`).
- `tool_calls`: one row per invocation (`tool_call_id`, `session_id`, `turn_number`, `tool_name`, `arguments`).
- `tool_results`: one row per output (`tool_call_id`, `exit_code`, `output`).

User and agent messages stay short and answer most behavioral questions,
whereas tool outputs hold kilobytes of logs. Query `messages` first, and read
`tool_results` only when message text cannot answer the question.

Build single-column turn views once with `CREATE VIEW` so every AI function
reads one text column and shares its KV:

```python
session.sql("""
CREATE VIEW user_replies AS
WITH ordered AS (
  SELECT m.turn_id, m.session_id, m.speaker, m.content,
         LAG(m.content) OVER (PARTITION BY m.session_id ORDER BY m.turn_number) AS previous_content,
         LAG(m.speaker) OVER (PARTITION BY m.session_id ORDER BY m.turn_number) AS previous_speaker
  FROM messages m
)
SELECT o.turn_id, o.session_id,
       CONCAT('Agent: ', LEFT(o.previous_content, 2000), '\n\nUser: ', LEFT(o.content, 2000)) AS turn_text
FROM ordered o
WHERE o.speaker = 'user' AND o.previous_speaker = 'agent'
""").run()
```

Create matching views for `agent_replies` (each agent reply paired with the
preceding user message), `last_replies` (each session's final agent reply), and
`failed_test_replies` (each failing test tail paired with the agent's next
reply) when questions need them.

## 5. Cut rows and tokens in SQL before AI calls

DataFusion pushes stored-column filters below model regions, so filtered rows
never reach the GPU:

- Filter structured columns first: dates, `agent`, `outcome`, `speaker`,
  `tool_name`, `exit_code`, and `length(m.content)`.
- Screen text on the CPU with `regexp_count`:
  `regexp_count(right(r.output, 500), '(--- FAIL|FAILED|Traceback| [1-9][0-9]* failed)') > 0`
  isolates failing test runs, and
  `regexp_count(u.turn_text, '(?i)(revert|undo|wrong|stop)') > 0` narrows
  candidate user corrections before `AI.IF` confirms them.
- Slice long text in SQL: `left(m.content, 4000)` bounds message length, and
  `right(r.output, 2000)` keeps the tail of a log where failures appear.
- Deduplicate repeated errors with `GROUP BY` before calling the model, then
  join answers back to all occurrences.
- Sample deterministically on hexadecimal IDs when estimating a rate:
  `right(s.session_id, 1) IN ('0', '8')` keeps 2 of 16 (12.5%) sessions.

## 6. Write short, specific prompts

Judge one turn and its immediate neighbor rather than passing a whole session.
Quail places the document `{0}` at the start of each prompt so multiple AI
calls on the same column share KV, and moves any template text written before
`{0}` to after `{0}`. Keep prompt templates short and omit multi-example
few-shot blocks, because template tokens run after every row's document rather
than sharing a cross-row prefix.

Prefer `AI.CLASSIFY` with mutually exclusive categories, including `'none'` or
`'other'`, over a vague yes-or-no question. State explicit exclusions for
near-miss cases, such as `'Routine git push or token setup does not count.'`

Common trace questions map directly to SQL aggregates and `GROUP BY`:

| Question | SQL pattern |
| --- | --- |
| Failure rate across turns | `SELECT AVG(CASE WHEN AI.IF(...) THEN 1.0 ELSE 0.0 END) FROM agent_replies r` |
| Outcome gap when a failure occurs | `GROUP BY has_failure` over session-level `BOOL_OR(AI.IF(...))` joined to `sessions` |
| First turn where a failure starts | `SELECT r.session_id, MIN(r.turn_number) FROM agent_replies r WHERE AI.IF(...) GROUP BY r.session_id` |
| Comparison across tools or versions | `GROUP BY t.tool_name` or `GROUP BY s.agent` |

## 7. Follow Quail SQL syntax rules

- Qualify column names with a table alias (such as `m.turn_text`), especially
  inside `AI.EXTRACT(m.turn_text, ...)`, because Snowflake SQL parses
  `EXTRACT(ident ...)` with an unqualified identifier as a date expression.
- `session.sql()` uses the Snowflake dialect by default. Snowflake
  `REGEXP_LIKE` and `RLIKE` match the entire string; use
  `regexp_count(col, pattern) > 0` or `ILIKE '%term%'` to match substrings, or
  pass `dialect="bq"` for unanchored `REGEXP_CONTAINS`.
- Write option and field objects as `{'selectivity': 0.05}` in the default
  Snowflake dialect, and as `STRUCT(0.05 AS selectivity)` when `dialect="bq"`.
- Pass one text column and a string literal instruction to
  `AI.MAP(col, 'instruction', max_tokens)`; combine multiple columns in a view
  or common table expression (CTE) first.

## 8. Validate every judge against labeled rows

Read 15 to 20 positive rows and 15 to 20 negative rows before trusting a count.
Measure true positive rate (TPR), the share of true positives the judge
catches, and true negative rate (TNR), the share of true negatives the judge
rejects, on 100 to 200 labeled examples:

```sql
WITH judged AS (
  SELECT h.label AS truth,
         AI.IF(PROMPT('Does the user correct a mistake by the agent? {0}', h.turn_text)) AS predicted
  FROM human_labels h
)
SELECT AVG(CASE WHEN j.predicted THEN 1.0 ELSE 0.0 END) FILTER (WHERE j.truth) AS tpr,
       AVG(CASE WHEN NOT j.predicted THEN 1.0 ELSE 0.0 END) FILTER (WHERE NOT j.truth) AS tnr
FROM judged j;
```
