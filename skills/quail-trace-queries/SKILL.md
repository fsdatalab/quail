---
name: quail-trace-queries
description: Write efficient Quail queries over agent traces. Use when modeling trace data for Quail or writing queries with AI.IF, AI.SCORE, AI.CLASSIFY, AI.EXTRACT, or AI.MAP over messages and tool calls.
---

# Efficient Quail queries over agent traces

A query costs the tokens the model processes. Every rule below cuts tokens while
keeping answers accurate: fewer rows reach the model, each row is shorter, and
the model writes less. Check every query plan with `query.explain()` before
running on the GPU.

## Choose a model and a GPU

Start with Qwen3.8 27B, the most accurate model Quail runs, unless your task
asks for another model. Pick the Qwen3.8 27B version for your GPU, or
`qwen3-32b-fp8` on an L40S:

| GPU | `device` | `model` | Weight precision |
| --- | --- | --- | --- |
| B200 | `b200` | `qwen3.8-27b-nvfp4` | 4-bit floating-point (NVFP4) on B200 FP4 tensor cores |
| H100 | `h100-sxm` | `qwen3.8-27b-fp8` | 8-bit floating-point (FP8) |
| RTX PRO 6000 Blackwell | `rtx-pro-6000-blackwell-server` | `qwen3.8-27b-fp8` | FP8 |
| L40S | `l40s` | `qwen3-32b-fp8` | FP8 (L40S cannot run Qwen3.8) |

```python
config = quail.EngineConfig(model="qwen3.8-27b-nvfp4", device="b200")
```

Smaller 4B models (`qwen3-4b-fp8`, `qwen3.5-4b-fp8`) cost less per token but
miss subtle trace failures such as reward hacking, skipped tests, or user
frustration. Check a 4B model's answers against Qwen3.8 27B or human labels on
at least 100 rows (section 9) before trusting its counts.

## 1. Store messages and tool calls in separate tables

Store each record kind in its own table so a query reads only the text the
question needs:

- `messages`: one row per user prompt or agent reply (`turn_id`, `session_id`, `turn_number`, `speaker`, `content`).
- `tool_calls`: one row per call (`tool_call_id`, `session_id`, `turn_number`, `tool_name`, `command`, `content`).
- `tool_results`: one row per result (`tool_call_id`, `exit_code`, `output`), joined to `tool_calls` by `tool_call_id`.
- `sessions`: one row per session (`session_id`, `agent`, `outcome`, and metadata such as git attribution).

## 2. Build context views once, with plain SQL

Most judgments need one message and the message before it. Build a view with one
combined text column once, with no model calls, so `AI.IF`, `AI.CLASSIFY`,
`AI.EXTRACT`, and `AI.MAP` can all read the same column and share KV, the
model's saved state for the text:

```python
session.sql("""
CREATE VIEW user_replies AS
WITH ordered AS (
  SELECT m.turn_id, m.session_id, m.speaker, m.content,
         LAG(m.content) OVER (PARTITION BY m.session_id ORDER BY m.turn_number) AS previous_message,
         LAG(m.speaker) OVER (PARTITION BY m.session_id ORDER BY m.turn_number) AS previous_speaker
  FROM messages m)
SELECT o.turn_id, o.session_id,
       CONCAT('Agent: ', LEFT(o.previous_message, 2000), '\n\nUser: ', LEFT(o.content, 2000)) AS turn_text
FROM ordered o
WHERE o.speaker = 'user' AND o.previous_speaker = 'agent'
""").run()
```

Create matching views when questions need them: `agent_replies` (each agent
reply with the message before it), `last_replies` (each session's last agent
message), and `test_replies` (each test run's output tail with the agent's next
reply).

## 3. Query agent and user messages first

- Agent and user messages answer most questions and stay short (median 108 to
  138 characters in SWE-chat).
- Tool calls and tool results are much longer (90th percentile 4,819
  characters), so read `tool_results` only when `messages` cannot answer the
  question.
- Check tool output with SQL where possible: a regular expression on the end of
  a log finds failing tests without calling the model.

## 4. Filter with SQL before every AI call

DataFusion pushes SQL filters below model regions, so filtered rows cost zero
GPU tokens:

- Filter on columns first: dates, `agent`, `outcome`, `speaker`, `tool_name`,
  and `length(m.content)`.
- Match text with `regexp_count` before the model reads it:
  `regexp_count(right(r.output, 500), '(--- FAIL|FAILED|Traceback| [1-9][0-9]* failed)') > 0`
  keeps only failing test runs, and
  `regexp_count(u.turn_text, '(?i)(revert|undo|wrong)') > 0` finds likely
  corrections for `AI.IF` to confirm.
- Remove duplicate text with `GROUP BY` before calling the model.
- Sample by id when a rate is enough: `right(s.session_id, 1) IN ('0', '8')`
  keeps 1 in 8 sessions because hexadecimal ids end in one of 16 digits.
- Cut text in SQL: `left(m.content, 4000)` caps messages, and
  `right(r.output, 2000)` keeps the end of a log where errors appear.

## 5. Rank AI functions by GPU cost and keep pipelines in one query

AI functions differ widely in GPU work per row. The table ranks operations from
cheapest to most expensive, with `query.explain()` time estimates on 1,000 rows
of 165 tokens on one H100 running `qwen3.8-27b-fp8`:

| Operation | GPU work per row | H100 estimate (1,000 rows) | When to use |
| --- | --- | --- | --- |
| Plain SQL (`WHERE`, `JOIN`, `GROUP BY`, `LAG`, `regexp_count`) | 0 tokens (CPU only) | 0 s | Filter columns, match logs, deduplicate rows, and build views first. |
| `AI.IF`, `AI.SCORE` | Reads input once + 1 answer token | 6.4 s | Filter on a yes-or-no question or rank rows by a 0-to-1 score. |
| `AI.CLASSIFY` | Reads input once + scores fixed label tokens | 7.0 s | Pick one label from a list of at least two categories. |
| `AI.MAP(col, instruction, 20)` | Reads input once + generates up to 20 tokens one by one | 8.7 s | Write short free text on filtered rows with a tight `max_tokens` limit. |
| `AI.EXTRACT`, `AI.EXTRACT_SPAN` | Reads input once + copies spans in verification rounds (up to 64 tokens per field) | 9.9 s (1 field) | Copy exact phrases from text on filtered rows. |
| `AI.MAP(col, instruction)` | Reads input once + generates up to 256 tokens (default `max_tokens`) | 19.1 s | Avoid leaving `max_tokens` at 256 for short answers. |
| AI join (`JOIN ... ON AI.IF(...)`) | Tests candidate pairs (`left_rows * right_rows` without filters) | 39.1 s (`1,000 x 50`) | Filter both tables first and add equality keys in `ON` where possible. |

Keep `AI.IF`, `AI.CLASSIFY`, and `AI.MAP` over one table inside a single SQL
query instead of running separate Python queries. Quail runs cheap filters first
and reuses KV so later operators do not read the document again. Guard free-text
generation with `CASE WHEN` so `AI.MAP` runs only on `'other new issue'` rows
while reusing KV from `AI.CLASSIFY`:

```sql
SELECT r.turn_id,
       AI.CLASSIFY(r.turn_text, ['claims done without testing', 'no issue', 'other new issue']) AS code,
       CASE WHEN AI.CLASSIFY(r.turn_text, ['claims done without testing', 'no issue', 'other new issue']) = 'other new issue'
            THEN AI.MAP(r.turn_text, 'Write a three-word code naming the failure.', 20)
       END AS new_code
FROM agent_replies r;
```

Write repeated `AI.CLASSIFY` calls identically so Quail evaluates them once.
Pass `{'selectivity': 0.02}` on a rare `AI.IF` filter when far fewer than 20% of
rows pass, because the planner assumes 20% by default when ordering filters.

## 6. Inspect the plan before using the GPU

Call `query.explain()` before `query.run()` or `query.collect()`.
`query.explain()` tokenizes inputs on the CPU and prints token counts, expected
rows, assumed pass rates, KV reuse, and estimated GPU seconds without loading
the model. Check `tokens=`, `est. rows` on `AI.EXTRACT` and `AI.MAP`, and any
`Cross Join` before running. Use `result = query.run()` when tuning a query so
`result.explain()` and `result.report` show measured pass rates, fresh tokens,
cached tokens, and `usd_per_query`.

## 7. Judge one message, with short prompts

- Give the model one turn with its neighbor, not a whole session transcript.
- Quail places `{0}` at the start of every prompt so multiple AI calls on one
  column share KV, and moves any template text before `{0}` to after `{0}`. Keep
  instructions short and skip long few-shot examples, because template tokens
  run after every row's document instead of sharing a prefix across rows.
- Prefer `AI.CLASSIFY` with contrasting labels, including `'no issue'` or
  `'other'`, over a vague yes-or-no question.
- Name what does not count: `'Merging, pushing, or setting tokens does not
  count.'`

## 8. Common query forms

| Form | Question | SQL pattern |
| --- | --- | --- |
| Frequency | How often does the failure happen? | `SELECT COUNT(*) FROM agent_replies r WHERE AI.IF(...)` |
| Impact | Do sessions with the failure keep less agent code? | `GROUP BY has_failure` over session `BOOL_OR(AI.IF(...))` with `AVG(s.agent_percentage)` |
| Onset | At which turn does the failure start? | `SELECT r.session_id, MIN(r.turn_number) FROM agent_replies r WHERE AI.IF(...) GROUP BY r.session_id` |
| Location | Which tool fails most? | `GROUP BY t.tool_name` |
| Comparison | Did a prompt change reduce the failure? | `GROUP BY s.prompt_version` |

## 9. Check every judge

Read 15 to 20 positive rows and 15 to 20 negative rows before trusting a count.
Label 100 to 200 examples per failure mode, with at least 50 passes and 50
failures, and measure true positive rate (TPR), the share of true positives the
judge catches, and true negative rate (TNR), the share of true negatives the
judge rejects:

```sql
WITH judged AS (
  SELECT h.label AS human,
         AI.IF(PROMPT('Does the agent claim completion without testing? {0}', h.turn_text)) AS judge
  FROM human_labels h)
SELECT AVG(CASE WHEN j.judge THEN 1.0 ELSE 0.0 END) FILTER (WHERE j.human)     AS tpr,
       AVG(CASE WHEN j.judge THEN 0.0 ELSE 1.0 END) FILTER (WHERE NOT j.human) AS tnr
FROM judged j;
```

## 10. SQL dialect and syntax rules

- Qualify columns with a table alias (such as `r.turn_text`), especially inside
  `AI.EXTRACT(r.turn_text, ...)`, because Snowflake SQL parses
  `EXTRACT(ident ...)` with a bare column name as a date expression.
- `session.sql()` uses the Snowflake dialect by default. Snowflake
  `REGEXP_LIKE` and `RLIKE` match the whole string; write
  `regexp_count(col, pattern) > 0` or `ILIKE '%term%'` to match substrings, or
  pass `dialect="bq"` for unanchored `REGEXP_CONTAINS`.
- Write option and question objects as `{'selectivity': 0.05}` in the default
  Snowflake dialect, and as `STRUCT(0.05 AS selectivity)` when `dialect="bq"`.
- `AI.MAP(col, 'instruction', max_tokens)` takes one text column and a string
  literal instruction; combine several columns in a view or common table
  expression (CTE), a named `WITH` subquery, first.
