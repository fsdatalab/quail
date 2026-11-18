---
name: quail-trace-queries
description: Write efficient Quail queries over agent traces. Use when modeling trace data for Quail or writing queries with AI.IF, AI.CLASSIFY, AI.EXTRACT, or AI.MAP over messages and tool calls.
---

# Efficient Quail queries over agent traces

A query costs the tokens the model processes. Every rule below cuts tokens:
fewer rows reach the model, each row is shorter, and the model writes less.

## 1. Store messages and tool calls in separate tables

- `messages`: one row per user prompt or agent reply, with `session_id`, `turn_number`, `speaker`, `content`.
- `tool_calls`: one row per call, with `tool_name`, `command`, `content`.
- `tool_results`: one row per result, joined to its call by `tool_call_id`.
- `sessions`: one row per session, with outcome columns such as git attribution.

Separate tables let a query read only the kind of text the question needs.

## 2. Build context tables once, with plain SQL

Most judgments need one message and the message before it. Build that table
once, with no model calls, and register it, so every query stays short:

```sql
-- user_replies: each user reply with the agent message before it
WITH ordered AS (
  SELECT turn_id, session_id, speaker, turn_number, content,
         LAG(content) OVER (PARTITION BY session_id ORDER BY turn_number) AS previous_message,
         LAG(speaker) OVER (PARTITION BY session_id ORDER BY turn_number) AS previous_speaker
  FROM messages)
SELECT turn_id, session_id, previous_message, content
FROM ordered
WHERE speaker = 'user' AND previous_speaker = 'agent'
```

Other useful tables: `agent_replies` (each agent reply with the message before
it), `last_replies` (each session's last agent message), and `test_replies`
(each test run's output with the agent's next reply).

## 3. Query agent and user messages first

- Agent and user messages answer most questions, and they are short (median 108 to 138 characters in SWE-chat).
- Tool calls and tool results come next; results are long (90th percentile 4,819 characters), so read them only when the question needs them.
- Check tool output with SQL where you can: a regex on the end of a log finds failing tests without the model.

## 4. Filter with SQL before every AI call

- Filter on columns: dates, agent, outcome, speaker, tool name, `length(content)`.
- Remove duplicates with `GROUP BY` before the AI call.
- Sample by id when a rate is enough: `right(session_id, 1) IN ('0', '8')` keeps 1 in 8 sessions, because ids end in one of 16 hex digits.
- Cut text in SQL: `left(content, 4000)` for messages, `right(output, 2000)` for logs, where errors appear.

## 5. Prefer AI.IF and AI.CLASSIFY

- `AI.IF` and `AI.CLASSIFY` read the prompt and answer with a label, so their cost is mostly the input.
- `AI.EXTRACT` and `AI.MAP` generate text token by token, so each row costs more. Run them last, on the rows the other filters leave.
- Order a query so the cheapest step runs first: SQL, then `AI.IF` or `AI.CLASSIFY`, then `AI.EXTRACT` or `AI.MAP`.
- To write free text for only some rows, label first, then generate for the rows that need it:

```sql
-- step 1: label every reply
SELECT turn_id, AI.CLASSIFY(context, ['claims done without testing', 'no issue', 'other new issue']) AS code
FROM agent_replies WHERE session_id = '…';

-- step 2: write a new code only for the 'other new issue' rows
SELECT turn_id, AI.MAP(context, 'Write one short code naming the failure …', 30) AS code
FROM agent_replies WHERE turn_id IN ('…', '…');
```

## 6. Judge one message, with its context

- Give the model one message, not a whole session.
- Fill the prompt's placeholders with the message and its context. `PROMPT` puts the first column after the template into `{0}`, the next into `{1}`: `AI.IF(PROMPT('Agent: {0} User: {1} Is the user frustrated?', previous_message, content))`.
- Put the fixed instruction and few-shot examples before the row's text. Every row then starts with the same tokens, and Quail's prefix sharing reads them once.

## 7. Ask questions a model can answer

- `AI.CLASSIFY` with contrasting labels, including "other", works better than a vague yes-or-no question.
- Name what does not count: "Merging, pushing, or setting tokens does not count."
- Read 15 to 20 result rows before you trust a count.

## 8. Common query forms

| Form | Question | Pattern |
| --- | --- | --- |
| Frequency | How often does the failure happen? | `SELECT COUNT(*) … WHERE AI.IF(…)` |
| Impact | Do sessions with it keep less agent code? | `GROUP BY AI.IF(…)` with `AVG(agent_percentage)` |
| Onset | At which step does it start? | `LAG` over steps, then `MIN(step)` per session |
| Location | Which tool fails most? | `GROUP BY tool_name` |
| Comparison | Did a prompt change fix it? | `GROUP BY prompt_version` |

## 9. Check every judge

A count is only as good as its judge. Label 100 to 200 examples per failure
mode, with at least 50 passes and 50 failures, and join the judge's answers to
your labels:

```sql
WITH judged AS (
  SELECT h.label AS human, AI.IF(PROMPT('… {0}', h.reply)) AS judge
  FROM human_labels h)
SELECT AVG(CASE WHEN judge THEN 1.0 ELSE 0.0 END) FILTER (WHERE human)     AS tpr,
       AVG(CASE WHEN judge THEN 0.0 ELSE 1.0 END) FILTER (WHERE NOT human) AS tnr
FROM judged
```

## 10. Known limits in Quail

- `IN (subquery)` fails; write a `LEFT JOIN`. An `IN` list of literals works.
- Quail knows only some DataFusion functions; `md5` fails.
- In the BigQuery dialect, backslashes in string literals are escapes; in a regex, write `[(]`, not `\(`.
- `AI.MAP` takes the instruction as a string literal.
- On `main` at `ceaa3975`, `AI.MAP` over a join fails with "nothing buildable and nothing in flight"; `AI.MAP` over one table works.
- Untested: whether `AI_EXTRACT` parses where `AI.EXTRACT(x, {...})` fails in the BigQuery dialect.
