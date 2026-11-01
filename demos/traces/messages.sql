-- Normalize Claude Code session logs into one row per message.
-- Set the variable `traces` to a glob of session files first:
--   SET VARIABLE traces = '~/.claude/projects/*/*.jsonl';
-- Run it once into a DuckDB file; later tables read `messages` from it.
-- Typed messages are found by structure (origin.kind = 'human'), not by
-- matching text. Needs DuckDB 2.0 or newer.
--
-- Columns: session_id, position, timestamp, role ('user', 'agent',
-- 'tool', or 'system'), typed, mid_turn, event, tool_name, text (whole).
--
-- `event` marks where the user steered the agent, from the records
-- Claude Code writes:
--   'mid_turn'   a message typed while the agent was working
--                (a queued_command attachment from a person)
--   'interrupt'  Esc: a user record with interruptedMessageId, or the
--                exact text Claude Code writes for it
--   'rejection'  a refused tool call: a tool result on a record with
--                toolDenialKind, or the exact text Claude Code writes

CREATE OR REPLACE TABLE messages AS
WITH records AS MATERIALIZED (
    SELECT json ->> 'sessionId' AS session_id,
           json ->> 'timestamp' AS ts,
           json ->> 'uuid' AS uuid,
           json ->> 'type' AS kind,
           json ->> '$.origin.kind' AS origin,
           json -> '$.message.content' AS content,
           json -> '$.attachment' AS attachment,
           json ->> 'interruptedMessageId' AS interrupted_id,
           json ->> 'toolDenialKind' AS denial_kind
    FROM read_ndjson_objects(getvariable('traces'))
    WHERE json ->> 'type' IN ('user', 'assistant', 'attachment')
      AND coalesce(json ->> 'isSidechain', 'false') = 'false'
),

-- Content is either a string or a list of blocks; give each block a row.
-- Parse the list once and read fields from each small block.
blocks AS (
    SELECT session_id, ts, uuid, kind, origin, interrupted_id, denial_kind,
           0 AS block_index, 'text' AS block_type,
           content ->> '$' AS body, NULL AS tool_name
    FROM records
    WHERE kind IN ('user', 'assistant') AND json_type(content) = 'VARCHAR'
    UNION ALL
    SELECT session_id, ts, uuid, kind, origin, interrupted_id, denial_kind,
           block_index, block ->> 'type',
           coalesce(block ->> 'text', block ->> 'content', block ->> 'input'),
           block ->> 'name'
    FROM (SELECT session_id, ts, uuid, kind, origin, interrupted_id, denial_kind,
                 block_index, block_text::JSON AS block
          FROM (SELECT session_id, ts, uuid, kind, origin, interrupted_id,
                       denial_kind, unnest(content ->> '$[*]') AS block_text,
                       generate_subscripts(content ->> '$[*]', 1) AS block_index
                FROM records
                WHERE kind IN ('user', 'assistant')
                  AND json_type(content) = 'ARRAY'))
    UNION ALL
    -- messages typed while the agent was working; a prompt with an
    -- image is a JSON list of blocks, so keep its text blocks only
    SELECT session_id, ts, uuid, 'user', attachment ->> '$.origin.kind',
           NULL, NULL, 0, 'queued',
           CASE WHEN json_valid(prompt) AND json_type(prompt::JSON) = 'ARRAY'
                THEN array_to_string(list_filter(
                         json_extract_string(prompt, '$[*].text'),
                         lambda x: x IS NOT NULL), chr(10))
                ELSE prompt
           END,
           NULL
    FROM (SELECT *, attachment ->> 'prompt' AS prompt FROM records)
    WHERE kind = 'attachment' AND attachment ->> 'type' = 'queued_command'
    -- one message can be written as two identical attachments
    QUALIFY row_number() OVER (PARTITION BY session_id, ts, prompt
                               ORDER BY uuid) = 1
)

SELECT session_id,
       row_number() OVER (PARTITION BY session_id
                          ORDER BY ts, uuid, block_index) AS position,
       ts AS timestamp,
       CASE
           WHEN kind = 'user' AND origin = 'human' THEN 'user'
           WHEN block_type = 'tool_result' THEN 'tool'
           WHEN kind = 'assistant' THEN 'agent'
           ELSE 'system'
       END AS role,
       coalesce(kind = 'user' AND origin = 'human'
                AND block_type IN ('text', 'queued'), false) AS typed,
       block_type = 'queued' AS mid_turn,
       CASE
           WHEN block_type = 'queued' AND origin = 'human' THEN 'mid_turn'
           WHEN kind = 'user' AND block_type = 'tool_result'
                AND (denial_kind IS NOT NULL
                     OR starts_with(body, 'The user doesn''t want to proceed with this tool use')
                     OR starts_with(body, 'The user doesn''t want to take this action right now'))
               THEN 'rejection'
           WHEN kind = 'user'
                AND (interrupted_id IS NOT NULL
                     OR body IN ('[Request interrupted by user]',
                                 '[Request interrupted by user for tool use]'))
               THEN 'interrupt'
       END AS event,
       tool_name,
       body AS text
FROM blocks
WHERE block_type NOT IN ('thinking', 'redacted_thinking', 'image')
  AND body IS NOT NULL AND trim(body) <> '';
