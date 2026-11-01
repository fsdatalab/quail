-- Normalize Claude Code session logs into one row per message.
-- Set the variable `traces` to a glob of session files first:
--   SET VARIABLE traces = '~/.claude/projects/*/*.jsonl';
-- Run it once into a DuckDB file; later tables read `messages` from it.
-- Typed messages are found by structure (origin.kind = 'human'), not by
-- matching text. Needs DuckDB 2.0 or newer.
--
-- Columns: session_id, position, timestamp, role ('user', 'agent',
-- 'tool', or 'system'), typed, mid_turn, tool_name, text (whole).

CREATE OR REPLACE TABLE messages AS
WITH records AS MATERIALIZED (
    SELECT json ->> 'sessionId' AS session_id,
           json ->> 'timestamp' AS ts,
           json ->> 'uuid' AS uuid,
           json ->> 'type' AS kind,
           json ->> '$.origin.kind' AS origin,
           json -> '$.message.content' AS content,
           json -> '$.attachment' AS attachment
    FROM read_ndjson_objects(getvariable('traces'))
    WHERE json ->> 'type' IN ('user', 'assistant', 'attachment')
      AND coalesce(json ->> 'isSidechain', 'false') = 'false'
),

-- Content is either a string or a list of blocks; give each block a row.
-- Parse the list once and read fields from each small block.
blocks AS (
    SELECT session_id, ts, uuid, kind, origin,
           0 AS block_index, 'text' AS block_type,
           content ->> '$' AS body, NULL AS tool_name
    FROM records
    WHERE kind IN ('user', 'assistant') AND json_type(content) = 'VARCHAR'
    UNION ALL
    SELECT session_id, ts, uuid, kind, origin, block_index,
           block ->> 'type',
           coalesce(block ->> 'text', block ->> 'content', block ->> 'input'),
           block ->> 'name'
    FROM (SELECT session_id, ts, uuid, kind, origin, block_index,
                 block_text::JSON AS block
          FROM (SELECT session_id, ts, uuid, kind, origin,
                       unnest(content ->> '$[*]') AS block_text,
                       generate_subscripts(content ->> '$[*]', 1) AS block_index
                FROM records
                WHERE kind IN ('user', 'assistant')
                  AND json_type(content) = 'ARRAY'))
    UNION ALL
    -- messages typed while the agent was working
    SELECT session_id, ts, uuid, 'user', attachment ->> '$.origin.kind',
           0, 'queued', attachment ->> 'prompt', NULL
    FROM records
    WHERE kind = 'attachment' AND attachment ->> 'type' = 'queued_command'
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
       tool_name,
       body AS text
FROM blocks
WHERE block_type NOT IN ('thinking', 'redacted_thinking', 'image')
  AND body IS NOT NULL AND trim(body) <> '';
