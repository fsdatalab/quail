-- One row per message a person typed in Claude Code, with the context
-- before it. Set the variable `traces` to a glob of session files first:
--   SET VARIABLE traces = '~/.claude/projects/*/*.jsonl';
-- Typed messages are found by structure (origin.kind = 'human'), not by
-- matching text. Every other record is kept as context, cut short.

WITH records AS MATERIALIZED (
    SELECT json ->> 'sessionId' AS session_id,
           json ->> 'timestamp' AS ts,
           json ->> 'uuid' AS uuid,
           json ->> 'type' AS kind,
           json ->> '$.origin.kind' AS origin,
           json -> '$.message.content' AS content,
           json -> '$.attachment' AS attachment,
           json ->> 'isSidechain' AS sidechain
    FROM read_ndjson_objects(getvariable('traces'))
    -- DuckDB 1.5.5 fails when this read has a second condition, so
    -- subagent records are dropped in the next step
    WHERE json ->> 'type' IN ('user', 'assistant', 'attachment')
),

main_records AS (
    SELECT * FROM records WHERE sidechain IS DISTINCT FROM 'true'
),

-- Content is either a string or a list of blocks; give each block a row.
-- Parse the list once and read fields from each small block.
blocks AS (
    SELECT session_id, ts, uuid, kind, origin,
           0 AS block_index, 'text' AS block_type,
           content ->> '$' AS body, NULL AS tool_name
    FROM main_records
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
                FROM main_records
                WHERE kind IN ('user', 'assistant')
                  AND json_type(content) = 'ARRAY'))
    UNION ALL
    -- messages typed while the agent was working
    SELECT session_id, ts, uuid, 'user', attachment ->> '$.origin.kind',
           0, 'queued', attachment ->> 'prompt', NULL
    FROM main_records
    WHERE kind = 'attachment' AND attachment ->> 'type' = 'queued_command'
),

entries AS (
    SELECT session_id, ts, uuid, block_index,
           (kind = 'user' AND origin = 'human'
            AND block_type IN ('text', 'queued')) AS typed,
           block_type = 'queued' AS mid_turn,
           body,
           CASE
               WHEN kind = 'user' AND origin = 'human' AND block_type = 'queued'
                   THEN 'USER (typed while the agent was working): ' || left(body, 4000)
               WHEN kind = 'user' AND origin = 'human'
                   THEN 'USER: ' || left(body, 4000)
               WHEN block_type = 'tool_use'
                   THEN 'AGENT CALLS ' || tool_name || ': ' || left(body, 300)
               WHEN block_type = 'tool_result'
                   THEN 'TOOL RESULT: ' || left(body, 300)
               WHEN kind = 'assistant'
                   THEN 'AGENT: ' || left(body, 1500)
               ELSE 'SYSTEM: ' || left(body, 300)
           END AS line
    FROM blocks
    WHERE block_type NOT IN ('thinking', 'redacted_thinking', 'image')
      AND body IS NOT NULL AND trim(body) <> ''
),

ordered AS MATERIALIZED (
    SELECT *,
           row_number() OVER (PARTITION BY session_id
                              ORDER BY ts, uuid, block_index) AS position,
           string_agg(line, chr(10) || chr(10)) OVER (
               PARTITION BY session_id ORDER BY ts, uuid, block_index
               ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) AS actions
    FROM entries
),

-- the conversation alone: typed messages and the agent's text replies
talk AS (
    SELECT session_id, position,
           string_agg(line, chr(10) || chr(10)) OVER (
               PARTITION BY session_id ORDER BY position
               ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS conversation
    FROM ordered
    WHERE typed OR line LIKE 'AGENT: %'
),

heads AS (
    SELECT session_id, left(arg_min(body, position), 1500) AS head
    FROM ordered
    WHERE typed
    GROUP BY session_id
)

SELECT o.session_id || ':' || o.position AS id,
       o.session_id,
       o.position,
       o.ts AS timestamp,
       o.mid_turn,
       o.body AS message,
       length(o.body) AS message_chars,
       'TASK (first user message of the session):' || chr(10) || h.head
           || chr(10) || chr(10)
           || 'CONVERSATION BEFORE THIS MESSAGE:' || chr(10)
           || coalesce(t.conversation, '') || chr(10) || chr(10)
           || 'LATEST AGENT ACTIONS:' || chr(10)
           || coalesce(o.actions, '') || chr(10) || chr(10)
           || o.line AS context
FROM ordered o
JOIN heads h USING (session_id)
JOIN talk t USING (session_id, position)
WHERE o.typed

