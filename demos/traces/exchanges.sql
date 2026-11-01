-- One row per typed message, preceded by everything the agent did since
-- the user last wrote. Reads the `messages` table from messages.sql.
--
-- `context` is what the model reads:
--   TASK: the session's first typed message, cut to 500 characters
--   AGENT: the agent's text replies since the user last wrote, whole up
--          to the last 6000 characters
--   TOOLS: each tool the agent called, with its count
--   LAST ACTION: the agent's last tool call or result, cut to 300
--   USER: the typed message, cut to 4000 characters

WITH segmented AS (
    -- `since` is the position of the user's previous typed message
    SELECT *,
           max(CASE WHEN typed THEN position END) OVER (
               PARTITION BY session_id ORDER BY position
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS since
    FROM messages
),

replies AS (
    SELECT session_id, since,
           string_agg(text, chr(10) || chr(10) ORDER BY position)
               FILTER (WHERE role = 'agent' AND tool_name IS NULL) AS agent_text,
           arg_max(CASE WHEN role = 'tool' THEN 'result: ' ELSE tool_name || ': ' END
                       || left(text, 300), position)
               FILTER (WHERE tool_name IS NOT NULL OR role = 'tool') AS last_action
    FROM segmented
    WHERE NOT typed
    GROUP BY session_id, since
),

tools AS (
    SELECT session_id, since,
           string_agg(tool_name || ' x' || calls, ', ' ORDER BY calls DESC,
                      tool_name) AS tool_calls
    FROM (SELECT session_id, since, tool_name, count(*) AS calls
          FROM segmented
          WHERE tool_name IS NOT NULL
          GROUP BY ALL)
    GROUP BY session_id, since
),

heads AS (
    SELECT session_id, left(arg_min(text, position), 500) AS head
    FROM messages
    WHERE typed
    GROUP BY session_id
)

SELECT s.session_id || ':' || s.position AS id,
       s.session_id,
       s.position,
       s.timestamp,
       s.mid_turn,
       s.text AS message,
       length(s.text) AS message_chars,
       length(r.agent_text) AS agent_chars,
       'TASK: ' || h.head || chr(10) || chr(10)
           || 'AGENT (everything since the user last wrote):' || chr(10)
           || CASE
                  WHEN r.agent_text IS NULL THEN '(no text reply)'
                  WHEN length(r.agent_text) > 6000
                      THEN '[...] ' || right(r.agent_text, 6000)
                  ELSE r.agent_text
              END || chr(10) || chr(10)
           || 'TOOLS: ' || coalesce(t.tool_calls, 'none') || chr(10)
           || 'LAST ACTION: ' || coalesce(r.last_action, 'none') || chr(10) || chr(10)
           || CASE WHEN s.mid_turn THEN 'USER (typed while the agent was working): '
                   ELSE 'USER: ' END
           || left(s.text, 4000) AS context
FROM segmented s
JOIN heads h USING (session_id)
LEFT JOIN replies r
    ON r.session_id = s.session_id AND r.since IS NOT DISTINCT FROM s.since
LEFT JOIN tools t
    ON t.session_id = s.session_id AND t.since IS NOT DISTINCT FROM s.since
WHERE s.typed
