-- One row per typed message, preceded by everything the agent did since
-- the user last wrote. Reads the `messages` table from messages.sql.
--
-- `context` is what the model reads:
--   TASK: the session's first typed message, cut to 500 characters
--   AGENT: the agent's turn since the user last wrote, in order: its text
--          replies whole, and each tool call as one line with the first
--          argument cut to 100 characters. Tool results are left out.
--          The turn keeps its last 6000 characters.
--   USER: the typed message, cut to 4000 characters

WITH segmented AS (
    -- `since` is the position of the user's previous typed message
    SELECT *,
           max(CASE WHEN typed THEN position END) OVER (
               PARTITION BY session_id ORDER BY position
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS since
    FROM messages
),

turns AS (
    SELECT session_id, since,
           string_agg(
               CASE WHEN tool_name IS NULL THEN text
                    ELSE '-> ' || tool_name || ': '
                         || left(regexp_replace(coalesce(
                                list_first(json_extract_string(text, '$.*')),
                                text), '\s+', ' ', 'g'), 100)
               END,
               chr(10) ORDER BY position) AS turn
    FROM segmented
    WHERE role = 'agent'
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
       length(t.turn) AS turn_chars,
       'TASK: ' || h.head || chr(10) || chr(10)
           || 'AGENT (everything since the user last wrote):' || chr(10)
           || CASE
                  WHEN t.turn IS NULL THEN '(nothing yet)'
                  WHEN length(t.turn) > 6000 THEN '[...] ' || right(t.turn, 6000)
                  ELSE t.turn
              END || chr(10) || chr(10)
           || CASE WHEN s.mid_turn THEN 'USER (typed while the agent was working): '
                   ELSE 'USER: ' END
           || left(s.text, 4000) AS context
FROM segmented s
JOIN heads h USING (session_id)
LEFT JOIN turns t
    ON t.session_id = s.session_id AND t.since IS NOT DISTINCT FROM s.since
WHERE s.typed
