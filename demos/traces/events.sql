-- One row per steering event (a message typed mid-turn, an Esc
-- interrupt, or a rejected tool call), with what the agent was doing
-- before it. Reads the `messages` table from messages.sql.
--
-- `context` is what the model reads:
--   TASK: the session's first typed message, cut to 500 characters
--   RECENT: everything from the agent's 5th-last text message before
--           the event, in order: typed messages, the agent's text
--           messages (each cut to 3000 characters), and each tool call
--           with its input cut to 300 characters. Tool results are left
--           out.
--   EVENT: what the user did
--   NEXT USER MESSAGE: after an interrupt or rejection, the next message
--           the user typed, which often says why

WITH counted AS (
    -- `agent_count`: the agent's text messages up to and including this row
    SELECT *,
           count(*) FILTER (WHERE role = 'agent' AND tool_name IS NULL) OVER (
               PARTITION BY session_id ORDER BY position
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS agent_count
    FROM messages
),

events AS (
    SELECT * FROM counted WHERE event IS NOT NULL
),

starts AS (
    -- each event's window starts at the agent's 5th-last text message
    SELECT e.session_id, e.position,
           coalesce(min(a.position), e.position) AS start
    FROM events e
    LEFT JOIN counted a
        ON a.session_id = e.session_id
       AND a.role = 'agent' AND a.tool_name IS NULL
       AND a.position < e.position
       AND a.agent_count > e.agent_count - 5
    GROUP BY e.session_id, e.position
),

recent AS (
    SELECT s.session_id, s.position,
           string_agg(
               CASE
                   WHEN m.typed THEN 'USER: ' || left(m.text, 2000)
                   WHEN m.tool_name IS NOT NULL
                       THEN '-> ' || m.tool_name || ': '
                            || left(regexp_replace(m.text, '\s+', ' ', 'g'), 300)
                   ELSE 'AGENT: ' || left(m.text, 3000)
               END,
               chr(10) || chr(10) ORDER BY m.position) AS lines
    FROM starts s
    JOIN counted m
        ON m.session_id = s.session_id
       AND m.position >= s.start AND m.position < s.position
       AND (m.typed OR m.role = 'agent')
    GROUP BY s.session_id, s.position
),

next_messages AS (
    SELECT e.session_id, e.position,
           arg_min(m.text, m.position) AS next_text
    FROM events e
    JOIN messages m
        ON m.session_id = e.session_id AND m.typed AND m.position > e.position
    WHERE e.event IN ('interrupt', 'rejection')
    GROUP BY e.session_id, e.position
),

heads AS (
    SELECT session_id, left(arg_min(text, position), 500) AS head
    FROM messages
    WHERE typed
    GROUP BY session_id
)

SELECT e.session_id || ':' || e.position AS id,
       e.session_id,
       e.position,
       e.timestamp,
       e.event,
       CASE WHEN e.event = 'mid_turn' THEN e.text ELSE n.next_text END AS message,
       'TASK: ' || coalesce(h.head, '') || chr(10) || chr(10)
           || 'RECENT:' || chr(10) || coalesce(r.lines, '(nothing yet)')
           || chr(10) || chr(10)
           || CASE e.event
                  WHEN 'mid_turn'
                      THEN 'EVENT: the user typed this while the agent was working: '
                           || left(e.text, 2000)
                  WHEN 'interrupt'
                      THEN 'EVENT: the user pressed Esc and interrupted the agent.'
                  ELSE 'EVENT: the user rejected the last tool call: '
                       || left(e.text, 500)
              END
           || CASE WHEN n.next_text IS NOT NULL
                   THEN chr(10) || chr(10) || 'NEXT USER MESSAGE: '
                        || left(n.next_text, 2000)
                   ELSE '' END AS context
FROM events e
LEFT JOIN heads h USING (session_id)
LEFT JOIN recent r USING (session_id, position)
LEFT JOIN next_messages n USING (session_id, position)
