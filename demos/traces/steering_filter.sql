-- Keep the typed messages that steer the agent, and copy out what in the
-- agent's turn the user reacted to. Both functions only prefill, so this
-- costs less than an AI.MAP over every row. Read `message` and `evidence`
-- for the rows returned, then write the AI.CLASSIFY query.
SELECT e.id,
       e.session_id,
       e.mid_turn,
       e.message,
       AI.EXTRACT(e.context,
                  {'trigger': 'Which words in the AGENT section is the user reacting to?'})
           AS evidence
FROM exchanges e
WHERE AI.IF(PROMPT('Does the USER line at the end correct, redirect, question, or stop the agent?\n\n{0}',
                   e.context),
            {'selectivity': 0.3})
