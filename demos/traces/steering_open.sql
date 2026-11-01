-- First pass over typed messages: describe each one in a few words,
-- with no categories fixed in advance. Read the answers, then write
-- the AI.CLASSIFY query from what they show.
SELECT e.id,
       e.session_id,
       e.mid_turn,
       AI.MAP(e.context,
              'The USER line at the end is a message the user typed to a coding agent. In at most 15 words, say what the user wants the agent to do differently from what it was doing. If the user is not correcting, redirecting, questioning, or stopping the agent, answer exactly NONE.',
              48) AS steer
FROM exchanges e
