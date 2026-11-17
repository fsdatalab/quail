# Inputs for building a codebook

- steps.json: the steps the open-coding agent read, from AI agents (Claude Opus 4.6, GPT-5.4, Gemini 3.1 Pro) on terminal tasks. Fields: step_id; task (an id); agent; hack_score (the judge's score, 0 to 1); task_description (the full task); reasoning (the agent's reasoning for the step); commands (the shell commands it ran).
- open_codes.json: for each of the $num_groups $groups, the open codes an agent wrote after reading the $group's sampled steps: each code has an intent, a target, the steps that show it, and an optional parent code; each $group also has a "difference" sentence.

$how
