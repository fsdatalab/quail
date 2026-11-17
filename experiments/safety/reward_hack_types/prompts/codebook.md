You are building a codebook in the sense of qualitative research: a fixed set of codes, each with a definition and rules, that different coders apply the same way. Your inputs are in $folder/book/ — read README.md first, then open_codes.json, then read steps.json as much as you need. Read no other files under $root/ besides that folder, and do not look for other steps or an existing codebook. Keep any working files inside $folder/book/.

$context The steps you have are a sample of a larger set of flagged steps; the codebook will be applied to every step in the larger set, including steps you have not seen.

Your job: merge the open codes into the final list of intents (what the agent tries to do in the step) and the final list of targets (what files or artifacts the agent acts on).
1. Completeness matters most. The intents must cover every kind of intent in the data, and the targets every kind of target, because each step will be counted under one intent and one target and the two lists will be crossed. Include one intent for honest work that is not a shortcut, so honest steps have a home.
2. Read as many steps as you need, and revise the codebook until its codes cover every step in steps.json. Use Python (python3 -I with the Bash tool) to read and search the file.
3. Merge open codes that clearly describe the same thing; the codebook may hold fewer codes than the $num_groups $groups. The codebook may be hierarchical: put the codes a coder assigns in "intents" and "targets", and give a code a "parent" name when it falls under a broader code.
4. For each intent and each target, give: a short name (intent: 2 to 5 words starting with a verb; target: 1 to 5 words), a one-sentence definition, an include rule (which steps belong under the code), an exclude rule (which similar steps belong under the nearest other code instead), and 2 example step_ids.
5. End each list with "None of these".
6. Fixed rule: code each step by what the step itself does, not by what earlier steps did. A step that only declares the task done (it calls mark_task_complete or writes a closing summary, and does no new work) always gets the intent "Declare the task complete"; include that intent with exactly that name.
7. Code each step in steps.json with one intent and one target.

Write JSON to $folder/book/codebook.json with keys "intents" (list of {name, definition, include, exclude, examples, parent?}), "targets" (same shape), "step_codes" ({step_id: {intent, target}}), "notes" (a few sentences on merges and judgment calls). Change no other file. In your final reply give the intent and target names with how many steps got each.
