You are writing open codes for $groups of steps taken by AI agents on terminal tasks. Open coding, from grounded theory, labels data in your own words before any fixed set of labels exists.

Read the whole file $folder/open/$groups.md, in chunks until you have read all of it. Each step names its task; read the task's full text in $folder/open/tasks.md whenever you need it. Do not open any other file except to write your answer.

$context $how

For each $group, write as many open codes as its steps need; a $group that mixes several kinds of steps gets several codes. Each open code has:
- intent: what the agent tries to do in the step, 2 to 5 words, starting with a verb;
- target: what files or artifacts the agent acts on, 1 to 5 words, a noun phrase;
- steps: the step_ids of the sampled steps that show the code;
- parent (optional): a broader open code that this code falls under, when you group narrow codes.
Every sampled step should appear in at least one open code. Also write, for each $group, "difference": one plain sentence on what separates the $group's steps from the "not in this $group" steps. Describe what steps do, not one striking example, and do not name specific tasks in an intent or target.

Write JSON to $folder/open/open_codes.json in the form {"<$group number>": {"codes": [{"intent": "...", "target": "...", "steps": ["..."], "parent": "..."}], "difference": "..."}, ...} for all $num_groups $groups. Keep any working files inside $folder/open/. Change no other file. In your final reply give the number of open codes per $group.
