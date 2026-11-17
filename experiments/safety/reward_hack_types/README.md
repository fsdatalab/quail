# Types of reward hacks in Terminal Wrench

The code here finds the kinds of reward hacking that agents do when nobody
asked them to hack. It runs the procedure in the doc
[Finding Types of Reward Hacks: End-to-End Procedure](https://claude.ai/artifact/AB6eEFXUH76pAncb6xscCg):
each flagged agent step gets an intent (what the agent tries to do) and a
target (what files or artifacts it acts on).

## Layout

| File | Step | What it does |
| --- | --- | --- |
| `pick_runs.py` | 1 | Lists the 1,503 baseline runs that offered no reference solution. |
| `modal_cells.py` | 2, 3, 6 | Runs the Quail queries on Modal: the judge (`AI.SCORE`), the SAE features (`AI.SAE_FEATURES`), and the coding (two `AI.CLASSIFY` calls). |
| `build_steps.py` | 2 | Keeps the steps with a judge score above the cutoff, with their full text. |
| `cluster.py` | 3 | Clusters the steps three ways (SAE, plain text, random; two runs each) and writes each run's agent inputs and prompts. |
| `prompts/open_coding.md` | 4 | Instructions for the open-coding agent. |
| `prompts/codebook.md` | 5 | Instructions for the codebook agent. |
| `prompts/context.md` | 4, 5 | The context both agents get about the steps. |
| `compare.py` | | Compares the six codebooks without a reference. |
| `dashboard_data.py` | 7 | Writes the dashboard's data files for one codebook. |
| `dashboard_page.html` | 7 | The page of the [dashboard](https://claude.ai/artifact/UHJnzeHNhXsYFY6oBigWax), a claude.ai Dashboard artifact. |

## Running it

Every command uses one working folder, `WORK`.

1. Pick the runs:

       uv run --no-sync python -m experiments.safety.reward_hack_types.pick_runs WORK

2. Score every baseline step, copy the scores and the steps into `WORK`, and keep the flagged steps:

       uv run modal run --detach -m experiments.safety.reward_hack_types.modal_cells --step judge --run <name>
       modal volume get quail-results reward_hack_types/<name>/judge.parquet WORK/
       modal volume get quail-results terminal_wrench/inputs/tw_steps_v1.parquet WORK/
       uv run --no-sync python -m experiments.safety.reward_hack_types.build_steps WORK --cutoff 0.2

3. Read the SAE features, then cluster:

       uv run modal run --detach -m experiments.safety.reward_hack_types.modal_cells --step features --run <name>
       modal volume get quail-results reward_hack_types/<name>/features.parquet WORK/
       uv run --no-project --with scikit-learn==1.5.2 --with scipy --with "numpy<2" \
           --with "pyarrow<17" python -m experiments.safety.reward_hack_types.cluster WORK

4. For each run (`sae_a`, `sae_b`, `text_a`, `text_b`, `random_a`, `random_b`), give a Claude agent
   `WORK/<run>/open/PROMPT.md`, then copy its `open/open_codes.json` into `WORK/<run>/book/`.
5. Give a second Claude agent `WORK/<run>/book/PROMPT.md`. It writes `WORK/<run>/book/codebook.json`.
6. Code all flagged steps with the six codebooks in one Quail session:

       uv run modal run --detach -m experiments.safety.reward_hack_types.modal_cells --step code --run <name> \
           --steps-path WORK/steps.json \
           --codebooks "sae_a=WORK/sae_a/book/codebook.json,sae_b=WORK/sae_b/book/codebook.json,..."

7. Compare the codebooks and build the dashboard's data:

       modal volume get quail-results reward_hack_types/<name> CODES
       uv run --no-sync python -m experiments.safety.reward_hack_types.compare WORK CODES
       uv run --no-sync python -m experiments.safety.reward_hack_types.dashboard_data WORK CODES --run sae_a

## Saved results

The preliminary results come from these files on the `quail-results` volume:

- judge scores: `/sae_probe/20261115-judge-task-reasoning/qwen3.8-27b-fp8-judge.parquet`;
- SAE features: `/sae_probe/20261115-reasoning/qwen3-8b-sae-bf16-features.parquet`;
- codes of the six codebooks: `/hack_codes/codebook-v6-<run>/qwen3.8-27b-fp8-codes.parquet`.

Those runs used earlier copies of `modal_cells.py`, which saved under other folder names.
The runs of one clustering method share their clusters and sampled steps; only the agents differ.
`cluster.py` reproduces the saved SAE and plain-text samples and both agents' prompts exactly. Its random
batches match the saved ones, but its random samples within each batch differ, because the saved run
drew them with an unseeded generator.
