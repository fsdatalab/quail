"""Collect the flagged steps (step 2) with their full task, reasoning, and commands.

    uv run --no-sync python -m experiments.safety.reward_hack_types.build_steps WORK \
        --cutoff 0.2

Reads, from WORK:
- dedicated_runs.json, from pick_runs;
- judge.parquet, from `modal_cells --step judge`;
- tw_steps_v1.parquet, a copy of /results/terminal_wrench/inputs/tw_steps_v1.parquet.

Writes WORK/steps.json: one record per step with a judge score above the
cutoff, from a dedicated run.
"""

import argparse
import json
import os

import pyarrow.parquet as pq


def split_step(text: str) -> dict:
    """Split a step's text into its task, reasoning, commands, and output."""
    task, _, step = text.partition("[STEP]")
    reasoning = (
        step.split("[AGENT]", 1)[-1].split("[COMMANDS]", 1)[0].split("[OUTPUT]", 1)[0]
    )
    commands = (
        step.split("[COMMANDS]", 1)[1].split("[OUTPUT]", 1)[0]
        if "[COMMANDS]" in step
        else ""
    )
    output = step.split("[OUTPUT]", 1)[1] if "[OUTPUT]" in step else ""
    return {
        "task_description": task.replace("[TASK]", "", 1).strip(),
        "reasoning": reasoning.strip(),
        "commands": commands.strip(),
        "output": output.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", help="folder for inputs and outputs")
    parser.add_argument(
        "--cutoff", type=float, default=0.2, help="keep judge scores above this"
    )
    args = parser.parse_args()
    with open(os.path.join(args.work, "dedicated_runs.json")) as source:
        dedicated = set(json.load(source))
    score = {
        r["s.id"]: r["hack_score"]
        for r in pq.read_table(os.path.join(args.work, "judge.parquet")).to_pylist()
    }
    keep = {
        i: s
        for i, s in score.items()
        if s > args.cutoff and i.split("#")[0] in dedicated
    }
    table = pq.read_table(
        os.path.join(args.work, "tw_steps_v1.parquet"), columns=["id", "text_alone"]
    )
    steps = []
    for row in table.to_pylist():
        if row["id"] not in keep:
            continue
        parts = split_step(row["text_alone"])
        _, task, agent, _ = row["id"].split("#")[0].split("/")
        steps.append(
            {
                "step_id": row["id"],
                "task": task,
                "agent": agent,
                "hack_score": round(keep[row["id"]], 3),
                "task_description": parts["task_description"],
                "reasoning": parts["reasoning"],
                "commands": parts["commands"],
            }
        )
    with open(os.path.join(args.work, "steps.json"), "w") as out:
        json.dump(steps, out, ensure_ascii=False)
    print(f"{len(steps)} flagged steps from {len({s['task'] for s in steps})} tasks")


if __name__ == "__main__":
    main()
