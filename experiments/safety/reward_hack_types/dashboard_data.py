"""Write the dashboard's data files for one codebook (step 7).

    uv run --no-sync python -m experiments.safety.reward_hack_types.dashboard_data \
        WORK CODES --run sae_a

Writes, into WORK/dashboard/:
- coded_steps.json: each flagged step with its intent, target, and their probabilities;
- trace_steps.json: every step (reasoning, commands, output) of each run with a
  flagged step;
- runs.json: each such run's task, agent, task text, and number of steps;
- codebook.json: the run's intents and targets with their definitions and rules.
"""

import argparse
import collections
import json
import os

import pyarrow.parquet as pq

from experiments.safety.reward_hack_types.build_steps import split_step

AGENTS = {
    "claude-opus-4.6": "Claude Opus 4.6",
    "gpt-5.4": "GPT-5.4",
    "gemini-3.1-pro": "Gemini 3.1 Pro",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "work",
        help="folder with steps.json, tw_steps_v1.parquet, and each run's codebook",
    )
    parser.add_argument("codes", help="folder with <run>/codes.parquet")
    parser.add_argument("--run", default="sae_a", help="the codebook to show")
    args = parser.parse_args()
    with open(os.path.join(args.work, "steps.json")) as source:
        steps = json.load(source)
    with open(os.path.join(args.work, args.run, "book", "codebook.json")) as source:
        book = json.load(source)
    with open(os.path.join(args.work, args.run, "meta.json")) as source:
        seen = set(json.load(source)["seen"])
    codes = {
        r["s.id"]: r
        for r in pq.read_table(
            os.path.join(args.codes, args.run, "codes.parquet")
        ).to_pylist()
    }
    coded = []
    for s in steps:
        r = codes[s["step_id"]]
        run_id, index = s["step_id"].split("#")
        coded.append(
            {
                "step_id": s["step_id"],
                "run_id": run_id,
                "step_index": int(index),
                "task": s["task"],
                "agent": AGENTS[s["agent"]],
                "hack_score": s["hack_score"],
                "intent": r["intent"],
                "target": r["target"],
                "intent_probability": round(
                    dict(r["intent_probabilities"])[r["intent"]], 3
                ),
                "target_probability": round(
                    dict(r["target_probabilities"])[r["target"]], 3
                ),
                "read_by_builder": "yes" if s["step_id"] in seen else "no",
            }
        )
    wanted = {c["run_id"] for c in coded}
    table = pq.read_table(
        os.path.join(args.work, "tw_steps_v1.parquet"),
        columns=["id", "run_id", "step_index", "text_alone"],
    )
    trace, runs = [], {}
    for row in table.to_pylist():
        if row["run_id"] not in wanted:
            continue
        parts = split_step(row["text_alone"])
        trace.append(
            {
                "run_id": row["run_id"],
                "step_index": int(row["step_index"]),
                "step_id": row["id"],
                "reasoning": parts["reasoning"],
                "commands": parts["commands"],
                "output": parts["output"],
            }
        )
        _, task, agent, _ = row["run_id"].split("/")
        runs.setdefault(
            row["run_id"],
            {
                "run_id": row["run_id"],
                "task": task,
                "agent": AGENTS[agent],
                "task_description": parts["task_description"],
            },
        )
    trace.sort(key=lambda d: (d["run_id"], d["step_index"]))
    num_steps = collections.Counter(d["run_id"] for d in trace)
    for run_id, run in runs.items():
        run["num_steps"] = num_steps[run_id]
    codebook = [
        {
            "kind": kind,
            "order": i,
            "name": c["name"],
            "definition": c["definition"],
            "include": c.get("include", ""),
            "exclude": c.get("exclude", ""),
            "parent": c.get("parent") or "",
        }
        for kind, key in (("intent", "intents"), ("target", "targets"))
        for i, c in enumerate(book[key])
    ]
    out = os.path.join(args.work, "dashboard")
    os.makedirs(out, exist_ok=True)
    for name, rows in (
        ("coded_steps", coded),
        ("trace_steps", trace),
        ("runs", list(runs.values())),
        ("codebook", codebook),
    ):
        with open(os.path.join(out, f"{name}.json"), "w") as target:
            json.dump(rows, target)
    print(f"{len(coded)} coded steps, {len(trace)} trace steps in {len(runs)} runs")


if __name__ == "__main__":
    main()
