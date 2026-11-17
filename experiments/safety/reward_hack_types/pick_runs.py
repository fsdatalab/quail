"""List the Terminal Wrench baseline runs that offered no reference solution.

The source dataset files 849 March runs (run_name `batch_<date>`) under
the baseline split together with 1,503 dedicated baseline runs. Most
March runs end their first message with a "## Reference Solution"
section that offers `/solution/`, so only the dedicated runs count as
runs whose agents were not offered help.

    uv run --no-sync python -m experiments.safety.reward_hack_types.pick_runs WORK

Writes WORK/dedicated_runs.json: the sorted run ids, such as
`baseline/1104/claude-opus-4.6/1`.
"""

import argparse
import glob
import json
import os

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

INDEX = "few-sh/terminal-wrench-trajectories"
INDEX_REVISION = "622449769ec6ddf33f770794489b8cd21712a0ad"


def run_id(trajectory_path: str) -> str:
    """Return the run id of a trajectory path.

    Args:
        trajectory_path: `tasks/<task>/<model>/baseline_trajectories/<trial>/...`.

    Returns:
        `baseline/<task>/<model>/<trial>`.
    """
    parts = trajectory_path.split("/")
    return f"baseline/{parts[1]}/{parts[2]}/{parts[4]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", help="folder for inputs and outputs")
    work = parser.parse_args().work
    root = snapshot_download(
        INDEX,
        repo_type="dataset",
        revision=INDEX_REVISION,
        allow_patterns=["data/baseline/*.parquet"],
    )
    path = glob.glob(f"{root}/data/baseline/*.parquet")[0]
    rows = pq.read_table(path, columns=["trajectory_path", "run_name"]).to_pylist()
    dedicated = sorted(
        {
            run_id(r["trajectory_path"])
            for r in rows
            if not r["run_name"].startswith("batch_")
        }
    )
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, "dedicated_runs.json"), "w") as out:
        json.dump(dedicated, out)
    print(f"{len(dedicated)} dedicated runs of {len(rows)} baseline runs")


if __name__ == "__main__":
    main()
