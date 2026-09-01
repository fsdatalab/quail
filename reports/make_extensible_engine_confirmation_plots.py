"""Plot the typed engine confirmation runs.

Pull the measured files from the quail-results volume, then pass the work
directory to this script:

    W=<workdir>
    modal volume get quail-results benchmarks/quailb/families/20260831T144252Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/biodex.json $W/biodex.json
    modal volume get quail-results benchmarks/quailb/runs/qb_20260831T062218Z_1192cd76/20260831T062218Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families.json $W/agent.json
    modal volume get quail-results ablations/extensible_engine_confirmation_4b.json $W/confirmation_4b.json
    modal volume get quail-results ablations/extensible_engine_confirmation_4b_agent1.json $W/confirmation_4b_agent1.json
    modal volume get quail-results ablations/extensible_engine_confirmation_4b_output_check.json $W/confirmation_4b_output_check.json
    modal volume get quail-results ablations/extensible_engine_confirmation_4b_timing_check.json $W/confirmation_4b_timing_check.json
    uv run --with matplotlib python reports/make_extensible_engine_confirmation_plots.py $W
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"
plt.style.use(HERE / "quail.mplstyle")
sys.path.insert(0, str(HERE))
from plot_colors import BLUE, DARK, GRAY, ORANGE, TEAL  # noqa: E402


def load(path):
    """Load one JSON file."""
    with path.open() as source:
        return json.load(source)


def query_rows(report):
    """Index confirmation rows by query ID."""
    return {row["query"]: row for row in report["queries"]}


def family_query(report, query_id):
    """Return one query row from a QuailB family run."""
    return next(
        row for row in report["passes"]["single"]["queries"]
        if row["query"] == query_id
    )


def main(workdir):
    """Build the confirmation figure from pulled result files."""
    workdir = Path(workdir)
    main_rows = {
        "BIO-2": family_query(
            load(workdir / "biodex.json")["quail"], "BIO-2"
        ),
        "AGENT-1": family_query(
            load(workdir / "agent.json"), "AGENT-1"
        ),
    }
    primary = query_rows(load(workdir / "confirmation_4b.json"))
    primary.update(query_rows(load(
        workdir / "confirmation_4b_agent1.json"
    )))
    output_check = query_rows(load(
        workdir / "confirmation_4b_output_check.json"
    ))
    timing_check = query_rows(load(
        workdir / "confirmation_4b_timing_check.json"
    ))

    queries = ("BIO-2", "AGENT-1")
    x = np.arange(len(queries))
    width = 0.3
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    main_wall = [main_rows[query]["wall_s"] for query in queries]
    typed_wall = [primary[query]["wall_s"] for query in queries]
    for positions, values, color, label in (
        (x - width / 2, main_wall, GRAY, "main"),
        (x + width / 2, typed_wall, BLUE, "typed engine"),
    ):
        axes[0].bar(positions, values, width, color=color)
        for position, value in zip(positions, values):
            axes[0].text(
                position,
                value + 2,
                f"{label}\n{value:.2f}",
                ha="center",
                va="bottom",
                color=DARK,
            )
    for index, query in enumerate(queries):
        change = 100 * (typed_wall[index] / main_wall[index] - 1)
        axes[0].text(
            index,
            min(main_wall[index], typed_wall[index]) * 0.55,
            f"{change:+.1f}%",
            ha="center",
            color="white",
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(queries)
    axes[0].set_ylabel("query time (seconds)")
    axes[0].set_ylim(0, max(main_wall + typed_wall) * 1.18)

    series = (
        ("primary", primary, BLUE, "o"),
        ("output check", output_check, ORANGE, "x"),
        ("timing check", timing_check, TEAL, "D"),
    )
    offsets = (-0.15, 0, 0.15)
    for (label, rows, color, marker), offset in zip(series, offsets):
        for index, query in enumerate(queries):
            if query not in rows:
                continue
            ratio = rows[query]["wall_s"] / main_rows[query]["wall_s"]
            axes[1].scatter(
                index + offset,
                ratio,
                color=color,
                marker=marker,
                s=55,
            )
            axes[1].text(
                index + offset,
                ratio + 0.09,
                f"{label}\n{ratio:.2f}x",
                ha="center",
                va="bottom",
                color=DARK,
            )
    axes[1].axhline(1, color=GRAY, linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(queries)
    axes[1].set_ylabel("query time divided by main")
    axes[1].set_ylim(0, 3.55)

    fig.tight_layout(w_pad=3)
    OUT.mkdir(exist_ok=True)
    fig.savefig(
        OUT / "extensible_engine_confirmation.png",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: make_extensible_engine_confirmation_plots.py <workdir>"
        )
    main(sys.argv[1])
