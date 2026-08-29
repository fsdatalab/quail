"""Plot the large join results counted through Acero.

Rebuild from the repository root:

    W=$(mktemp -d)
    modal volume get quail-results \
      benchmarks/quailb/runs/qb_20260827T065151Z_46683ed6/\
20260827T065151Z-quailb-sf0.1-lf1-qwen3-4b-fp8-parallel2.json $W/
    uv run --with matplotlib python \
      reports/make_arrow_result_streaming_plots.py $W
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from plot_colors import BLUE, GREEN


SOURCE = (
    "20260827T065151Z-quailb-sf0.1-lf1-"
    "qwen3-4b-fp8-parallel2.json"
)


def main(workdir: Path) -> None:
    data = json.loads((workdir / SOURCE).read_text())
    by_query = {
        row["query"]: row
        for row in data["passes"]["single"]["queries"]
    }
    names = ["IMDB-9", "BIO-7"]
    rows_millions = [by_query[name]["rows"] / 1_000_000 for name in names]

    plt.style.use(Path(__file__).parent / "quail.mplstyle")
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    bars = ax.barh(
        names,
        rows_millions,
        color=[BLUE, GREEN],
        height=0.55,
    )
    ax.set_xlabel("Final result rows counted by Acero (millions)")
    ax.set_xlim(0, max(rows_millions) * 1.2)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    for bar, value in zip(bars, rows_millions):
        ax.text(
            value + max(rows_millions) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f} million",
            va="center",
        )
    output = Path(__file__).parent / "plots" / "arrow_result_streaming.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: make_arrow_result_streaming_plots.py WORKDIR")
    main(Path(sys.argv[1]))
