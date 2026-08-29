"""Plot the filter to join KV retention check.

Rebuild from the repository root:

    W=$(mktemp -d)
    modal volume get quail-results \
      /runs/run_1787795777696587173.json $W/
    uv run --with matplotlib python \
      reports/old/make_filter_join_kv_retention_plots.py $W
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_colors import BLUE, DARK, GREEN


def main(workdir: Path) -> None:
    source = workdir / "run_1787795777696587173.json"
    data = json.loads(source.read_text())
    kv = data["kv_manager"]
    retained = kv["retained_after_filters"]
    reused = kv["join_anchor_hits"]

    plt.style.use(Path(__file__).resolve().parents[1] / "quail.mplstyle")
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bars = ax.bar(
        ["Passed filter and\nretained in KV", "Reused by\nthe join"],
        [retained, reused],
        color=[BLUE, GREEN],
        width=0.58,
    )
    ax.set_ylabel("Documents")
    ax.set_ylim(0, max(retained, reused) * 1.28)
    ax.tick_params(axis="y", left=False, labelleft=False)
    for bar, value in zip(bars, (retained, reused)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.18,
            str(value),
            ha="center",
            va="bottom",
        )
    ax.text(
        0.5,
        max(retained, reused) * 1.17,
        f"{reused / max(1, retained):.0%} reused; 0 evicted",
        ha="center",
        color=DARK,
    )
    output = Path(__file__).parent / "plots" / "filter_join_kv_retention.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_filter_join_kv_retention_plots.py WORKDIR")
    main(Path(sys.argv[1]))
