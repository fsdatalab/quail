"""Plot the FEV-9 fixed-plan comparison from pulled volume summaries.

Pull the comparison directory, then pass it as the first argument:

    uv run modal volume get quail-results \
      /ablations/fixed-join-plan-20260906T043918Z /tmp
    uv run --with matplotlib python reports/make_fixed_join_plan_plots.py \
      /tmp/fixed-join-plan-20260906T043918Z

The report names the exact comparison directory on the volume.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pyarrow.parquet as pq

from plot_colors import BLUE, GRAY


def compare_answers(root):
    """Compare every saved predicate answer table."""
    names = {path.name for path in (root / "adaptive").glob("*.parquet")}
    assert names == {path.name for path in (root / "fixed").glob("*.parquet")}
    assert len(names) == 7
    for name in sorted(names):
        tables = [pq.read_table(root / label / name) for label in ("adaptive", "fixed")]
        columns = sorted(tables[0].column_names)
        ordered = [table.select(columns).sort_by([(name, "ascending") for name in columns])
                   for table in tables]
        assert ordered[0].equals(ordered[1]), name
        print(f"{name}: {len(ordered[0]):,} identical answers")


def main(workdir):
    """Plot query time and document prefix recomputation."""
    root = Path(workdir)
    records = [json.loads((root / label / "summary.json").read_text())
               for label in ("adaptive", "fixed")]
    compare_answers(root)
    assert records[0]["rows"] == records[1]["rows"]
    for row in records:
        print(f"{row['label']}: {row['query_seconds']:.2f} seconds, "
              f"{row['document_pairs_per_second']:,.2f} document pairs/second, "
              f"${row['usd_per_query']:.5f}/query, {row['rows']:,} output rows")
    for key in ("query_seconds", "fresh_tokens", "regret_tokens"):
        values = [row[key] if key in row else row["report"][key] for row in records]
        delta = values[1] - values[0]
        print(f"{key}: {delta:+,.2f}, {100 * delta / values[0]:+.2f}%")
    plt.style.use(Path(__file__).parent / "quail.mplstyle")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    metrics = [
        ("FEV-9 query time", "seconds", [row["query_seconds"] for row in records]),
        ("FEV-9 document recomputation", "tokens",
         [row["report"]["regret_tokens"] for row in records]),
    ]
    for axis, (title, unit, values) in zip(axes, metrics):
        axis.bar([0, 1], values, color=[GRAY, BLUE], width=0.6)
        axis.set_xticks([0, 1], ["Adaptive plan", "Fixed plan"])
        axis.set_ylabel(unit)
        axis.set_title(title)
        for position, value in enumerate(values):
            label = f"{value:,.2f}" if unit == "seconds" else f"{value:,}"
            axis.annotate(label, (position, value), xytext=(0, 5),
                          textcoords="offset points", ha="center")
        maximum = max(values) or 1
        axis.set_ylim(0, maximum * 1.35)
        delta = values[1] - values[0]
        change = f"{delta:+,.2f} {unit}" if unit == "seconds" else f"{delta:+,} {unit}"
        if values[0]:
            change += f" ({100 * delta / values[0]:+.1f}%)"
        axis.text(0.5, 0.96, change, transform=axis.transAxes,
                  ha="center", va="top")
    figure.subplots_adjust(wspace=0.4, bottom=0.16, top=0.87)
    output = Path(__file__).parent / "plots" / "fixed_join_plan.png"
    figure.savefig(output, dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    main(sys.argv[1])
