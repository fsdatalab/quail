"""Plot the FEV-9 shared KV retention comparison from pulled volume summaries.

Pull the comparison directory, then pass it as the first argument:

    uv run modal volume get quail-results \
      /ablations/shared-kv-retention-20260906T054932Z /tmp
    uv run --with matplotlib python reports/make_shared_kv_retention_plots.py \
      /tmp/shared-kv-retention-20260906T054932Z

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
    names = {path.name for path in (root / "first_anchor").glob("*.parquet")}
    assert names == {path.name for path in (root / "shared").glob("*.parquet")}
    assert len(names) == 7
    for name in sorted(names):
        tables = [pq.read_table(root / label / name) for label in ("first_anchor", "shared")]
        columns = sorted(tables[0].column_names)
        ordered = [table.select(columns).sort_by([(name, "ascending") for name in columns])
                   for table in tables]
        assert ordered[0].equals(ordered[1]), name
        print(f"{name}: {len(ordered[0]):,} identical answers")


def main(workdir):
    """Plot time, total work, prefix recomputation, and answer agreement."""
    root = Path(workdir)
    records = [json.loads((root / label / "summary.json").read_text())
               for label in ("first_anchor", "shared")]
    accuracy = json.loads((root / "accuracy.json").read_text())["configurations"]
    output_accuracy = accuracy["shared"]["output_accuracy"]
    precision = 100 * output_accuracy["matching_rows"] / output_accuracy["predicted_rows"]
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
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    metrics = [
        ("FEV-9 query time", "seconds", [row["query_seconds"] for row in records]),
        ("FEV-9 document recomputation", "tokens",
         [row["report"]["regret_tokens"] for row in records]),
        ("FEV-9 total fresh tokens", "tokens",
         [row["report"]["fresh_tokens"] for row in records]),
        ("FEV-9 answer agreement with Qwen3 32B", "percent",
         [100 * accuracy[row["label"]]["answer_accuracy"]["accuracy"] for row in records]),
    ]
    for axis, (title, unit, values) in zip(axes.flat, metrics):
        axis.bar([0, 1], values, color=[GRAY, BLUE], width=0.6)
        axis.set_xticks([0, 1], ["First anchor only", "Shared retention"])
        axis.set_ylabel(unit)
        axis.set_title(title)
        for position, value in enumerate(values):
            label = f"{value:,.2f}" if unit != "tokens" else f"{value:,}"
            axis.annotate(label, (position, value), xytext=(0, 5),
                          textcoords="offset points", ha="center")
        maximum = max(values) or 1
        axis.set_ylim(0, maximum * 1.35)
        delta = values[1] - values[0]
        change = f"{delta:+,.2f} {unit}" if unit != "tokens" else f"{delta:+,} {unit}"
        if unit == "percent":
            change = f"Identical answers; final output precision {precision:.2g}%"
            axis.set_ylim(0, 100)
        elif values[0]:
            change += f" ({100 * delta / values[0]:+.1f}%)"
        axis.text(0.5, 0.96, change, transform=axis.transAxes,
                  ha="center", va="top")
    figure.subplots_adjust(wspace=0.35, hspace=0.45, bottom=0.07, top=0.94)
    output = Path(__file__).parent / "plots" / "shared_kv_retention.png"
    figure.savefig(output, dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    main(sys.argv[1])
