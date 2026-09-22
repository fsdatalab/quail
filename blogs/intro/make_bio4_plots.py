"""Build the current latency, filter submission, and BIO-4 figures.

Pull the saved results before running the script:

    W=/tmp
    R=benchmarks/quailb/20260922T190951Z-efa30103
    uv run modal volume get quail-results \
      "$R/quail/run.json" "$W/quailb-gigatoken-quail.json"
    uv run modal volume get quail-results \
      "$R/stock_vllm/run.json" "$W/quailb-gigatoken-stock.json"
    uv run modal volume get quail-results \
      "$R/pipelined_vllm/run.json" "$W/quailb-gigatoken-pipelined.json"
    uv run modal volume get quail-results \
      reports/quailb-raw-2026-09-19/comparison.json \
      "$W/quailb-gigatoken-sol-base.json"
    uv run modal volume get quail-results \
      sol/2026-09-20-bio4-qwen3-4b-sf0.1.json \
      "$W/quailb-gigatoken-sol-bio4.json"
    G=/ground_truth/quailb/schema_v1/collections
    uv run modal volume get quail-results \
      "$G/gt_be81cb241d74555dc2da79b5b0662554/manifest.json" \
      "$W/quailb-sol-collection.json"
    uv run modal volume get quail-results \
      "$G/gt_cd3ebdb784f64b9e028e50ea73cdedd0/manifest.json" \
      "$W/quailb-run-collection.json"
    R=benchmarks/quailb/family-runs/20260920T064415Z-bio4-4b-sf1.0
    uv run modal volume get quail-results \
      "$R/quail/biodex/run.json" "$W/quail.json"
    uv run modal volume get quail-results \
      "$R/pipelined_vllm/biodex/run.json" "$W/vllm.json"
    uv run modal volume get quail-results \
      sol/2026-09-20-bio4-qwen3-4b-sf1.0.json "$W/sol.json"
    BENCH=git+https://github.com/fsdatalab/quail-bench.git
    REV=35d026dc2f5b5c1e787268173e81e512b749081a
    uv run --with matplotlib --with "quail-b@$BENCH@$REV" \
      python blogs/intro/make_bio4_plots.py "$W"
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from quailb_results import PIPELINE_QUERIES, QUERY_ORDER, load_results

HERE = Path(__file__).resolve().parent
BLUE = "#4C72B0"
ORANGE = "#DD8452"
TEAL = "#55A868"
DARK = "#333333"


def _load(path):
    """Load one JSON file."""
    return json.loads(Path(path).read_text())


def _bio4_row(suite):
    """Return the single completed BIO-4 query row."""
    assert suite["status"] == "complete"
    assert suite["scale_factor"] in (0.1, 1.0)
    assert suite["metadata"]["model"] == "qwen3-4b-fp8"
    assert suite["gpu_count"] == 1
    assert len(suite["queries"]) == 1
    row = suite["queries"][0]
    assert row["id"] == "BIO-4"
    assert row["status"] == "complete"
    return row


def _configure_matplotlib():
    """Set the figure style used by the blog."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_aggregate(workdir):
    """Plot current scale factor 0.1 latency measurements."""
    rows, sol = load_results(workdir)

    figure, axis = plt.subplots(figsize=(18, 6.5))
    positions = list(range(len(QUERY_ORDER)))
    width = 0.34
    methods = (
        ("quail", "Quail", BLUE, -width / 2),
        ("stock_vllm", "Stock vLLM", ORANGE, width / 2),
    )
    for method, _, color, offset in methods:
        for position, query in zip(positions, QUERY_ORDER):
            value = rows[method][query]["runtime_s"]
            x = position + offset
            axis.bar(x, value, width, color=color, zorder=2)
    for position, query in zip(positions, QUERY_ORDER):
        axis.hlines(
            sol[query]["runtime_s"],
            position - 0.42,
            position + 0.42,
            color=DARK,
            linewidth=1.6,
            zorder=3,
        )

    axis.set_yscale("log")
    axis.set_ylabel("seconds (log scale)")
    axis.set_xticks(positions, QUERY_ORDER, rotation=55, ha="right")
    axis.set_xlim(-0.7, len(QUERY_ORDER) - 0.3)
    axis.set_title(
        "QUAIL-B query latency at scale factor 0.1, Qwen3 4B FP8, one H100"
    )
    axis.grid(False)
    axis.legend(
        handles=[
            Patch(facecolor=BLUE, label="Quail"),
            Patch(facecolor=ORANGE, label="Stock vLLM"),
            Line2D([0], [0], color=DARK, linewidth=1.6, label="SoL estimate"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        frameon=False,
    )
    figure.subplots_adjust(bottom=0.30, left=0.08, right=0.99, top=0.90)
    destination = HERE / "figures" / "quailb_latency"
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_filter_submission(workdir):
    """Compare vLLM filter submission strategies where they differ."""
    rows, sol = load_results(workdir)
    figure, axis = plt.subplots(figsize=(12.5, 5.8))
    positions = list(range(len(PIPELINE_QUERIES)))
    width = 0.34
    methods = (
        ("stock_vllm", "Stock vLLM", ORANGE, -width / 2),
        ("pipelined_vllm", "Pipelined vLLM", TEAL, width / 2),
    )
    for method, _, color, offset in methods:
        values = [rows[method][query]["runtime_s"] for query in PIPELINE_QUERIES]
        axis.bar(
            [position + offset for position in positions],
            values,
            width,
            color=color,
            zorder=2,
        )
    for position, query in zip(positions, PIPELINE_QUERIES):
        axis.hlines(
            sol[query]["runtime_s"],
            position - 0.42,
            position + 0.42,
            color=DARK,
            linewidth=1.6,
            zorder=3,
        )
        stock = rows["stock_vllm"][query]["runtime_s"]
        pipelined = rows["pipelined_vllm"][query]["runtime_s"]
        axis.annotate(
            f"{stock / pipelined:.2f}×",
            (position, max(stock, pipelined)),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axis.set_yscale("log")
    axis.set_ylabel("seconds (log scale)")
    axis.set_xticks(positions, PIPELINE_QUERIES)
    axis.set_xlim(-0.65, len(PIPELINE_QUERIES) - 0.35)
    axis.set_title(
        "vLLM filter submission on multi-filter chains\n"
        "Qwen3 4B FP8, one H100, scale factor 0.1"
    )
    axis.legend(
        handles=[
            Patch(facecolor=ORANGE, label="Stock vLLM"),
            Patch(facecolor=TEAL, label="Pipelined vLLM"),
            Line2D([0], [0], color=DARK, linewidth=1.6, label="SoL estimate"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        frameon=False,
    )
    figure.text(
        0.08,
        0.015,
        "Labels are stock time divided by pipelined time. "
        "Only queries with multiple filters on one document stream are shown.",
        fontsize=10,
    )
    figure.subplots_adjust(bottom=0.27, left=0.09, right=0.98, top=0.82)
    destination = HERE / "figures" / "quailb_filter_submission"
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _duration(seconds):
    """Format seconds as minutes or hours for the result image."""
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} hours"
    return f"{seconds / 60:.2f} minutes"


def _millions(value):
    """Format a token count in millions."""
    return f"{value / 1_000_000:.1f} million"


def plot_bio4_results(workdir):
    """Create a compact image of the full scale BIO-4 results."""
    quail = _bio4_row(_load(workdir / "quail.json"))
    vllm = _bio4_row(_load(workdir / "vllm.json"))
    sol_source = _load(workdir / "sol.json")
    assert sol_source["query"] == "BIO-4"
    assert sol_source["scale_factor"] == 1.0
    sol = sol_source["estimate"]

    columns = ("Quail", "Pipelined vLLM", "SoL estimate")
    values = (
        (
            _duration(quail["runtime_s"]),
            _duration(vllm["runtime_s"]),
            _duration(sol["sol_s"]),
        ),
        (
            f"${quail['metrics']['cost_usd']:.2f}",
            f"${vllm['metrics']['cost_usd']:.2f}",
            f"${sol['usd_per_query']:.2f}",
        ),
        (
            _millions(quail["metrics"]["fresh_tokens"]),
            _millions(vllm["metrics"]["fresh_tokens"]),
            _millions(sol["tokens"]),
        ),
        (
            _millions(quail["metrics"]["regret_tokens"]),
            _millions(vllm["metrics"]["regret_tokens"]),
            "0 (assumed)",
        ),
    )
    rows = (
        "Query time",
        "GPU cost per query",
        "Fresh input tokens",
        "Recomputed KV tokens",
    )

    figure, axis = plt.subplots(figsize=(12, 4.8))
    axis.axis("off")
    axis.set_title(
        "BIO-4 results at scale factor 1.0",
        loc="left",
        pad=24,
        fontsize=21,
        weight="bold",
    )
    axis.text(
        0,
        1.01,
        "Qwen3 4B FP8, one H100, model startup excluded",
        transform=axis.transAxes,
        color="#475467",
        fontsize=12,
    )
    table = axis.table(
        cellText=values,
        rowLabels=rows,
        colLabels=columns,
        cellLoc="center",
        rowLoc="left",
        bbox=[0, 0.19, 1, 0.70],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_text_props(weight="bold", color="#101828")
            cell.set_facecolor("#F2F4F7")
        elif column == -1:
            cell.set_text_props(weight="bold", color="#344054")
            cell.set_facecolor("#FFFFFF")
        else:
            cell.set_facecolor("#FFFFFF")
    speedup = vllm["runtime_s"] / quail["runtime_s"]
    axis.text(
        0,
        0.06,
        f"Quail is {speedup:.2f} times faster than pipelined vLLM.",
        transform=axis.transAxes,
        fontsize=14,
        weight="bold",
        color="#1D4ED8",
    )
    destination = HERE / "figures" / "bio4_results"
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main(workdir):
    """Build all plots from saved benchmark results."""
    _configure_matplotlib()
    plot_aggregate(workdir)
    plot_filter_submission(workdir)
    plot_bio4_results(workdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    main(parser.parse_args().workdir)
