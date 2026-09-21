"""Build the current aggregate latency plot and BIO-4 result image.

Pull the saved results before running the script:

    W=/tmp/quail-blog-bio4; mkdir -p "$W"
    uv run modal volume get quail-results \
      reports/quailb-raw-2026-09-19/comparison.json "$W/comparison.json"
    R=benchmarks/quailb/family-runs/20260920T062701Z-bio4-4b
    uv run modal volume get quail-results \
      "$R/quail/biodex/run.json" "$W/bio4-sf01-quail.json"
    uv run modal volume get quail-results \
      "$R/pipelined_vllm/biodex/run.json" "$W/bio4-sf01-vllm.json"
    uv run modal volume get quail-results \
      sol/2026-09-20-bio4-qwen3-4b-sf0.1.json \
      "$W/bio4-sf01-sol.json"
    R=benchmarks/quailb/family-runs/20260920T064415Z-bio4-4b-sf1.0
    uv run modal volume get quail-results \
      "$R/quail/biodex/run.json" "$W/quail.json"
    uv run modal volume get quail-results \
      "$R/pipelined_vllm/biodex/run.json" "$W/vllm.json"
    uv run modal volume get quail-results \
      sol/2026-09-20-bio4-qwen3-4b-sf1.0.json "$W/sol.json"
    uv run --with matplotlib python blogs/intro/make_bio4_plots.py "$W"
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
BLUE = "#4C72B0"
ORANGE = "#DD8452"
DARK = "#333333"
GRAY = "#98A2B3"

BASE_QUERY_ORDER = (
    [f"IMDB-{number}" for number in range(1, 11)]
    + [f"BIO-{number}" for number in range(1, 4)]
    + [f"FEV-{number}" for number in range(1, 11)]
    + [f"LEP-{number}" for number in range(1, 6)]
    + ["AGENT-1", "AGENT-2"]
)
QUERY_ORDER = BASE_QUERY_ORDER[:13] + ["BIO-4"] + BASE_QUERY_ORDER[13:]
SOURCE_QUERY_IDS = {
    query: "LEP-7" if query == "LEP-5" else query
    for query in BASE_QUERY_ORDER
}


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
    comparison = _load(workdir / "comparison.json")
    bio4 = {
        "quail": _bio4_row(_load(workdir / "bio4-sf01-quail.json")),
        "pipelined_vllm": _bio4_row(
            _load(workdir / "bio4-sf01-vllm.json")
        ),
    }
    bio4_sol = _load(workdir / "bio4-sf01-sol.json")
    assert bio4_sol["query"] == "BIO-4"
    assert bio4_sol["scale_factor"] == 0.1

    rows = {"quail": {}, "pipelined_vllm": {}}
    sol = {}
    for query in QUERY_ORDER:
        if query == "BIO-4":
            for method in rows:
                rows[method][query] = bio4[method]["runtime_s"]
            sol[query] = bio4_sol["estimate"]["sol_s"]
            continue
        source = SOURCE_QUERY_IDS[query]
        for method in rows:
            saved = comparison["rows"][method].get(source)
            rows[method][query] = None if saved is None else saved["runtime_s"]
        sol[query] = comparison["sol"][source]["sol_s"]

    figure, axis = plt.subplots(figsize=(18, 6.5))
    positions = list(range(len(QUERY_ORDER)))
    width = 0.34
    methods = (
        ("quail", "Quail", BLUE, -width / 2),
        ("pipelined_vllm", "vLLM baseline", ORANGE, width / 2),
    )
    positive = [
        value
        for method in rows.values()
        for value in method.values()
        if value is not None and value > 0
    ]
    missing_height = min(positive) / 1.5
    for method, _, color, offset in methods:
        for position, query in zip(positions, QUERY_ORDER):
            value = rows[method][query]
            x = position + offset
            if value is None:
                axis.scatter(
                    [x],
                    [missing_height],
                    marker="x",
                    s=42,
                    color=color,
                    linewidth=2,
                    zorder=4,
                )
            else:
                axis.bar(x, value, width, color=color, zorder=2)
    for position, query in zip(positions, QUERY_ORDER):
        axis.hlines(
            sol[query],
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
            Patch(facecolor=ORANGE, label="vLLM baseline"),
            Line2D([0], [0], color=DARK, linewidth=1.6, label="SoL estimate"),
            Line2D(
                [0],
                [0],
                color=GRAY,
                marker="x",
                linestyle="none",
                label="Not measured",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=4,
        frameon=False,
    )
    figure.subplots_adjust(bottom=0.30, left=0.08, right=0.99, top=0.90)
    destination = HERE / "figures" / "quailb_latency"
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

    columns = ("Quail", "vLLM baseline", "SoL estimate")
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
        f"Quail is {speedup:.2f} times faster than the vLLM baseline.",
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
    plot_bio4_results(workdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    main(parser.parse_args().workdir)
