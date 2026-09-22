"""Build the current QUAIL-B throughput figure as a percent of SoL.

Tokens/second is each method's requested input tokens divided by its query
runtime. Requested input tokens count every complete evaluated prompt,
including positions read from KV. SoL runtime uses exact reference survivors.
The saved BIO-4 SoL result does not include its full requested-token count,
so BIO-4 uses Quail's measured requested-token count as the SoL numerator.

Each bar is that dataset's mean tokens/second divided by its mean SoL
tokens/second. The y-axis is a log scale and runs up to 100%, so the peak
is the SoL estimate for every dataset. SoL is a dashed line at 100%, not
a bar.

Pull the inputs, then run the script:

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
    BENCH=git+https://github.com/fsdatalab/quail-bench.git
    REV=35d026dc2f5b5c1e787268173e81e512b749081a
    uv run --with matplotlib --with "quail-b@$BENCH@$REV" \
      python blogs/intro/make_headline_tok_per_sec.py "$W"
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from quailb_results import QUERY_ORDER, load_results

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

HERE = Path(__file__).resolve().parent
BLUE = "#4C72B0"
ORANGE = "#DD8452"
DARK = "#333333"
DATASETS = ("BIO", "IMDB", "FEV", "LEP", "AGENT")
# QUAIL-B README: 10 IMDB, 4 BioDEX, 10 FEVER, 5 LePaRD, 2 SWE-Next.
QUERY_COUNTS = {"IMDB": 10, "FEV": 10, "LEP": 5, "AGENT": 2, "BIO": 4}
Y_MIN = 1
Y_MAX = 100


def _dataset(query: str) -> str:
    return query.split("-", 1)[0]


def _tok_per_sec(shared_tokens: float, runtime_s: float) -> float:
    return shared_tokens / runtime_s


def percent_of_sol(method_rates: list[float], sol_rates: list[float]) -> float:
    """Return mean tokens/second as a percent of mean SoL tokens/second.

    Args:
        method_rates: Per-query tokens/second for one method.
        sol_rates: Per-query SoL tokens/second for the same dataset.

    Returns:
        100 times the mean method rate divided by the mean SoL rate.

    Raises:
        ValueError: SoL tokens/second is not positive.
    """
    sol_mean = statistics.mean(sol_rates)
    if sol_mean <= 0:
        raise ValueError("SoL tokens/second must be positive")
    return 100.0 * statistics.mean(method_rates) / sol_mean


def collect_rows(workdir: Path):
    """Return current per-query rows for Quail, stock vLLM, and SoL."""
    measured, sol = load_results(workdir)
    rows = []
    for query in QUERY_ORDER:
        quail = measured["quail"][query]
        stock = measured["stock_vllm"][query]
        estimate = sol[query]
        rows.append(
            {
                "query": query,
                "method": "Quail",
                "tok_per_sec": quail["input_tokens_per_second"],
                "runtime_s": quail["runtime_s"],
            }
        )
        rows.append(
            {
                "query": query,
                "method": "Stock vLLM",
                "tok_per_sec": stock["input_tokens_per_second"],
                "runtime_s": stock["runtime_s"],
            }
        )
        rows.append(
            {
                "query": query,
                "method": "SoL",
                "tok_per_sec": _tok_per_sec(
                    estimate["input_tokens"], estimate["runtime_s"]
                ),
                "runtime_s": estimate["runtime_s"],
            }
        )

    return rows


def _rates_by_dataset(rows):
    by = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by[_dataset(row["query"])][row["method"]].append(row["tok_per_sec"])
    return by


def _require_datasets(by) -> None:
    datasets = [name for name in DATASETS if name in by]
    if datasets != list(DATASETS):
        raise ValueError(f"Expected datasets {DATASETS}, found {tuple(datasets)}")
    for name, count in QUERY_COUNTS.items():
        for method in ("Quail", "Stock vLLM", "SoL"):
            found = len(by[name][method])
            if found != count:
                raise ValueError(
                    f"{name} {method} has {found} queries, expected {count}"
                )


def _percent_tick(value, _position):
    """Format y ticks as percents of SoL."""
    return f"{value:.0f}%"


def _tokens_per_second_label(value: float) -> str:
    """Format a tokens/second rate on two lines."""
    if value >= 1_000_000:
        scaled = value / 1_000_000
        decimals = 1 if scaled >= 10 else 2
        number = f"{scaled:.{decimals}f}M"
    elif value >= 1_000:
        number = f"{value / 1_000:.0f}k"
    else:
        number = f"{value:.0f}"
    return f"{number}\ntok/sec"


def plot_headline(rows, destination: Path):
    """Draw Quail and stock vLLM relative to each dataset's SoL estimate."""
    by = _rates_by_dataset(rows)
    _require_datasets(by)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )

    figure, axis = plt.subplots(figsize=(11.8, 5.5))
    figure.suptitle(
        "QUAIL-B average tokens/sec relative to SoL\n"
        "Qwen3 4B FP8, one H100, scale factor 0.1",
        fontsize=16,
        y=0.98,
    )
    positions = list(range(len(DATASETS)))
    width = 0.32
    pair_gap = 0.08

    for method_index, (method, color) in enumerate(
        (("Quail", BLUE), ("Stock vLLM", ORANGE))
    ):
        offset = (method_index - 0.5) * (width + pair_gap)
        values = [
            percent_of_sol(by[name][method], by[name]["SoL"]) for name in DATASETS
        ]
        containers = axis.bar(
            [position + offset for position in positions],
            [value - Y_MIN for value in values],
            width,
            bottom=Y_MIN,
            color=color,
            zorder=2,
        )
        axis.bar_label(
            containers,
            labels=[f"{value:.1f}%" for value in values],
            padding=2,
            fontsize=11,
            color=DARK,
        )
        if method != "Quail":
            continue
        for position, percent, name in zip(positions, values, DATASETS):
            rate = statistics.mean(by[name][method])
            axis.annotate(
                _tokens_per_second_label(rate),
                xy=(position + offset, percent),
                xytext=(0, 16),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color=BLUE,
                fontstyle="italic",
                linespacing=0.95,
                zorder=4,
                annotation_clip=False,
            )

    axis.axhline(
        Y_MAX,
        color=DARK,
        linewidth=1.6,
        linestyle="--",
        dash_capstyle="butt",
        zorder=3,
    )

    axis.set_xlim(-0.58, len(DATASETS) - 0.42)
    axis.set_yscale("log")
    axis.set_ylim(Y_MIN, Y_MAX)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [f"{name}\n({len(by[name]['Quail'])} queries)" for name in DATASETS]
    )
    axis.set_ylabel("Percent of SoL estimate (log scale)")
    axis.yaxis.set_major_locator(FixedLocator([1, 10, 100]))
    axis.yaxis.set_minor_locator(NullLocator())
    axis.yaxis.set_major_formatter(FuncFormatter(_percent_tick))

    axis.legend(
        handles=[
            Patch(facecolor=BLUE, label="Quail"),
            Patch(facecolor=ORANGE, label="Stock vLLM"),
            Line2D(
                [0],
                [0],
                color=DARK,
                linewidth=1.6,
                linestyle="--",
                marker="",
                label="SoL estimate",
            ),
        ],
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=3,
        borderaxespad=0.2,
        handlelength=2.2,
    )

    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.78)
    destination = Path(destination)
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "figures" / "quailb_tok_per_sec",
    )
    args = parser.parse_args()
    rows = collect_rows(args.workdir)
    plot_headline(rows, args.out)
    print(f"wrote {args.out.with_suffix('.png')} and .pdf ({len(rows)} rows)")


if __name__ == "__main__":
    main()
