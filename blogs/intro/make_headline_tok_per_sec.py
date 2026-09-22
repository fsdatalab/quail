"""Build the QUAIL-B average tokens/second-by-dataset headline figure.

Tokens/second follows Quail AGENTS.md / quail-b: requested input tokens per
second. One shared numerator is used per query for Quail, vLLM, and SoL:

    requested_tokens / runtime_s

SoL uses ``sol_s`` as its runtime. Requested tokens are the full prompt lengths
summed, counting shared prefixes every time whether or not KV was reused.
quail-b exposes this as input_tokens / input_tokens_per_second. Do not use
per-method fresh_tokens.

The figure is one plot with one shared x-axis. IMDB, FEV, LEP, and AGENT use the
left 0--4M scale; BIO uses the explicit right 0--30M scale. SoL is drawn as a
dashed horizontal segment over each dataset group, not as a bar.

Pull inputs the same way as make_bio4_plots.py, then run:

    uv run --with matplotlib python blogs/intro/make_headline_tok_per_sec.py "$W"
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, FixedLocator

HERE = Path(__file__).resolve().parent
BLUE = "#4C72B0"
ORANGE = "#DD8452"
DARK = "#333333"
DATASETS = ("IMDB", "FEV", "LEP", "AGENT", "BIO")
LEFT_DATASETS = DATASETS[:-1]
BIO_DATASET = DATASETS[-1]
LEFT_YLIM = (0, 4_000_000)
RIGHT_YLIM = (0, 30_000_000)


def _load(path: Path):
    return json.loads(Path(path).read_text())


def _dataset(query: str) -> str:
    return query.split("-", 1)[0]


def _tok_per_sec(shared_tokens: float, runtime_s: float) -> float:
    return shared_tokens / runtime_s


def collect_rows(workdir: Path):
    """Return per-query rows for Quail, vLLM, and SoL."""
    comparison = _load(workdir / "comparison.json")
    rows = []

    for query, quail in comparison["rows"]["quail"].items():
        vllm = comparison["rows"]["pipelined_vllm"][query]
        sol = comparison.get("sol", {}).get(query) or {}
        shared = float(sol.get("requested_tokens") or quail["requested_tokens"])
        rows.append(
            {
                "query": query,
                "method": "Quail",
                "tok_per_sec": _tok_per_sec(shared, float(quail["runtime_s"])),
                "runtime_s": float(quail["runtime_s"]),
                "shared_tokens": shared,
            }
        )
        rows.append(
            {
                "query": query,
                "method": "vLLM",
                "tok_per_sec": _tok_per_sec(shared, float(vllm["runtime_s"])),
                "runtime_s": float(vllm["runtime_s"]),
                "shared_tokens": shared,
            }
        )
        sol_s = sol.get("sol_s")
        if sol_s is not None:
            rows.append(
                {
                    "query": query,
                    "method": "SoL",
                    "tok_per_sec": _tok_per_sec(shared, float(sol_s)),
                    "runtime_s": float(sol_s),
                    "shared_tokens": shared,
                }
            )

    return rows


def _tick_label(value, _position):
    """Format axis ticks with M/k suffixes and no scientific notation."""
    if value == 0:
        return "0"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def plot_headline(rows, destination: Path):
    """Draw one shared-x plot with an explicit BIO-only right scale."""
    by = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by[_dataset(row["query"])][row["method"]].append(row["tok_per_sec"])

    datasets = [name for name in DATASETS if name in by]
    if datasets != list(DATASETS):
        raise ValueError(f"Expected datasets {DATASETS}, found {tuple(datasets)}")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.spines.top": False,
            "pdf.fonttype": 42,
        }
    )

    figure, left_axis = plt.subplots(figsize=(11.8, 5.5))
    right_axis = left_axis.twinx()
    positions = list(range(len(DATASETS)))
    width = 0.36

    # Bars for the first four datasets use the left scale.
    for method_index, (method, color) in enumerate(
        (("Quail", BLUE), ("vLLM", ORANGE))
    ):
        offset = (method_index - 0.5) * width
        values = [statistics.mean(by[name][method]) for name in LEFT_DATASETS]
        left_axis.bar(
            [position + offset for position in positions[:4]],
            values,
            width,
            color=color,
            zorder=2,
        )

    # BIO occupies the fifth x position but is measured only by the right axis.
    bio_position = positions[-1]
    for method_index, (method, color) in enumerate(
        (("Quail", BLUE), ("vLLM", ORANGE))
    ):
        offset = (method_index - 0.5) * width
        right_axis.bar(
            bio_position + offset,
            statistics.mean(by[BIO_DATASET][method]),
            width,
            color=color,
            zorder=2,
        )

    # Dashed, marker-free SoL segments, each drawn against its dataset's scale.
    for position, name in zip(positions[:4], LEFT_DATASETS):
        sol = statistics.mean(by[name]["SoL"])
        left_axis.plot(
            [position - 0.42, position + 0.42],
            [sol, sol],
            color=DARK,
            linewidth=2.2,
            linestyle="--",
            dash_capstyle="butt",
            marker="",
            zorder=3,
        )
    bio_sol = statistics.mean(by[BIO_DATASET]["SoL"])
    right_axis.plot(
        [bio_position - 0.42, bio_position + 0.42],
        [bio_sol, bio_sol],
        color=DARK,
        linewidth=2.2,
        linestyle="--",
        dash_capstyle="butt",
        marker="",
        zorder=3,
    )

    left_axis.set_xlim(-0.58, len(DATASETS) - 0.42)
    left_axis.set_ylim(*LEFT_YLIM)
    right_axis.set_ylim(*RIGHT_YLIM)
    left_axis.set_yscale("linear")
    right_axis.set_yscale("linear")

    left_axis.set_xticks(positions)
    left_axis.set_xticklabels(
        [f"{name}\n({len(by[name]['Quail'])} queries)" for name in DATASETS]
    )
    left_axis.set_ylabel("Average tokens / second")
    right_axis.set_ylabel("Average tokens / second")

    left_axis.yaxis.set_major_locator(FixedLocator(range(0, 4_000_001, 1_000_000)))
    right_axis.yaxis.set_major_locator(FixedLocator(range(0, 30_000_001, 5_000_000)))
    left_axis.yaxis.set_major_formatter(FuncFormatter(_tick_label))
    right_axis.yaxis.set_major_formatter(FuncFormatter(_tick_label))

    # A restrained separator reinforces that only BIO uses the explicit right scale.
    left_axis.axvline(3.5, color="#B8B8B8", linewidth=0.9, linestyle=(0, (2, 3)), zorder=1)
    left_axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65, zorder=0)
    left_axis.set_axisbelow(True)
    right_axis.grid(False)
    right_axis.patch.set_visible(False)

    left_axis.legend(
        handles=[
            Patch(facecolor=BLUE, label="Quail"),
            Patch(facecolor=ORANGE, label="vLLM"),
            Line2D(
                [0],
                [0],
                color=DARK,
                linewidth=2.2,
                linestyle="--",
                marker="",
                label="SoL estimate",
            ),
        ],
        frameon=False,
        loc="upper left",
        ncol=1,
        borderaxespad=0.6,
        handlelength=2.5,
    )

    # No title or subtitle: the blog caption supplies the figure context.
    figure.subplots_adjust(left=0.085, right=0.90, bottom=0.18, top=0.96)
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
