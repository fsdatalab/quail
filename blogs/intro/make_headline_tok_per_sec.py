"""Build the QUAIL-B headline throughput figure as a percent of SoL.

Tokens/second follows Quail AGENTS.md / quail-b: requested input tokens per
second. One shared numerator is used per query for Quail, vLLM, and SoL:

    requested_tokens / runtime_s

SoL uses ``sol_s`` as its runtime. Requested tokens are the full prompt lengths
summed, counting shared prefixes every time whether or not KV was reused.
quail-b exposes this as input_tokens / input_tokens_per_second. Do not use
per-method fresh_tokens.

Each bar is that dataset's mean tokens/second divided by its mean SoL
tokens/second. The y-axis runs from 0% to 100%, so the peak is the SoL
estimate for every dataset. SoL is a dashed line at 100%, not a bar.

The 2026-09-19 comparison marks BIO-1 and BIO-3 missing and has no BIO-4
row. When the remake files below are in the workdir, this script fills
BIO-1, BIO-3, and BIO-4 from that remake. BIO-2 stays on the comparison
file. BIO-1 and BIO-3 keep the requested-token total already stored on
their SoL rows. BIO-4 uses the Quail run's input-token total, which is
the shared numerator in the blog table.

The figure averages the 31 queries in the QUAIL-B README. The saved
comparison still has the deleted LePaRD queries. Original LEP-5, LEP-6,
and LEP-8 are skipped. Current LEP-5 is read from the row saved as LEP-7.

Pull the inputs, then run the script:

    W=/tmp/quail-blog-headline; mkdir -p "$W"
    uv run modal volume get quail-results \
      reports/quailb-raw-2026-09-19/comparison.json "$W/comparison.json"
    uv run modal volume get quail-results \
      benchmarks/quailb/20260921T190132Z-a2059688/quail/biodex/run.json \
      "$W/bio-remake-quail.json"
    uv run modal volume get quail-results \
      benchmarks/quailb/20260921T190132Z-a2059688/pipelined_vllm/biodex/run.json \
      "$W/bio-remake-vllm.json"
    uv run modal volume get quail-results \
      sol/2026-09-20-bio4-qwen3-4b-sf0.1.json \
      "$W/bio4-sf01-sol.json"
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
from matplotlib.ticker import FixedLocator, FuncFormatter

HERE = Path(__file__).resolve().parent
BLUE = "#4C72B0"
ORANGE = "#DD8452"
DARK = "#333333"
DATASETS = ("BIO", "IMDB", "FEV", "LEP", "AGENT")
# QUAIL-B README: 10 IMDB, 4 BioDEX, 10 FEVER, 5 LePaRD, 2 SWE-Next.
QUERY_COUNTS = {"IMDB": 10, "FEV": 10, "LEP": 5, "AGENT": 2, "BIO": 4}
# The saved comparison still uses ids from before the LePaRD deletion.
# Current LEP-5 was saved as LEP-7. Original LEP-5, LEP-6, and LEP-8
# have empty reference outputs and are not in the benchmark.
EXCLUDED_SAVED_QUERIES = frozenset({"LEP-5", "LEP-6", "LEP-8"})
CURRENT_QUERY_ID = {"LEP-7": "LEP-5"}
BIO_REMAKE_QUERIES = ("BIO-1", "BIO-3", "BIO-4")
Y_MAX = 100


def _load(path: Path):
    return json.loads(Path(path).read_text())


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


def _overlay_bio_remake(comparison: dict, workdir: Path) -> None:
    """Fill BIO-1, BIO-3, and BIO-4 from the 2026-09-21 remake, when present."""
    quail_path = workdir / "bio-remake-quail.json"
    vllm_path = workdir / "bio-remake-vllm.json"
    sol_path = workdir / "bio4-sf01-sol.json"
    if not (quail_path.exists() and vllm_path.exists() and sol_path.exists()):
        return

    quail_by_id = {row["id"]: row for row in _load(quail_path)["queries"]}
    vllm_by_id = {row["id"]: row for row in _load(vllm_path)["queries"]}
    bio4_sol_s = float(_load(sol_path)["estimate"]["sol_s"])
    sol_rows = comparison.setdefault("sol", {})

    for query in BIO_REMAKE_QUERIES:
        quail = quail_by_id[query]
        vllm = vllm_by_id[query]
        if query == "BIO-4":
            shared = float(quail["metrics"]["input_tokens"])
            sol = {"requested_tokens": shared, "sol_s": bio4_sol_s}
        else:
            sol = sol_rows[query]
            shared = float(sol["requested_tokens"])
        comparison["rows"]["quail"][query] = {
            "runtime_s": float(quail["runtime_s"]),
            "requested_tokens": shared,
        }
        comparison["rows"]["pipelined_vllm"][query] = {
            "runtime_s": float(vllm["runtime_s"]),
            "requested_tokens": shared,
        }
        sol_rows[query] = sol


def collect_rows(workdir: Path):
    """Return per-query rows for Quail, vLLM, and SoL."""
    comparison = _load(workdir / "comparison.json")
    _overlay_bio_remake(comparison, workdir)
    rows = []

    for saved_id, quail in comparison["rows"]["quail"].items():
        if saved_id in EXCLUDED_SAVED_QUERIES:
            continue
        query = CURRENT_QUERY_ID.get(saved_id, saved_id)
        vllm = comparison["rows"]["pipelined_vllm"][saved_id]
        sol = comparison.get("sol", {}).get(saved_id) or {}
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
        for method in ("Quail", "vLLM", "SoL"):
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
    """Draw Quail and vLLM as a percent of each dataset's SoL estimate."""
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
        (("Quail", BLUE), ("vLLM", ORANGE))
    ):
        offset = (method_index - 0.5) * (width + pair_gap)
        values = [
            percent_of_sol(by[name][method], by[name]["SoL"]) for name in DATASETS
        ]
        containers = axis.bar(
            [position + offset for position in positions],
            values,
            width,
            color=color,
            zorder=2,
        )
        axis.bar_label(
            containers,
            fmt="%.1f%%",
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
    axis.set_ylim(0, Y_MAX)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [f"{name}\n({len(by[name]['Quail'])} queries)" for name in DATASETS]
    )
    axis.set_ylabel("Percent of SoL estimate")
    axis.yaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
    axis.yaxis.set_major_formatter(FuncFormatter(_percent_tick))

    axis.legend(
        handles=[
            Patch(facecolor=BLUE, label="Quail"),
            Patch(facecolor=ORANGE, label="vLLM"),
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
        loc="upper left",
        borderaxespad=0.6,
        handlelength=2.5,
    )

    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.82)
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
