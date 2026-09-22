"""Build the QUAIL-B avg tokens/sec-by-dataset headline figure.

Tok/sec follows Quail AGENTS.md / quail-b: requested input tokens per
second. One shared numerator per query for Quail, vLLM, and SoL:
    requested_tokens / runtime_s
(SoL uses sol_s). Requested tokens are the full prompt lengths summed,
counting shared prefixes every time whether or not KV was reused.
quail-b exposes this as input_tokens / input_tokens_per_second.
Do not use per-method fresh_tokens.

SoL is drawn as a horizontal line over each dataset group (not a bar),
matching the latency plot style in make_bio4_plots.py. The plot uses two
linear panels: the four shorter-document datasets share the wide panel, while
BIO's long medical reports get a separate scale in the narrow panel. This
keeps the Quail/vLLM bar gaps legible without changing the settled metric.

Pull inputs the same way as make_bio4_plots.py, then:

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
from matplotlib.ticker import FuncFormatter, MaxNLocator

HERE = Path(__file__).resolve().parent
BLUE = "#4C72B0"
ORANGE = "#DD8452"
DARK = "#333333"
LEFT_DATASETS = ("IMDB", "FEV", "LEP", "AGENT")
RIGHT_DATASETS = ("BIO",)


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
        shared = float(
            sol.get("requested_tokens")
            or quail["requested_tokens"]
        )
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

    # BIO-4 only when present in comparison.json (same sf 0.1 suite).
    # Do not side-load a separate bio4-sf01-*.json family here — that mixed
    # a different run into the headline and looked like a scale mismatch.
    bio4_quail_path = workdir / "bio4-sf01-quail.json"
    bio4_vllm_path = workdir / "bio4-sf01-vllm.json"
    bio4_sol_path = workdir / "bio4-sf01-sol.json"
    if False and bio4_quail_path.exists() and bio4_vllm_path.exists() and bio4_sol_path.exists():
        quail_q = _load(bio4_quail_path)["queries"][0]
        vllm_q = _load(bio4_vllm_path)["queries"][0]
        sol_est = _load(bio4_sol_path)["estimate"]
        assert quail_q["id"] == "BIO-4" and vllm_q["id"] == "BIO-4"
        shared = float(quail_q["metrics"]["input_tokens"])
        rows.append(
            {
                "query": "BIO-4",
                "method": "Quail",
                "tok_per_sec": _tok_per_sec(shared, float(quail_q["runtime_s"])),
                "runtime_s": float(quail_q["runtime_s"]),
                "shared_tokens": shared,
            }
        )
        rows.append(
            {
                "query": "BIO-4",
                "method": "vLLM",
                "tok_per_sec": _tok_per_sec(shared, float(vllm_q["runtime_s"])),
                "runtime_s": float(vllm_q["runtime_s"]),
                "shared_tokens": shared,
            }
        )
        rows.append(
            {
                "query": "BIO-4",
                "method": "SoL",
                "tok_per_sec": _tok_per_sec(shared, float(sol_est["sol_s"])),
                "runtime_s": float(sol_est["sol_s"]),
                "shared_tokens": shared,
            }
        )

    return rows


def plot_headline(rows, destination: Path):
    """Draw two linear-scale panels with Quail/vLLM bars and SoL lines."""
    by = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by[_dataset(row["query"])][row["method"]].append(row["tok_per_sec"])

    left_datasets = [name for name in LEFT_DATASETS if name in by]
    right_datasets = [name for name in RIGHT_DATASETS if name in by]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )

    def _fmt(value, _pos):
        if value == 0:
            return "0"
        if value >= 1_000_000:
            return f"{value / 1_000_000:g}M"
        if value >= 1_000:
            return f"{value / 1_000:g}k"
        return f"{value:g}"

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 5.35),
        gridspec_kw={"width_ratios": [4, 1], "wspace": 0.34},
    )
    width = 0.36

    def _nice_ymax(values):
        peak = max(values)
        # Keep SoL off the spine without jumping to a distant empty tick.
        raw = peak * 1.18
        if raw >= 1_000_000:
            step = 5_000_000 if raw >= 10_000_000 else 1_000_000
        elif raw >= 1_000:
            step = 1_000
        else:
            step = 1
        return ((int(raw) + step - 1) // step) * step

    def _draw_panel(axis, datasets, title):
        positions = list(range(len(datasets)))
        panel_values = []
        for index, (method, color) in enumerate(
            (("Quail", BLUE), ("vLLM", ORANGE))
        ):
            values = [statistics.mean(by[name][method]) for name in datasets]
            panel_values.extend(values)
            offset = (index - 0.5) * width
            axis.bar(
                [position + offset for position in positions],
                values,
                width,
                color=color,
                zorder=2,
            )

        for position, name in enumerate(datasets):
            if not by[name]["SoL"]:
                continue
            sol = statistics.mean(by[name]["SoL"])
            panel_values.append(sol)
            # Plain segment only — no center markers.
            axis.plot(
                [position - 0.42, position + 0.42],
                [sol, sol],
                color=DARK,
                linewidth=2.2,
                solid_capstyle="butt",
                marker="",
                zorder=3,
            )

        axis.set_xlim(-0.58, len(datasets) - 0.42)
        axis.set_ylim(0, _nice_ymax(panel_values))
        axis.set_yscale("linear")
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [
                f"{name}\n({len(by[name]['Quail'])} queries)"
                for name in datasets
            ]
        )
        axis.set_ylabel("Average tokens / second")
        axis.set_title(title, fontsize=13, pad=8)
        axis.yaxis.set_major_locator(
            MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10])
        )
        axis.yaxis.set_major_formatter(FuncFormatter(_fmt))
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65, zorder=0)
        axis.set_axisbelow(True)

    _draw_panel(axes[0], left_datasets, "IMDB, FEV, LEP, AGENT")
    _draw_panel(axes[1], right_datasets, "BIO: long medical reports")

    figure.suptitle(
        "QUAIL-B average tokens/sec by dataset\n"
        "Qwen3 4B FP8, one H100, scale factor 0.1",
        fontsize=16,
        y=0.99,
    )
    # One shared legend, parked in the empty upper-left of the wide panel.
    axes[0].legend(
        handles=[
            Patch(facecolor=BLUE, label="Quail"),
            Patch(facecolor=ORANGE, label="vLLM"),
            Line2D([0], [0], color=DARK, linewidth=2.2, marker="", label="SoL estimate"),
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        borderaxespad=0.4,
    )
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.15, top=0.82, wspace=0.34)
    destination = Path(destination)
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(
        destination.with_suffix(".png"), dpi=300, bbox_inches="tight"
    )
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
