"""Plot the kernel-source ablation on the two QUAIL-B queries:
IMDB-7 (unified attention path) and BIO-2 (merge_quant attention
path), our fused kernels against vLLM's.

Pull the measured inputs from the quail-results volume, then pass the
work directory to this script:

    W=<workdir>
    modal volume get quail-results experiments/kernel_source_imdb7.json $W/
    modal volume get quail-results experiments/kernel_source_bio2.json $W/
    modal volume get quail-results experiments/kernel_source_profile.json $W/
    uv run --with matplotlib python reports/make_kernel_source_plots.py $W

Besides the two PNGs it prints the derived per-query metrics table
(query time, throughput, dollars per query) for the report.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "plots"
OUT.mkdir(exist_ok=True)
plt.style.use(HERE / "quail.mplstyle")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from plot_colors import BLUE, GRAY, GREEN, ORANGE  # noqa: E402

from quail.bench.evaluate import H100_USD_PER_HOUR  # noqa: E402

SOURCES = ("quail", "vllm_ops", "vllm_compiled")
# the attention path lives in the plot title, so the bar labels
# carry only the kernel source
SOURCE_LABELS = {
    "quail": "Quail fused kernels",
    "vllm_ops": "vLLM ops, unfused",
    "vllm_compiled": "vLLM compiled set",
}
COLORS = {"quail": GREEN, "vllm_ops": GRAY, "vllm_compiled": BLUE}
FILES = {"IMDB-7": "kernel_source_imdb7.json",
         "BIO-2": "kernel_source_bio2.json"}


def load(workdir, name):
    with open(Path(workdir) / name) as f:
        return json.load(f)


def last(report, source):
    """The last repetition, matching the cell's comparison rows."""
    return report["runs"][source][-1]


def work_items(report):
    """(count, unit): documents for a filter query, evaluated pairs
    for a join query."""
    rows = report["table_rows"]
    if report["query"] == "IMDB-7":
        return rows["reviews"], "documents"
    if report["query"] == "BIO-2":
        return rows["reports"] * rows["terms"], "document pairs"
    raise ValueError(report["query"])


def rate_panel(workdir):
    """Grouped bars: microseconds per fresh token, both queries."""
    reports = {q: load(workdir, f) for q, f in FILES.items()}
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.0))
    fig.set_tight_layout(False)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.86,
                        bottom=0.18, wspace=0.85)
    for ax, (query, report) in zip(axes, reports.items()):
        tokens = last(report, "quail")["fresh_tokens"]
        base = last(report, "quail")["us_per_token"]
        for i, source in enumerate(SOURCES):
            row = last(report, source)
            value = row["us_per_token"]
            ax.barh(i, value, color=COLORS[source], height=0.62)
            note = f"{value:.2f}  ({row['wall_s']:.1f} s)"
            if source != "quail":
                pct = 100 * (value - base) / base
                note = f"{value:.2f}  (+{pct:.0f}%, {row['wall_s']:.1f} s)"
            ax.annotate(note, (value, i), xytext=(4, 0),
                        textcoords="offset points", va="center",
                        fontsize=8.2)
        ax.set_yticks(range(len(SOURCES)))
        ax.set_yticklabels([SOURCE_LABELS[s] for s in SOURCES],
                           fontsize=8.8)
        ax.invert_yaxis()
        ax.set_xlabel("microseconds per fresh token")
        ax.set_title(
            f"{query} — {report['attention_path']} attention path "
            f"({tokens / 1e6:.2f}M fresh tokens)",
            fontsize=9.8, fontweight="normal")
        ax.set_xlim(0, None)
        ax.margins(x=0.33)
        ax.xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(integer=True))
    fig.savefig(OUT / "kernel_source_rates.png", dpi=300)
    plt.close(fig)


def profile_panel(workdir):
    """Stacked per-category GPU time per token on IMDB-7."""
    try:
        prof = load(workdir, "kernel_source_profile.json")
    except FileNotFoundError:
        return
    path = prof["attention_path"]
    order = ["gemm", "attention", "quail_fused", "inductor",
             "vllm_elementwise", "quant", "merge", "copies", "other"]
    # in-bar labels name the WORK; the row label and the legend name
    # who provides the kernels. The three middle buckets are the same
    # work (norms, rotary, silu) from three providers.
    work = {"gemm": "matmuls", "attention": "attention",
            "quail_fused": "norms, rope, silu",
            "inductor": "norms, rope, silu",
            "vllm_elementwise": "norms, rope, silu",
            "quant": "group quant", "merge": "merge",
            "copies": "copies", "other": "other"}
    legend = {"gemm": "matmuls (DeepGEMM)",
              "attention": "attention (FA3)",
              "quail_fused": "our fused Triton kernels",
              "inductor": "Inductor-generated kernels",
              "vllm_elementwise": "vLLM ops",
              "quant": "group quant", "merge": "merge",
              "copies": "copies", "other": "other"}
    palette = {"gemm": GRAY, "attention": ORANGE,
               "quail_fused": GREEN, "inductor": "#7FA6D9",
               "vllm_elementwise": "#9A9A9A", "quant": "#C9A26B",
               "merge": "#B085C9", "copies": "#D6D6D6",
               "other": "#EDEDED"}
    fig, ax = plt.subplots(figsize=(7.8, 3.0))
    sources = [s for s in SOURCES if s in prof["sources"]]
    seen = []
    for i, source in enumerate(sources):
        cats = prof["sources"][source]["category_us_per_token"]
        left = 0.0
        for cat in order:
            value = cats.get(cat, 0.0)
            if not value:
                continue
            ax.barh(i, value, left=left, color=palette[cat],
                    height=0.6,
                    label=(legend[cat] if cat not in seen else None))
            if cat not in seen:
                seen.append(cat)
            # a two-line label fits only in wide segments; the
            # legend identifies the narrow ones
            if value >= 1.0:
                ax.annotate(f"{work[cat]}\n{value:.2f}",
                            (left + value / 2, i), ha="center",
                            va="center", fontsize=7.4)
            left += value
        ax.annotate(f"  {left:.2f} total", (left, i), va="center",
                    fontsize=8.5)
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels([SOURCE_LABELS[s] for s in sources],
                       fontsize=8.8)
    ax.invert_yaxis()
    ax.set_xlabel("microseconds per fresh token")
    ax.set_title(
        f"{prof['query']} — GPU kernel time by category "
        f"({path} attention path, profiled run)",
        fontsize=9.8, fontweight="normal")
    ax.margins(x=0.12)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.55),
              ncol=len(seen), fontsize=7.4, frameon=False,
              handlelength=1.0, columnspacing=1.0)
    fig.savefig(OUT / "kernel_source_profile.png", dpi=300)
    plt.close(fig)


def metrics_table(workdir):
    """Query time, throughput, and dollars per query, derived from
    the stored walls and table sizes (one H100)."""
    print(f"\nH100 price: ${H100_USD_PER_HOUR}/hour, 1 GPU\n")
    for query, name in FILES.items():
        report = load(workdir, name)
        items, unit = work_items(report)
        path = report["attention_path"]
        print(f"{query} ({path} path, {items} {unit}):")
        for source in SOURCES:
            row = last(report, source)
            wall = row["wall_s"]
            rate = items / wall
            usd = wall / 3600 * H100_USD_PER_HOUR
            print(f"  {source:14s} {wall:8.2f} s   "
                  f"{rate:10.1f} {unit}/s   ${usd:.4f}/query")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_kernel_source_plots.py <workdir>")
    workdir = sys.argv[1]
    rate_panel(workdir)
    profile_panel(workdir)
    metrics_table(workdir)
    print(f"\nwrote {OUT / 'kernel_source_rates.png'}")
    print(f"wrote {OUT / 'kernel_source_profile.png'}")


if __name__ == "__main__":
    main()
