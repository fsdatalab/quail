"""Plot the kernel-source ablation: our fused kernels against
vLLM's, on the filter query and the join query.

Pull the measured inputs from the quail-results volume, then pass the
work directory to this script:

    W=<workdir>
    modal volume get quail-results ablations/kernel_source_filter.json $W/
    modal volume get quail-results ablations/kernel_source_join.json $W/
    modal volume get quail-results ablations/kernel_source_profile.json $W/
    uv run --with matplotlib python reports/make_kernel_source_plots.py $W
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"
OUT.mkdir(exist_ok=True)
plt.style.use(HERE / "quail.mplstyle")
sys.path.insert(0, str(HERE))
from plot_colors import BLUE, GRAY, GREEN, ORANGE  # noqa: E402

SOURCES = ("quail", "vllm_ops", "vllm_compiled")
LABELS = {
    "quail": "Quail fused kernels",
    "vllm_ops": "vLLM ops, unfused",
    "vllm_compiled": "vLLM compiled-graph set",
}
COLORS = {"quail": GREEN, "vllm_ops": GRAY, "vllm_compiled": BLUE}


def load(workdir, name):
    with open(Path(workdir) / name) as f:
        return json.load(f)


def last_us(report, source):
    """The last repetition, matching the cell's comparison rows (the
    container's very first measured run carries first-run effects)."""
    return report["runs"][source][-1]["us_per_token"]


def rate_panel(workdir):
    """Grouped bars: microseconds per fresh token, both queries."""
    filt = load(workdir, "kernel_source_filter.json")
    join = load(workdir, "kernel_source_join.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=False)
    for ax, report, title in (
            (axes[0], filt,
             f"filter query ({report_tokens(filt)} fresh tokens)"),
            (axes[1], join,
             f"join query ({report_tokens(join)} fresh tokens)")):
        base = last_us(report, "quail")
        for i, source in enumerate(SOURCES):
            value = last_us(report, source)
            ax.barh(i, value, color=COLORS[source], height=0.62)
            note = f"{value:.2f}"
            if source != "quail":
                pct = 100 * (value - base) / base
                note += f"  (+{pct:.0f}%)" if pct >= 0 \
                    else f"  ({pct:.0f}%)"
            ax.annotate(note, (value, i), xytext=(4, 0),
                        textcoords="offset points", va="center",
                        fontsize=8.5)
        ax.set_yticks(range(len(SOURCES)))
        ax.set_yticklabels([LABELS[s] for s in SOURCES], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("wall microseconds per fresh token")
        ax.set_title(title, fontsize=10, fontweight="normal")
        ax.set_xlim(0, None)
        ax.margins(x=0.22)
    fig.savefig(OUT / "kernel_source_rates.png", dpi=300)
    plt.close(fig)


def report_tokens(report):
    tokens = report["runs"]["quail"][-1]["fresh_tokens"]
    return f"{tokens / 1e6:.2f}M" if tokens >= 1e6 \
        else f"{tokens / 1e3:.0f}k"


def profile_panel(workdir):
    """Stacked per-category GPU time per token on the filter query."""
    try:
        prof = load(workdir, "kernel_source_profile.json")
    except FileNotFoundError:
        return
    order = ["gemm", "attention", "quail_fused", "vllm_fused",
             "inductor", "vllm_elementwise", "quant", "merge",
             "copies", "other"]
    palette = {"gemm": GRAY, "attention": ORANGE,
               "quail_fused": GREEN, "vllm_fused": BLUE,
               "inductor": "#7FA6D9", "vllm_elementwise": "#9A9A9A",
               "quant": "#C9A26B", "merge": "#B085C9",
               "copies": "#D6D6D6", "other": "#EDEDED"}
    names = {"gemm": "matmuls", "attention": "attention",
             "quail_fused": "our fused", "vllm_fused": "vLLM fused",
             "inductor": "Inductor", "vllm_elementwise": "vLLM ops",
             "quant": "group quant", "merge": "merge",
             "copies": "copies", "other": "other"}
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
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
                    label=(names[cat] if cat not in seen else None))
            if cat not in seen:
                seen.append(cat)
            # a two-line label fits only in wide segments; the
            # legend identifies the narrow ones
            if value >= 1.0:
                ax.annotate(f"{names[cat]}\n{value:.2f}",
                            (left + value / 2, i), ha="center",
                            va="center", fontsize=7.4)
            left += value
        ax.annotate(f"  {left:.2f} total", (left, i), va="center",
                    fontsize=8.5)
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels([LABELS[s] for s in sources], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("GPU kernel microseconds per fresh token "
                  "(filter query, profiled run)")
    ax.margins(x=0.12)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.52),
              ncol=len(seen), fontsize=7.4, frameon=False,
              handlelength=1.0, columnspacing=1.0)
    fig.savefig(OUT / "kernel_source_profile.png", dpi=300)
    plt.close(fig)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_kernel_source_plots.py <workdir>")
    workdir = sys.argv[1]
    rate_panel(workdir)
    profile_panel(workdir)
    print(f"wrote {OUT / 'kernel_source_rates.png'}")
    print(f"wrote {OUT / 'kernel_source_profile.png'}")


if __name__ == "__main__":
    main()
