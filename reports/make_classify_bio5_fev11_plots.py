r"""Plot BIO-5 and FEV-11 on Qwen3 4B before and after the trie_decode changes.

One page, four panels: latency, input tokens per second, KV regret,
and label agreement, for Quail before the changes, Quail after them,
and stock vLLM, with the SoL estimate as a line; a second, wide page
holds the first three panels side by side. Pull the files, then
run from the repository root:

    W=/tmp/classify-bio5-fev11; mkdir -p "$W"
    B=benchmarks/quailb
    uv run modal volume get quail-results \
      "$B/20261001T010217Z-1e7f7ba8/quail/run.json" "$W/quail-before.json"
    uv run modal volume get quail-results \
      "$B/20261001T010217Z-1e7f7ba8/stock_vllm/run.json" "$W/stock_vllm.json"
    uv run modal volume get quail-results \
      "$B/20261002T005838Z-45fc1a7e/quail/run.json" "$W/quail-now-bio5.json"
    uv run modal volume get quail-results \
      "$B/20261002T020016Z-ae533a0e/quail/run.json" "$W/quail-now-fev11.json"
    uv run modal volume get quail-results \
      sol/2026-10-01-classify-sf0.5.json "$W/sol.json"
    uv run --with matplotlib python reports/make_classify_bio5_fev11_plots.py "$W"
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from plot_colors import BLUE, DARK, GRAY, ORANGE

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots" / "classify_bio5_fev11_qwen3_4b.pdf"
# the slide version: latency, throughput, and KV regret side by side
SLIDE = HERE / "plots" / "classify_bio5_fev11_qwen3_4b_slide.pdf"
QUERIES = ["BIO-5", "FEV-11"]
# (key, legend label, color, files holding its query records)
METHODS = [
    ("before", "Quail before", GRAY, ["quail-before.json"]),
    ("now", "Quail now", BLUE, ["quail-now-bio5.json", "quail-now-fev11.json"]),
    ("stock_vllm", "Stock vLLM (operator-at-a-time)", ORANGE,
     ["stock_vllm.json"]),
]
# group centers, spaced so each SoL label fits beside its group
POSITIONS = [0.0, 1.6]
plt.rcParams["pdf.fonttype"] = 42


def metrics(record):
    """Return the four plotted metrics of one run.json query record."""
    m = record["metrics"]
    label = m["accuracy"]["label_accuracy"]
    return {
        "seconds": m["runtime_s"],
        "input_tokens": m["input_tokens"],
        "tokens_per_second": m["input_tokens"] / m["runtime_s"],
        "regret_tokens": m["regret_tokens"],
        "regret_approximate": m["regret_approximate"],
        "agreement": 100 * label["correct"] / label["evaluated"],
        "evaluated": label["evaluated"],
        "rows": m["accuracy"]["input_document_rows"],
    }


def load(workdir):
    """Read every method's BIO-5 and FEV-11 records and the SoL seconds."""
    rows = {}
    for key, _, _, files in METHODS:
        rows[key] = {}
        for name in files:
            for record in json.loads((workdir / name).read_text())["queries"]:
                if record["id"] in QUERIES and record["status"] == "complete":
                    rows[key][record["id"]] = metrics(record)
    sol = json.loads((workdir / "sol.json").read_text())["qwen"]
    rows["sol"] = {q: {"seconds": sol[q]["sol_s"],
                       "tokens_per_second":
                           rows["now"][q]["input_tokens"] / sol[q]["sol_s"]}
                   for q in QUERIES}
    return rows


def label(value, metric, compact=False):
    """Format a bar's value label; compact drops units and rounds to thousands."""
    if metric == "agreement":
        return f"{value:.1f}%"
    if metric == "seconds":
        return f"{value:.2f}" if compact else f"{value:.2f} s"
    if metric == "tokens_per_second" or compact:
        return f"{value / 1e3:.0f}k" if value >= 1e4 else f"{value / 1e3:.1f}k"
    return f"{value:,.0f}"


def panel(axis, rows, metric, title, unit, log, sol=False, compact=False):
    """Draw one metric as grouped bars per query, with an optional SoL line."""
    width = 0.8 / len(METHODS)
    values = [rows[key][q][metric] for key, *_ in METHODS for q in QUERIES]
    if sol:
        values += [rows["sol"][q][metric] for q in QUERIES]
    if log:
        axis.set_yscale("log")
        axis.set_ylim(min(values) / 2.5, max(values) * 6)
        axis.set_ylabel(f"{unit} (log scale)")
    else:
        axis.set_ylim(0, 115)
        axis.set_ylabel(unit)
    for index, (key, _, color, _) in enumerate(METHODS):
        for position, query in zip(POSITIONS, QUERIES):
            x = position - 0.4 + (index + 0.5) * width
            record = rows[key][query]
            value = record[metric]
            text = label(value, metric, compact)
            # note an engine that left at least 1% of rows unlabeled
            if (metric == "agreement"
                    and record["evaluated"] < 0.99 * record["rows"]):
                text += f"\nof {record['evaluated']:,} labeled"
            axis.bar(x, value, width=width * 0.9, color=color)
            axis.annotate(text, (x, value), xytext=(0, 3),
                          textcoords="offset points", ha="center",
                          va="bottom", fontsize=8 if compact else 9)
    if sol:
        for position, query in zip(POSITIONS, QUERIES):
            value = rows["sol"][query][metric]
            axis.hlines(value, position - 0.42, position + 0.42, color=DARK,
                        linewidth=2, linestyle="--", zorder=3)
            axis.annotate(f"SoL\n{label(value, metric, compact)}",
                          (position + 0.42, value), xytext=(5, 0),
                          textcoords="offset points", va="center",
                          fontsize=9, color=DARK)
    axis.set_xlim(-0.55, POSITIONS[-1] + 0.75)
    axis.set_xticks(POSITIONS, QUERIES, fontsize=11)
    axis.set_title(title, fontsize=12, loc="left")


def legend_handles():
    """Return the legend entries shared by both pages."""
    handles = [Patch(facecolor=color, label=name)
               for _, name, color, _ in METHODS]
    handles.append(Line2D([0], [0], color=DARK, linewidth=2, linestyle="--",
                          label="SoL estimate"))
    return handles


def speedups(axis, rows):
    """Write stock vLLM time over Quail time, and Quail over SoL, per query."""
    for position, query in zip(POSITIONS, QUERIES):
        vllm = rows["stock_vllm"][query]["seconds"]
        now = rows["now"][query]["seconds"]
        sol = rows["sol"][query]["seconds"]
        axis.annotate(
            f"stock vLLM / Quail now: {vllm / now:.2f}x\n"
            f"Quail now / SoL: {now / sol:.1f}x",
            (position, axis.get_ylim()[1]), xytext=(0, -6),
            textcoords="offset points", ha="center", va="top", fontsize=9)


def slide(rows):
    """Write the wide three-panel page for slides."""
    figure, axes = plt.subplots(1, 3, figsize=(13.33, 4.8))
    panel(axes[0], rows, "seconds", "Latency (lower is better)", "seconds",
          log=True, sol=True, compact=True)
    for position, query in zip(POSITIONS, QUERIES):
        vllm = rows["stock_vllm"][query]["seconds"]
        now = rows["now"][query]["seconds"]
        axes[0].annotate(f"Quail {vllm / now:.2f}x faster than vLLM",
                         (position, axes[0].get_ylim()[1]), xytext=(0, -6),
                         textcoords="offset points", ha="center", va="top",
                         fontsize=10, weight="bold", color=BLUE)
    panel(axes[1], rows, "tokens_per_second", "Throughput (higher is better)",
          "input tokens per second", log=True, sol=True, compact=True)
    panel(axes[2], rows, "regret_tokens", "KV regret (lower is better)",
          "tokens", log=True, compact=True)
    figure.suptitle("BIO-5 and FEV-11, Qwen3 4B FP8, sf 0.5, one H100",
                    y=0.985, fontsize=14, weight="bold")
    figure.legend(handles=legend_handles(), loc="upper center", ncol=4,
                  bbox_to_anchor=(0.5, 0.925), fontsize=10, frameon=False)
    figure.text(
        0.5, 0.01,
        "Latency includes planning, not model startup. Throughput is every "
        "prompt's input tokens over query time. KV regret is fresh tokens "
        "minus the token minimum; SoL's is 0 and stock vLLM's is "
        "approximate. SoL: reference-label survivors, unlimited KV.",
        ha="center", va="bottom", fontsize=8, color="#555555")
    figure.subplots_adjust(left=0.06, right=0.97, top=0.78, bottom=0.12,
                           wspace=0.35)
    with PdfPages(SLIDE) as pdf:
        pdf.savefig(figure)
    plt.close(figure)
    print(SLIDE)


def main():
    """Write the one-page PDF and its slide version."""
    workdir = Path(sys.argv[1])
    plt.style.use(HERE / "quail.mplstyle")
    plt.rcParams["figure.autolayout"] = False
    rows = load(workdir)
    figure, axes = plt.subplots(4, 1, figsize=(8, 12.5))
    panel(axes[0], rows, "seconds", "Latency: query time with planning, "
          "without model startup (lower is better)", "seconds", log=True,
          sol=True)
    speedups(axes[0], rows)
    panel(axes[1], rows, "tokens_per_second", "Throughput: every prompt's "
          "input tokens over query time (higher is better)",
          "input tokens per second", log=True, sol=True)
    panel(axes[2], rows, "regret_tokens", "KV regret: fresh tokens minus "
          "the token minimum (lower is better)\nSoL is 0; stock vLLM's is "
          "approximate", "tokens", log=True)
    panel(axes[3], rows, "agreement", "Label agreement with Qwen3 32B "
          "reference labels (higher is better)", "percent", log=False)
    handles = legend_handles()
    figure.suptitle("BIO-5 and FEV-11, Qwen3 4B FP8, sf 0.5, one H100",
                    y=0.985, fontsize=14, weight="bold")
    figure.legend(handles=handles, loc="upper center", ncol=4,
                  bbox_to_anchor=(0.5, 0.965), fontsize=9, frameon=False)
    figure.text(
        0.5, 0.008,
        "Runs on quail-results benchmarks/quailb/: before and stock vLLM "
        "20261001T010217Z-1e7f7ba8; now BIO-5 20261002T005838Z-45fc1a7e, "
        "FEV-11 20261002T020016Z-ae533a0e.\nSoL: quail.speed_of_light_"
        "estimate with Qwen3 32B reference labels as survivors, unlimited "
        "KV (/results/sol/2026-10-01-classify-sf0.5.json); its throughput "
        "uses Quail's input tokens.", ha="center", va="bottom", fontsize=7.5,
        color="#555555")
    figure.subplots_adjust(left=0.11, right=0.97, top=0.91, bottom=0.07,
                           hspace=0.55)
    OUT.parent.mkdir(exist_ok=True)
    with PdfPages(OUT) as pdf:
        pdf.savefig(figure)
    plt.close(figure)
    print(OUT)
    slide(rows)


if __name__ == "__main__":
    main()
