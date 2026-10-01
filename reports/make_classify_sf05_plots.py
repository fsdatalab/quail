r"""Plot the sf 0.5 QUAIL-B classification runs, one PDF per model.

Each PDF shows, for every query, Quail, stock vLLM, and the SoL estimate:
latency, input tokens per second, recomputed KV tokens, and label
agreement. The SoL estimate comes from `make_classify_sol.py`. Pull the
saved files, then run from the repository root:

    W=/tmp/classify-sf05; mkdir -p "$W"
    QWEN=benchmarks/quailb/20261001T010217Z-1e7f7ba8
    DGEMMA=benchmarks/quailb/20261001T010215Z-854f9a9e
    uv run modal volume get quail-results "$QWEN/quail/run.json" \
      "$W/qwen-quail.json"
    uv run modal volume get quail-results "$QWEN/stock_vllm/run.json" \
      "$W/qwen-stock_vllm.json"
    uv run modal volume get quail-results "$DGEMMA/quail/run.json" \
      "$W/dgemma-quail.json"
    uv run modal volume get quail-results \
      sol/2026-10-01-classify-sf0.5.json "$W/sol_reference_sf0.5.json"
    uv run --with matplotlib python reports/make_classify_sf05_plots.py "$W"
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from plot_colors import BLUE, DARK, ORANGE

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"
METHODS = [("quail", "Quail", BLUE),
           ("stock_vllm", "Stock vLLM (operator-at-a-time)", ORANGE)]
QUERIES = ([f"IMDB-{n}" for n in range(11, 16)]
           + ["BIO-5", "BIO-6", "FEV-11", "LEP-6"]
           + [f"AGENT-{n}" for n in range(3, 6)])
MODELS = [
    ("qwen", "Qwen3 4B FP8", "classify_sf05_qwen3_4b",
     "20261001T010217Z-1e7f7ba8"),
    ("dgemma", "DiffusionGemma 26B-A4B FP8", "classify_sf05_dgemma",
     "20261001T010215Z-854f9a9e"),
]
plt.rcParams["pdf.fonttype"] = 42


def query_metrics(query):
    """Derive the four plotted metrics from one run.json query record."""
    m = query["metrics"]
    label = m["accuracy"].get("label_accuracy") or {}
    return {
        "seconds": m["runtime_s"],
        "input_tokens": m["input_tokens"],
        "tokens_per_second": m["input_tokens"] / m["runtime_s"],
        "regret_tokens": m["regret_tokens"],
        "regret_approximate": m["regret_approximate"],
        "label_agreement": (100 * label["correct"] / label["evaluated"]
                            if label.get("evaluated") else None),
    }


def load(workdir, key):
    """Read each method's run.json and the SoL estimate for one model."""
    rows, reference = {}, None
    for method, _, _ in METHODS:
        path = workdir / f"{key}-{method}.json"
        rows[method] = {}
        if path.exists():
            run = json.loads(path.read_text())
            reference = run["reference_model"]
            rows[method] = {q["id"]: query_metrics(q) for q in run["queries"]
                            if q["status"] == "complete"}
    sol = json.loads((workdir / "sol_reference_sf0.5.json").read_text())[key]
    rows["sol"] = {q: {"seconds": s["sol_s"],
                       "tokens_per_second":
                           rows["quail"][q]["input_tokens"] / s["sol_s"],
                       "regret_tokens": 0}
                   for q, s in sol.items()}
    return rows, reference


def short(value, metric):
    """Format a bar label."""
    if metric == "label_agreement":
        return f"{value:.1f}%"
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    if value >= 1e4:
        return f"{value / 1e3:.0f}k"
    if metric == "regret_tokens":
        return f"{value:,.0f}"
    return f"{value:.1f}"


def bars(axis, rows, metric, title, unit, log=False, sol=False):
    """Draw grouped bars per query; x marks a missing value, a dash a zero."""
    width = 0.82 / len(METHODS)
    values = [v for method, _, _ in METHODS for q in QUERIES
              if (v := rows[method].get(q, {}).get(metric))]
    lines = [rows["sol"][q][metric] for q in QUERIES if sol]
    top = max(values + lines)
    if log:
        bottom = min(values + lines) / 2
        axis.set_yscale("log")
        axis.set_ylim(bottom, top * 12)
        axis.set_ylabel(f"{unit} (log scale)")
    else:
        bottom = 0
        axis.set_ylim(0, min(top * 1.3, 112))
        axis.set_ylabel(unit)
    for index, (method, _, color) in enumerate(METHODS):
        for position, query in enumerate(QUERIES):
            x = position - 0.41 + (index + 0.5) * width
            value = rows[method].get(query, {}).get(metric)
            if not value:
                axis.plot(x, 0.02, marker="x" if value is None else "_",
                          color=color, markersize=6, linestyle="none",
                          transform=axis.get_xaxis_transform(), clip_on=False)
                continue
            axis.bar(x, value - bottom, bottom=bottom, width=width * 0.9,
                     color=color, edgecolor="none")
            axis.annotate(short(value, metric), (x, value), xytext=(0, 3),
                          textcoords="offset points", rotation=90, ha="center",
                          va="bottom", fontsize=7.5)
    if sol:
        for position, query in enumerate(QUERIES):
            axis.hlines(rows["sol"][query][metric], position - 0.41,
                        position + 0.41, color=DARK, linewidth=1.8, zorder=3)
    axis.set_xlim(-0.6, len(QUERIES) - 0.4)
    axis.set_xticks(range(len(QUERIES)), QUERIES, rotation=35, ha="right")
    axis.set_title(title, fontsize=12)


def ratios(axis, rows):
    """Write stock vLLM time over Quail time, and Quail over SoL."""
    for position, query in enumerate(QUERIES):
        quail = rows["quail"][query]["seconds"]
        text = f"SoL {rows['sol'][query]['seconds']:.1f} s"
        if rows["stock_vllm"]:
            vllm = rows["stock_vllm"][query]["seconds"]
            text = f"vLLM/Quail {vllm / quail:.2f}x\n" + text
        top = max(quail, rows["stock_vllm"].get(query, {}).get("seconds", 0))
        axis.annotate(text, (position, top), xytext=(0, 34),
                      textcoords="offset points", ha="center", fontsize=7.5)


def plot_model(workdir, key, name, filename, run_id):
    """Write one model's one-page PDF of the four metrics."""
    rows, reference = load(workdir, key)
    figure, axes = plt.subplots(4, 1, figsize=(14, 24))
    bars(axes[0], rows, "seconds", "Latency: query time with planning, "
         "without model startup", "seconds", log=True, sol=True)
    ratios(axes[0], rows)
    bars(axes[1], rows, "tokens_per_second", "Throughput: every prompt's "
         "full input length over query time", "input tokens per second",
         log=True, sol=True)
    approximate = any(m["regret_approximate"]
                      for m in rows["stock_vllm"].values())
    bars(axes[2], rows, "regret_tokens", "KV regret: fresh tokens minus the "
         "token minimum (SoL is 0 by definition)"
         + (" ; stock vLLM's is approximate" if approximate else ""),
         "tokens", log=True)
    bars(axes[3], rows, "label_agreement",
         f"Accuracy: label agreement with {reference} (no SoL accuracy)",
         "percent")
    handles = []
    for method, label, color in METHODS:
        handles.append(Patch(facecolor=color, label=label) if rows[method]
                       else Line2D([0], [0], marker="x", color=color,
                                   linestyle="none", label=f"{label}: not run"))
    handles.append(Line2D([0], [0], color=DARK, linewidth=1.8,
                          label="SoL estimate (distinct prefixes once, "
                                "unlimited KV, roofline)"))
    height = figure.get_figheight()
    figure.suptitle(f"{name}, sf 0.5, one H100 per engine", y=1 - 0.25 / height,
                    fontsize=16, weight="bold")
    figure.legend(handles=handles, loc="upper center", ncol=len(handles),
                  bbox_to_anchor=(0.5, 1 - 0.6 / height), fontsize=10)
    figure.text(0.5, 0.15 / height,
                f"Run {run_id}. SoL: quail.speed_of_light_estimate "
                f"(fsdatalab/quail#180) with {reference} labels as "
                "survivors; its throughput uses "
                "Quail's input tokens. A dash marks zero; an x marks a "
                "missing value.",
                ha="center", fontsize=9, color="#555555")
    figure.subplots_adjust(left=0.07, right=0.98, top=1 - 1.45 / height,
                           bottom=1.0 / height, hspace=0.45)
    with PdfPages(OUT / f"{filename}.pdf") as pdf:
        pdf.savefig(figure)
    plt.close(figure)
    return OUT / f"{filename}.pdf"


def main():
    """Write both models' PDFs."""
    workdir = Path(sys.argv[1])
    plt.style.use(HERE / "quail.mplstyle")
    plt.rcParams["figure.autolayout"] = False
    OUT.mkdir(exist_ok=True)
    for model in MODELS:
        print(plot_model(workdir, *model))


if __name__ == "__main__":
    main()
