"""Make the join performance and SoL plot for the vLLM baseline report.

Pull the seven source files into one work directory, then pass that directory
as the first argument to this script. For example, with ``W=/tmp/quail-pr52``:

    modal volume get quail-results /benchmarks/quailb/runs/qb_20260826T061917Z_2a6a3ed0/20260826T061917Z-quailb-sf0.1-lf1-qwen3-4b-fp8.json $W/quail-4b.json
    modal volume get quail-results /benchmarks/quailb/runs/qb_20260826T062254Z_9843d222/20260826T062254Z-quailb-sf0.1-lf1-qwen3-32b-fp8.json $W/quail-32b.json
    modal volume get quail-results /vllm_opbench/2026-08-26_071748/summary.json $W/naive-4b.json
    modal volume get quail-results /vllm_opbench/2026-08-26_071907/summary.json $W/naive-32b.json
    modal volume get quail-results /vllm_opbench/2026-08-26_071911/summary.json $W/stock-4b.json
    modal volume get quail-results /vllm_opbench/2026-08-26_071944/summary.json $W/stock-32b.json
    modal volume get quail-results /sol/sol_quailb_sf0.1.json $W/sol.json
    uv run --with matplotlib python reports/old/make_vllm_opbench_plots.py $W
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent
OUT = HERE / "plots"

plt.style.use(REPORTS / "quail.mplstyle")
sys.path.insert(0, str(REPORTS))
from plot_colors import BLUE, DARK, GRAY, ORANGE, RED  # noqa: E402

OUTPUT_SCALE = 2
for key in ("font.size", "axes.labelsize", "axes.titlesize",
            "axes.titlepad", "xtick.labelsize", "ytick.labelsize",
            "legend.fontsize"):
    plt.rcParams[key] = float(plt.rcParams[key]) * OUTPUT_SCALE

QUERY_IDS = ["BIO-2", "FEV-2", "IMDB-2"]
OPERATORS = {
    "BIO-2": "REACTION",
    "FEV-2": "SUPPORT",
    "IMDB-2": "DISCUSS_ASPECT",
}
QUERY_LABELS = {
    "BIO-2": "BIO-2\n122,800 pairs",
    "FEV-2": "FEV-2\n5,700 pairs",
    "IMDB-2": "IMDB-2\n60,000 pairs",
}


def load_json(path: Path) -> dict:
    with path.open() as source:
        return json.load(source)


def load_quail(path: Path, expected_model: str) -> dict[str, float]:
    data = load_json(path)
    if data["model"] != expected_model or data["sf"] != 0.1:
        raise ValueError(f"unexpected Quail result metadata in {path}")
    cold = {row["query"]: row for row in data["passes"]["cold"]["queries"]}
    return {query_id: cold[query_id]["wall_s"] for query_id in QUERY_IDS}


def load_baseline(path: Path, expected_model: str) -> dict[str, float]:
    data = load_json(path)
    if data["model"] != expected_model or data["sf"] != 0.1:
        raise ValueError(f"unexpected vLLM result metadata in {path}")
    if data["gpu"] != "H100!":
        raise ValueError(f"expected gpu='H100!' in {path}, got {data['gpu']!r}")
    by_operator = {row["operator"]: row for row in data["queries"]}
    return {
        query_id: by_operator[OPERATORS[query_id]]["generate_wall_time_s"]
        for query_id in QUERY_IDS
    }


def load_sol(path: Path, expected_model: str) -> dict[str, float]:
    data = load_json(path)
    if data["scale_factor"] != 0.1:
        raise ValueError(f"unexpected SoL result metadata in {path}")
    return {
        query_id: data["queries"][query_id]["models"][expected_model]["sol_s"]
        for query_id in QUERY_IDS
    }


def main(workdir: Path) -> None:
    results = {
        "4B": {
            "SoL estimate": load_sol(
                workdir / "sol.json", "qwen3-4b-fp8"),
            "Quail": load_quail(workdir / "quail-4b.json", "qwen3-4b-fp8"),
            "Naive vLLM": load_baseline(
                workdir / "naive-4b.json", "qwen3-4b"),
            "Stock vLLM": load_baseline(
                workdir / "stock-4b.json", "qwen3-4b-stock"),
        },
        "32B": {
            "SoL estimate": load_sol(
                workdir / "sol.json", "qwen3-32b-fp8"),
            "Quail": load_quail(
                workdir / "quail-32b.json", "qwen3-32b-fp8"),
            "Naive vLLM": load_baseline(
                workdir / "naive-32b.json", "qwen3-32b"),
            "Stock vLLM": load_baseline(
                workdir / "stock-32b.json", "qwen3-32b-stock"),
        },
    }

    systems = ["SoL estimate", "Quail", "Naive vLLM", "Stock vLLM"]
    colors = [GRAY, BLUE, RED, ORANGE]
    x = np.arange(len(QUERY_IDS))
    width = 0.19
    fig, axes = plt.subplots(
        1, 2, figsize=(10.5 * OUTPUT_SCALE, 4.2 * OUTPUT_SCALE),
        sharey=True)

    for ax, (model, model_results) in zip(axes, results.items()):
        for index, (system, color) in enumerate(zip(systems, colors)):
            values = [model_results[system][query_id]
                      for query_id in QUERY_IDS]
            bars = ax.bar(x + (index - 1.5) * width, values, width,
                          color=color, label=system)
            for bar, query_id, value in zip(bars, QUERY_IDS, values):
                label = f"{value:.1f}s"
                ax.annotate(
                    label,
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 4 * OUTPUT_SCALE),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5 * OUTPUT_SCALE,
                    color=DARK,
                )

        ax.set_yscale("log")
        ax.set_ylim(0.35, 650)
        ax.set_xticks(x)
        ax.set_xticklabels([QUERY_LABELS[query_id]
                            for query_id in QUERY_IDS])
        ax.set_title(model)
        ax.set_ylabel("Time (seconds, log scale)")

    axes[1].set_ylabel("")
    axes[0].legend(loc="upper right", ncols=2)
    OUT.mkdir(exist_ok=True)
    output = OUT / "vllm_opbench_vs_quail.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_vllm_opbench_plots.py WORKDIR")
    main(Path(sys.argv[1]))
