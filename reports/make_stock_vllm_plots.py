"""Make the join performance and SoL plot for the stock vLLM report.

Pull the five source files into one work directory, then pass that directory
as the first argument to this script. For example, with ``W=/tmp/quail-pr52``:

    modal volume get quail-results /benchmarks/quailb/runs/qb_20260826T061917Z_2a6a3ed0/20260826T061917Z-quailb-sf0.1-lf1-qwen3-4b-fp8.json $W/quail-4b.json
    modal volume get quail-results /benchmarks/quailb/runs/qb_20260826T062254Z_9843d222/20260826T062254Z-quailb-sf0.1-lf1-qwen3-32b-fp8.json $W/quail-32b.json
    modal volume get quail-results /stock_quailb/2026-08-26_202641_qwen3-4b-fp8/summary.json $W/stock-4b.json
    modal volume get quail-results /stock_quailb/2026-08-26_202641_qwen3-32b-fp8/summary.json $W/stock-32b.json
    modal volume get quail-results /sol/sol_quailb_sf0.1.json $W/sol.json
    uv run --with matplotlib python reports/make_stock_vllm_plots.py $W
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"

plt.style.use(HERE / "quail.mplstyle")
sys.path.insert(0, str(HERE))
from plot_colors import BLUE, DARK, GRAY, ORANGE  # noqa: E402

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


def load_stock(path: Path, expected_model: str) -> dict[str, float]:
    data = load_json(path)
    if data["model"] != expected_model or data["sf"] != 0.1:
        raise ValueError(f"unexpected stock vLLM result metadata in {path}")
    if data["gpu"] != "H100!":
        raise ValueError(f"expected gpu='H100!' in {path}, got {data['gpu']!r}")
    if data.get("runner") != "baselines.stock.run_join_grouped":
        raise ValueError(f"unexpected stock vLLM runner in {path}")
    if data.get("weight_dtype") != "fp8":
        raise ValueError(f"expected FP8 model weights in {path}")
    if data.get("kv_cache_dtype") != "bfloat16":
        raise ValueError(f"expected BF16 KV in {path}")
    by_operator = {row["operator"]: row for row in data["queries"]}
    if any("warmup_wall_time_s" not in by_operator[OPERATORS[query_id]]
           for query_id in QUERY_IDS):
        raise ValueError(f"expected warmed stock vLLM results in {path}")
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
            "Stock vLLM": load_stock(
                workdir / "stock-4b.json", "qwen3-4b-fp8"),
        },
        "32B": {
            "SoL estimate": load_sol(
                workdir / "sol.json", "qwen3-32b-fp8"),
            "Quail": load_quail(
                workdir / "quail-32b.json", "qwen3-32b-fp8"),
            "Stock vLLM": load_stock(
                workdir / "stock-32b.json", "qwen3-32b-fp8"),
        },
    }

    systems = ["SoL estimate", "Quail", "Stock vLLM"]
    colors = [GRAY, BLUE, ORANGE]
    x = np.arange(len(QUERY_IDS))
    width = 0.23
    fig, axes = plt.subplots(
        1, 2, figsize=(10.5 * OUTPUT_SCALE, 4.2 * OUTPUT_SCALE),
        sharey=True)

    for ax, (model, model_results) in zip(axes, results.items()):
        for index, (system, color) in enumerate(zip(systems, colors)):
            values = [model_results[system][query_id]
                      for query_id in QUERY_IDS]
            bars = ax.bar(x + (index - 1) * width, values, width,
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
    output = OUT / "stock_vllm_vs_quail.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_stock_vllm_plots.py WORKDIR")
    main(Path(sys.argv[1]))
