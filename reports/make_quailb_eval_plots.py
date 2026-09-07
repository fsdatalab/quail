"""Create the plot and Markdown report for one QUAIL-B evaluation.

uv run --with matplotlib python reports/make_quailb_eval_plots.py \
        --input results/benchmark/<summary>.json \
        --report results/benchmark/<report>.md
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PLOTS = HERE / "plots" / "benchmark"
PLOTS.mkdir(parents=True, exist_ok=True)

plt.style.use(HERE / "quail.mplstyle")
sys.path.insert(0, str(HERE))
from plot_colors import BLUE, GREEN, ORANGE  # noqa: E402


def _successful(data: dict, pass_name: str) -> list[dict]:
    return [row for row in data["passes"][pass_name]["queries"]
            if "error" not in row]


def plot_path_for(data: dict, report_path: Path) -> Path:
    stem = data.get("artifact_stem", report_path.stem)
    return PLOTS / f"{stem}.png"


def make_plot(data: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    pass_name = "warm" if "warm" in data["passes"] else next(
        iter(data["passes"]))
    rows = _successful(data, pass_name)
    labels = [row["query"] for row in rows]
    y = list(range(len(rows)))
    height = max(5.5, 0.33 * len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, height), sharey=True)

    runtime = [row["runtime_s"] for row in rows]
    axes[0].barh(y, runtime, color=BLUE, height=0.62)
    for index, value in enumerate(runtime):
        axes[0].annotate(f"{value:.1f}", (value, index), xytext=(4, 0),
                         textcoords="offset points", va="center", fontsize=8)
    axes[0].set_xlabel("query runtime (s)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)

    cost = [row["inference_cost_per_million_tokens_usd"] for row in rows]
    axes[1].barh(y, cost, color=ORANGE, height=0.62)
    for index, value in enumerate(cost):
        axes[1].annotate(f"${value:.3f}", (value, index), xytext=(4, 0),
                         textcoords="offset points", va="center", fontsize=8)
    axes[1].set_xlabel("inference cost per 1M tokens (USD)")

    accuracy = [100 * row["accuracy"]["answer_accuracy"]["accuracy"]
                for row in rows]
    axes[2].scatter(accuracy, y, color=GREEN, s=25, zorder=3)
    for index, value in enumerate(accuracy):
        axes[2].annotate(f"{value:.1f}%", (value, index), xytext=(4, 0),
                         textcoords="offset points", va="center", fontsize=8)
    axes[2].set_xlim(0, 110)
    axes[2].set_xlabel("answer agreement with ground truth (%)")

    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def _query_table(rows: list[dict]) -> list[str]:
    lines = [
        "| Query | Runtime (s) | Boot (s) | H100 inference cost (USD) | "
        "H100 cost with boot (USD) | Tokens | Cost per 1M tokens | "
        "Documents per second | Answer accuracy | Output F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if "error" in row:
            lines.append(
                f"| {row['query']} | error |  |  |  |  |  |  |  |  |")
            continue
        answer = row["accuracy"]["answer_accuracy"]
        output = row["accuracy"]["output_accuracy"]
        lines.append(
            f"| {row['query']} | {row['runtime_s']:.2f} | "
            f"{float(row.get('boot_s') or 0):.2f} | "
            f"{row['inference_cost_usd']:.6f} | "
            f"{row['cost_with_boot_usd']:.6f} | "
            f"{row['tokens_processed']:,} | "
            f"${row['inference_cost_per_million_tokens_usd']:.4f} | "
            f"{row['documents_per_second']:.2f} | "
            f"{_percent(answer['accuracy'])} | "
            f"{_percent(output['f1'])} |")
    return lines


def write_report(data: dict, input_path: Path,
                 report_path: Path, plot_path: Path) -> None:
    try:
        shown_input = input_path.relative_to(ROOT)
    except ValueError:
        shown_input = input_path
    shown_plot = Path(os.path.relpath(
        plot_path, start=report_path.parent)).as_posix()
    lines = [
        f"# QUAIL-B evaluation for {data['model']}",
        "",
        "## Setup",
        "",
        f"- The scale factor was {data['sf']}.",
        f"- The run used {data['gpus']} H100 GPU or GPUs.",
        f"- The benchmark corpus ID was `{data['corpus_id']}`.",
        ("- The ground truth collection was "
         f"`{data['ground_truth']['collection_id']}`."),
        ("- The ground truth reference model was "
         f"`{data['ground_truth']['reference_model']}`."),
        ("- The H100 price was "
         f"${data['pricing']['h100_usd_per_hour']:.4f} per GPU hour."),
        "- The cost estimate excludes Modal CPU, host memory, and volume costs.",
        ("- Ground truth loading happens before the run starts. It is "
         "excluded from every runtime and cost metric."),
        "",
        "## Prediction",
        "",
        data.get("prediction") or "No prediction was supplied before the run.",
        "",
        "## Measured results",
        "",
        ("The figure shows the warm pass when the run contains one. "
         "The tables below contain every pass."),
        "",
        f"Figure: {shown_plot}",
        "",
    ]
    for pass_name, passed in data["passes"].items():
        summary = passed.get("summary")
        lines.extend([f"### {pass_name.capitalize()} pass", ""])
        if summary:
            lines.extend([
                (f"The pass completed {summary['queries_completed']} queries "
                 f"in {summary['query_runtime_s']:.2f} seconds of query "
                 "runtime."),
                "",
                (f"The queries processed {summary['tokens_processed']:,} "
                 f"tokens and cost ${summary['inference_cost_usd']:.6f} "
                 "in H100 time."),
                "",
                ("The H100 cost including model load and warmup was "
                 f"${summary['cost_with_boot_usd']:.6f}."),
                "",
                ("The answer accuracy across evaluated calls was "
                 f"{_percent(summary['answer_accuracy']['accuracy'])}."),
                "",
            ])
        lines.extend(_query_table(passed["queries"]))
        lines.append("")

    lines.extend([
        "## Metric definitions",
        "",
        ("`runtime_s` is GPU worker query runtime. It excludes corpus "
         "construction, ground truth loading, and local evaluation. "
         "`runtime_with_boot_s` adds model load and warmup time."),
        "",
        ("`pass_wall_s` is host time for the query loop. Ground truth "
         "loading happens before that timer starts."),
        "",
        ("`tokens_processed` is the sum of fresh tokens sent through "
         "model forward calls. Tokens read from KV are not counted again."),
        "",
        ("`documents_per_second` divides the input row count by query "
         "runtime. A self join counts the table once for each alias."),
        "",
        ("`inference_cost_usd` multiplies query runtime by the H100 hourly "
         "price and GPU count. `cost_with_boot_usd` also includes model "
         "load and warmup time."),
        "",
        ("Answer accuracy compares each evaluated model call with its saved "
         "label. Output precision, recall, and F1 compare final query rows "
         "with rows derived from all saved labels."),
        "",
        "## Data",
        "",
        f"- The local aggregate summary is `{shown_input}`.",
        ("- The same aggregate summary is "
         f"`{data['aggregate_volume_path']}` on the `quail-results` "
         "Modal volume."),
        f"- The raw query data is `{data['raw_volume_path']}` on the "
        "`quail-results` Modal volume.",
        "- Ground-truth labels remain under "
        "`/results/ground_truth/quailb/schema_v1`.",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    report_path = Path(args.report).resolve()
    data = json.loads(input_path.read_text())
    plot_path = plot_path_for(data, report_path)
    make_plot(data, plot_path)
    write_report(data, input_path, report_path, plot_path)
    print(f"wrote {report_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
