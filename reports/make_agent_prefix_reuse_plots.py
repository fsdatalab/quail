"""Plot the SWE-Next cumulative trace query comparison.

Pull the measured inputs from the quail-results volume, then pass the work
directory to this script:

    W=<workdir>
    modal volume get quail-results benchmarks/quailb/runs/qb_20260831T062218Z_1192cd76/20260831T062218Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families.json $W/quail.json
    modal volume get quail-results stock_vllm/20260831T062218Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json $W/stock.json
    modal volume get quail-results pipelined_vllm/20260831T062218Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json $W/pipelined.json
    uv run --with matplotlib python reports/make_agent_prefix_reuse_plots.py $W
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from quail.bench.evaluate import H100_USD_PER_HOUR

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"
plt.style.use(HERE / "quail.mplstyle")
sys.path.insert(0, str(HERE))
from plot_colors import BLUE, DARK, GRAY, TEAL  # noqa: E402

QUERIES = ("AGENT-1", "AGENT-2")
SYSTEMS = ("Quail", "Stock vLLM", "Pipelined vLLM")
SYSTEM_COLORS = (BLUE, GRAY, TEAL)
EXPECTED_COLLECTION = "gt_77bb8b128743a79aedddaa24c808c3f8"
DOCUMENTS_PER_QUERY = 1772


def load(path):
    with path.open() as source:
        return json.load(source)


def by_query(rows):
    indexed = {row["query"]: row for row in rows}
    if set(indexed) != set(QUERIES):
        raise ValueError("expected AGENT-1 and AGENT-2")
    return indexed


def load_inputs(workdir):
    quail_data = load(workdir / "quail.json")
    stock_data = load(workdir / "stock.json")
    pipelined_data = load(workdir / "pipelined.json")

    if (quail_data["model"] != "qwen3-4b-fp8"
            or quail_data["sf"] != 0.1
            or quail_data["gpus"] != 1
            or quail_data["ground_truth"]["agent"]["collection_id"]
            != EXPECTED_COLLECTION):
        raise ValueError("unexpected Quail configuration")
    for data, baseline, submission in (
        (stock_data, "stock_vllm", "stage-major"),
        (pipelined_data, "pipelined_vllm", "pipelined"),
    ):
        if (data["baseline"] != baseline
                or data["filter_submission"] != submission
                or data["hf_name"] != "Qwen/Qwen3-4B-FP8"
                or data["sf"] != 0.1
                or data["checkpoint"] != "pre-quantized FP8"
                or not data["enable_prefix_caching"]
                or data["ground_truth"]["agent"]["collection_id"]
                != EXPECTED_COLLECTION):
            raise ValueError(f"unexpected {baseline} configuration")

    return {
        "Quail": by_query(quail_data["passes"]["single"]["queries"]),
        "Stock vLLM": by_query(stock_data["results"][0]),
        "Pipelined vLLM": by_query(pipelined_data["results"][0]),
    }


def seconds(system, row):
    return row["runtime_s"] if system == "Quail" else row["total_wall_s"]


def token_counts(system, row):
    if system == "Quail":
        return row["fresh_tokens"], 0
    steps = row["steps"]
    return (sum(step["fresh_tokens"] for step in steps),
            sum(step["cached_tokens"] for step in steps))


def accuracy_counts(_system, row):
    return row["accuracy"]["answer_accuracy"]


def query_metrics(system, row):
    runtime = seconds(system, row)
    fresh, cached = token_counts(system, row)
    accuracy = accuracy_counts(system, row)
    return {
        "time": runtime,
        "throughput": DOCUMENTS_PER_QUERY / runtime,
        "cost": runtime / 3600 * H100_USD_PER_HOUR,
        "accuracy": accuracy["accuracy"],
        "kv_reuse": cached / (fresh + cached),
    }


def aggregate_metrics(system, rows):
    runtime = sum(seconds(system, row) for row in rows)
    fresh, cached = map(sum, zip(
        *(token_counts(system, row) for row in rows), strict=True))
    accuracies = [accuracy_counts(system, row) for row in rows]
    correct = sum(accuracy["correct"] for accuracy in accuracies)
    evaluated = sum(accuracy["evaluated"] for accuracy in accuracies)
    return {
        "time": runtime,
        "throughput": DOCUMENTS_PER_QUERY * len(rows) / runtime,
        "cost": runtime / 3600 * H100_USD_PER_HOUR,
        "accuracy": correct / evaluated,
        "kv_reuse": cached / (fresh + cached),
    }


METRICS = (
    ("time", "query time, seconds", lambda value: f"{value:.1f}"),
    ("throughput", "throughput, documents/second",
     lambda value: f"{value:.2f}"),
    ("cost", "GPU cost, $/query", lambda value: f"${value:.3f}"),
    ("accuracy", "answer accuracy", lambda value: f"{value:.1%}"),
    ("kv_reuse", "prompt tokens served from KV",
     lambda value: f"{value:.1%}"),
)


def _set_limit(axis, key, values):
    if key in {"accuracy", "kv_reuse"}:
        axis.set_ylim(0, 1.12)
    else:
        axis.set_ylim(0, max(values) * 1.25)


def plot_per_query(records):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    width = 0.24
    centers = list(range(len(QUERIES)))
    offsets = (-width, 0, width)

    for axis, (key, label, formatter) in zip(axes, METRICS):
        all_values = []
        for system, color, offset in zip(
                SYSTEMS, SYSTEM_COLORS, offsets, strict=True):
            values = [query_metrics(system, records[system][query])[key]
                      for query in QUERIES]
            all_values.extend(values)
            bars = axis.bar(
                [center + offset for center in centers], values, width,
                color=color, label=system)
            for bar, value in zip(bars, values, strict=True):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + max(values) * 0.035,
                    formatter(value), ha="center", va="bottom",
                    color=DARK, fontsize=8)
        axis.set_xticks(centers)
        axis.set_xticklabels(QUERIES)
        axis.set_ylabel(label)
        _set_limit(axis, key, all_values)

    axes[-1].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "agent_prefix_reuse_per_query.png", dpi=300)
    plt.close(fig)


def plot_aggregate(records):
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.2))
    aggregate = {
        system: aggregate_metrics(
            system, [records[system][query] for query in QUERIES])
        for system in SYSTEMS
    }

    for axis, (key, label, formatter) in zip(axes, METRICS, strict=True):
        values = [aggregate[system][key] for system in SYSTEMS]
        bars = axis.bar(range(len(SYSTEMS)), values, color=SYSTEM_COLORS)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(values) * 0.035,
                formatter(value), ha="center", va="bottom",
                color=DARK, fontsize=8)
        axis.set_xticks([])
        axis.set_ylabel(
            label.replace("query time", "total query time")
            .replace("$/query", "$ for two queries"))
        _set_limit(axis, key, values)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color)
               for color in SYSTEM_COLORS]
    fig.legend(handles, SYSTEMS, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "agent_prefix_reuse_aggregate.png", dpi=300)
    plt.close(fig)


def print_metrics(records):
    print("| Query | System | Time | Throughput | Cost | Accuracy | KV reuse |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for query in QUERIES:
        for system in SYSTEMS:
            metrics = query_metrics(system, records[system][query])
            print(f"| {query} | {system} | {metrics['time']:.2f} s | "
                  f"{metrics['throughput']:.2f} documents/s | "
                  f"${metrics['cost']:.4f} | "
                  f"{metrics['accuracy']:.2%} | "
                  f"{metrics['kv_reuse']:.2%} |")

    print("\n| System | Total time | Overall throughput | Total cost | "
          "Overall accuracy | KV reuse |")
    print("|---|---:|---:|---:|---:|---:|")
    for system in SYSTEMS:
        metrics = aggregate_metrics(
            system, [records[system][query] for query in QUERIES])
        print(f"| {system} | {metrics['time']:.2f} s | "
              f"{metrics['throughput']:.2f} documents/s | "
              f"${metrics['cost']:.4f} | "
              f"{metrics['accuracy']:.2%} | "
              f"{metrics['kv_reuse']:.2%} |")


def main(workdir):
    records = load_inputs(Path(workdir))
    plot_per_query(records)
    plot_aggregate(records)
    print_metrics(records)


if __name__ == "__main__":
    main(sys.argv[1])
