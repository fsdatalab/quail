"""Plot the complete QuailB SF 0.1 Qwen3 4B fp8 comparison.

Pull the measured inputs from the quail-results volume, then pass the
work directory to this script:

    W=<workdir>
    modal volume get quail-results benchmarks/quailb/runs/qb_20260829T185407Z_cbb14b36/20260829T185407Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families.json $W/quail.json
    modal volume get quail-results stock_vllm/20260829T185407Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json $W/stock.json
    modal volume get quail-results pipelined_vllm/20260829T185407Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json $W/pipelined.json
    modal volume get quail-results benchmarks/quailb/runs/qb_20260829T212047Z_dd7f1686/20260829T212047Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families.json $W/quail_lepard.json
    modal volume get quail-results stock_vllm/20260829T212047Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json $W/stock_lepard.json
    modal volume get quail-results pipelined_vllm/20260829T212047Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json $W/pipelined_lepard.json
    modal volume get quail-results /sol/sol_quailb_sf0.1.json $W/sol.json
    uv run --with matplotlib python reports/make_quailb_sf01_4b_plots.py $W
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quail.bench.evaluate import H100_USD_PER_HOUR

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"
plt.style.use(HERE / "quail.mplstyle")
sys.path.insert(0, str(HERE))
from plot_colors import BLUE, DARK, GRAY, ORANGE, TEAL  # noqa: E402

MODEL = "qwen3-4b-fp8"
SYSTEMS = ("SoL estimate", "Quail", "Stock vLLM", "Pipelined vLLM")


def load(path):
    with path.open() as source:
        return json.load(source)


def by_query(rows):
    indexed = {row["query"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate query ids")
    return indexed


def load_inputs(workdir):
    quail_data = load(workdir / "quail.json")
    stock_full = load(workdir / "stock.json")
    pipelined = load(workdir / "pipelined.json")
    quail_lepard = load(workdir / "quail_lepard.json")
    stock_lepard = load(workdir / "stock_lepard.json")
    pipelined_lepard = load(workdir / "pipelined_lepard.json")
    sol = load(workdir / "sol.json")

    if (quail_data["model"] != MODEL or quail_data["sf"] != 0.1
            or quail_data["gpus"] != 1):
        raise ValueError("unexpected Quail configuration")
    if (quail_lepard["model"] != MODEL or quail_lepard["sf"] != 0.1
            or quail_lepard["gpus"] != 1):
        raise ValueError("unexpected revised LePaRD Quail configuration")
    if (stock_full["baseline"] != "stock_vllm"
            or stock_full["filter_submission"] != "stage-major"
            or stock_full["hf_name"] != "Qwen/Qwen3-4B-FP8"
            or stock_full["sf"] != 0.1
            or stock_full["checkpoint"] != "pre-quantized FP8"
            or stock_full["gpu_memory_utilization"] != 0.91):
        raise ValueError("unexpected stock vLLM configuration")
    if (pipelined["baseline"] != "pipelined_vllm"
            or pipelined["filter_submission"] != "pipelined"
            or pipelined["hf_name"] != "Qwen/Qwen3-4B-FP8"
            or pipelined["sf"] != 0.1
            or pipelined["checkpoint"] != "pre-quantized FP8"
            or pipelined["gpu_memory_utilization"] != 0.91):
        raise ValueError("unexpected pipelined vLLM configuration")
    for focused, name, submission in (
        (stock_lepard, "stock_vllm", "stage-major"),
        (pipelined_lepard, "pipelined_vllm", "pipelined"),
    ):
        if (focused["baseline"] != name
                or focused["filter_submission"] != submission
                or focused["hf_name"] != "Qwen/Qwen3-4B-FP8"
                or focused["sf"] != 0.1
                or focused["checkpoint"] != "pre-quantized FP8"
                or focused["gpu_memory_utilization"] != 0.91):
            raise ValueError(
                f"unexpected revised LePaRD {name} configuration")
    if sol["scale_factor"] != 0.1:
        raise ValueError("unexpected SoL configuration")

    order = stock_full["query_ids"]
    if len(order) != 30:
        raise ValueError("expected all 30 current QuailB queries")

    quail = by_query(quail_data["passes"]["single"]["queries"])
    stock = by_query(stock_full["results"][0])
    pipelined_rows = by_query(pipelined["results"][0])
    revised = {
        "Quail": by_query(quail_lepard["passes"]["single"]["queries"]),
        "Stock vLLM": by_query(stock_lepard["results"][0]),
        "Pipelined vLLM": by_query(pipelined_lepard["results"][0]),
    }
    expected_lepard = {f"LEP-{index}" for index in range(1, 9)}
    if any(set(rows) != expected_lepard for rows in revised.values()):
        raise ValueError("revised inputs must contain all eight LePaRD queries")
    quail.update(revised["Quail"])
    stock.update(revised["Stock vLLM"])
    pipelined_rows.update(revised["Pipelined vLLM"])
    if (set(order) != set(quail)
            or set(order) != set(stock)
            or set(order) != set(pipelined_rows)
            or not set(order).issubset(sol["queries"])):
        raise ValueError("measured query coverage does not match")

    def seconds(rows, query):
        value = rows[query]["total_wall_s"]
        return np.nan if value is None else value

    results = {
        "SoL estimate": [sol["queries"][query]["models"][MODEL]["sol_s"]
                         for query in order],
        "Quail": [quail[query]["wall_s"] for query in order],
        "Stock vLLM": [seconds(stock, query) for query in order],
        "Pipelined vLLM": [seconds(pipelined_rows, query)
                            for query in order],
    }
    records = {
        "SoL estimate": {
            query: sol["queries"][query]["models"][MODEL]
            for query in order
        },
        "Quail": quail,
        "Stock vLLM": stock,
        "Pipelined vLLM": pipelined_rows,
    }
    return order, results, records


def query_kind(sol_row):
    has_filters = bool(sol_row["filter_stages"])
    has_joins = bool(sol_row["join_stages"])
    if has_filters and has_joins:
        return "filter + join"
    if has_joins:
        return "join only"
    return "filter only"


def query_work(system, row, has_joins):
    if system == "SoL estimate":
        if has_joins:
            return row["join_pair_evaluations"]
        return row["input_document_rows"]
    if system == "Quail":
        if has_joins:
            return sum(stage["tuples"] for stage in row["stages"]
                       if stage["op"] == "join")
        return row["input_document_rows"]
    if has_joins:
        return sum(stage["n_pairs"] for stage in row["steps"]
                   if stage["kind"] == "join")
    return sum(stage["n_in"] for stage in row["steps"]
               if stage["kind"] in ("filter", "filter_chain"))


def aggregate_throughput(order, results, records, group, system):
    total_work = 0
    total_seconds = 0.0
    measured = 0
    for index, query in enumerate(order):
        if query_kind(records["SoL estimate"][query]) != group:
            continue
        seconds = results[system][index]
        if not np.isfinite(seconds):
            continue
        total_work += query_work(
            system, records[system][query], group == "join only")
        total_seconds += seconds
        measured += 1
    return total_work / total_seconds, measured


def answer_accuracy(records, system):
    evaluated = 0
    correct = 0
    for row in records[system].values():
        answer = row.get("accuracy", {}).get("answer_accuracy")
        if not answer:
            continue
        evaluated += answer["evaluated"]
        correct += answer["correct"]
    return correct / evaluated


def print_metrics(order, results, records):
    print("\nAggregate metrics")
    print("| System | Queries | Total time (s) | Mean time/query (s) | "
          "Mean $/query | Answer accuracy |")
    print("|---|---:|---:|---:|---:|---:|")
    for system in SYSTEMS:
        values = np.asarray(results[system], dtype=float)
        measured = np.isfinite(values)
        total = values[measured].sum()
        mean = values[measured].mean()
        cost = mean * H100_USD_PER_HOUR / 3600
        accuracy = ("not applicable" if system == "SoL estimate" else
                    f"{answer_accuracy(records, system):.2%}")
        print(f"| {system} | {measured.sum()} | {total:.2f} | {mean:.2f} | "
              f"${cost:.4f} | {accuracy} |")

    print("\nAggregate throughput")
    print("| System | Query group | Measured queries | Throughput |")
    print("|---|---|---:|---:|")
    for group in ("filter only", "join only"):
        for system in SYSTEMS:
            throughput, measured = aggregate_throughput(
                order, results, records, group, system)
            unit = "docs/s" if group == "filter only" else "pairs/s"
            print(f"| {system} | {group} | {measured} | "
                  f"{throughput:,.1f} {unit} |")

    print("\nPer-query metrics")
    print("| Query | Operators | Unit | SoL estimate | Quail | "
          "Stock vLLM | Pipelined vLLM |")
    print("|---|---|---|---:|---:|---:|---:|")
    for index, query in enumerate(order):
        sol_row = records["SoL estimate"][query]
        kind = query_kind(sol_row)
        has_joins = kind != "filter only"
        unit = "pairs/s" if has_joins else "docs/s"
        cells = []
        for system in SYSTEMS:
            seconds = results[system][index]
            if not np.isfinite(seconds):
                cells.append("not measured")
                continue
            work = query_work(system, records[system][query], has_joins)
            throughput = work / seconds
            cost = seconds * H100_USD_PER_HOUR / 3600
            cells.append(
                f"{seconds:.2f} s; {throughput:,.1f} {unit}; "
                f"${cost:.4f}")
        print(f"| {query} | {kind} | {unit} | "
              + " | ".join(cells) + " |")


def _finite_mean(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)].mean()


def make_metrics_plot(order, results, records):
    mean_seconds = {
        system: _finite_mean(results[system]) for system in SYSTEMS
    }
    mean_cost = {
        system: seconds * H100_USD_PER_HOUR / 3600
        for system, seconds in mean_seconds.items()
    }
    filter_throughput = {
        system: aggregate_throughput(
            order, results, records, "filter only", system)[0]
        for system in SYSTEMS
    }
    join_throughput = {
        system: aggregate_throughput(
            order, results, records, "join only", system)[0]
        for system in SYSTEMS
    }
    panels = (
        ("Mean time per query", "seconds per query (log scale)",
         mean_seconds, lambda value: f"{value:.2f} s", "higher", True),
        ("Mean cost per query", "USD per query (log scale)",
         mean_cost, lambda value: f"${value:.4f}", "higher", True),
        ("Filter only throughput", "documents per second",
         filter_throughput, lambda value: f"{value:,.0f}", "lower", False),
        ("Join only throughput", "document pairs per second",
         join_throughput, lambda value: f"{value:,.0f}", "lower", False),
    )
    colors = (GRAY, BLUE, ORANGE, TEAL)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, (title, unit, values_by_system, formatter,
             ratio_direction, use_log_scale) in zip(axes.flat, panels):
        values = np.asarray(
            [values_by_system[system] for system in SYSTEMS], dtype=float)
        bars = ax.bar(SYSTEMS, values, color=colors, width=0.68)
        sol = values[0]
        for index, (bar, value) in enumerate(zip(bars, values)):
            if index == 0:
                ratio = ""
            elif ratio_direction == "higher":
                ratio = f"\n{value / sol:.1f}x SoL"
            else:
                ratio = f"\n{value / sol:.2f}x SoL"
            ax.annotate(
                f"{formatter(value)}{ratio}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=DARK,
            )
        ax.set_title(title)
        ax.set_ylabel(unit)
        if use_log_scale:
            ax.set_yscale("log")
            ax.set_ylim(values.min() / 1.8, values.max() * 2.1)
        else:
            ax.set_ylim(0, values.max() * 1.28)
        ax.tick_params(axis="x", labelrotation=16)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    output = OUT / "quailb_sf01_4b_metrics.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"wrote {output}")


def _per_query_throughput(query_ids, order, results, records, has_joins):
    index_by_query = {query: index for index, query in enumerate(order)}
    values = {}
    for system in SYSTEMS:
        system_values = []
        for query in query_ids:
            seconds = results[system][index_by_query[query]]
            work = query_work(system, records[system][query], has_joins)
            if not np.isfinite(seconds):
                system_values.append(np.nan)
            else:
                system_values.append(work / seconds)
        values[system] = np.asarray(system_values, dtype=float)
    return values


def _grouped_bars(ax, labels, values_by_system, ylabel, log_scale):
    colors = (GRAY, BLUE, ORANGE, TEAL)
    x = np.arange(len(labels), dtype=float)
    width = 0.2
    offsets = (np.arange(len(SYSTEMS)) - 1.5) * width
    for system, color, offset in zip(SYSTEMS, colors, offsets):
        values = np.asarray(values_by_system[system], dtype=float)
        positive = np.isfinite(values) & (values > 0)
        ax.bar(x[positive] + offset, values[positive], width=width,
               color=color, label=system)
        for index in np.flatnonzero(~np.isfinite(values)):
            ax.text(x[index] + offset, 0.01, "not\nmeasured", rotation=90,
                    ha="center", va="bottom", fontsize=5.5, color=DARK,
                    transform=ax.get_xaxis_transform())
        for index in np.flatnonzero(values == 0):
            ax.text(x[index] + offset, 0.01, "0", ha="center", va="bottom",
                    fontsize=6, color=DARK,
                    transform=ax.get_xaxis_transform())
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="center")
    if log_scale:
        positive = np.concatenate([
            values[np.isfinite(values) & (values > 0)]
            for values in values_by_system.values()
        ])
        ax.set_yscale("log")
        ax.set_ylim(positive.min() / 1.8, positive.max() * 2.2)


def make_per_query_plot(order, results, records):
    runtime = {
        system: np.asarray(results[system], dtype=float)
        for system in SYSTEMS
    }
    filter_queries = [
        query for query in order
        if query_kind(records["SoL estimate"][query]) == "filter only"
    ]
    join_queries = [
        query for query in order
        if query_kind(records["SoL estimate"][query]) != "filter only"
    ]
    filter_throughput = _per_query_throughput(
        filter_queries, order, results, records, has_joins=False)
    join_throughput = _per_query_throughput(
        join_queries, order, results, records, has_joins=True)

    fig, axes = plt.subplots(
        3, 1, figsize=(16, 14),
        gridspec_kw={"height_ratios": (1.15, 0.8, 1.0)},
    )
    _grouped_bars(
        axes[0], order, runtime,
        "seconds per query (log scale)", log_scale=True)
    axes[0].set_title("Runtime and H100! cost for every query")
    dollars = axes[0].secondary_yaxis(
        "right",
        functions=(
            lambda seconds: seconds * H100_USD_PER_HOUR / 3600,
            lambda cost: cost * 3600 / H100_USD_PER_HOUR,
        ),
    )
    dollars.set_ylabel("USD per query")

    _grouped_bars(
        axes[1], filter_queries, filter_throughput,
        "documents per full-query second (log scale)", log_scale=True)
    axes[1].set_title("Filter only queries")

    _grouped_bars(
        axes[2], join_queries, join_throughput,
        "evaluated document pairs per full-query second (log scale)",
        log_scale=True)
    axes[2].set_title("Queries containing a join")
    axes[0].legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.14), ncol=4)

    fig.tight_layout(h_pad=2.4)
    OUT.mkdir(exist_ok=True)
    output = OUT / "quailb_sf01_4b_per_query.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"wrote {output}")


def main(workdir):
    order, results, records = load_inputs(workdir)
    make_metrics_plot(order, results, records)
    make_per_query_plot(order, results, records)
    print_metrics(order, results, records)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_quailb_sf01_4b_plots.py WORKDIR")
    main(Path(sys.argv[1]))
