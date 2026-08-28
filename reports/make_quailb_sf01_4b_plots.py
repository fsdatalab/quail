"""Plot the complete QuailB SF 0.1 Qwen3 4B fp8 comparison.

Pull the measured inputs from the quail-results volume, then pass the
work directory to this script:

    W=<workdir>
    modal volume get quail-results benchmarks/quailb/runs/qb_20260827T070457Z_fdd34ac9/20260827T070457Z-quailb-sf0.1-lf1-qwen3-4b-fp8-parallel4.json $W/quail-full.json
    modal volume get quail-results benchmarks/quailb/runs/qb_20260827T081139Z_0518bcfc/20260827T081139Z-quailb-sf0.1-lf1-qwen3-4b-fp8-parallel1.json $W/quail-lepard.json
    modal volume get quail-results stock_vllm/2026-08-28_paired_except_bio8/summary.json $W/stock-full.json
    modal volume get quail-results pipelined_vllm/2026-08-28_paired_except_bio8/summary.json $W/pipelined.json
    modal volume get quail-results sol/sol_quailb_sf0.1.json $W/sol.json
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
from plot_colors import BLUE, GRAY, ORANGE, TEAL  # noqa: E402

MODEL = "qwen3-4b-fp8"
LEPARD = [f"LEP-{index}" for index in range(1, 9)]
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
    quail_full = load(workdir / "quail-full.json")
    quail_lepard = load(workdir / "quail-lepard.json")
    stock_full = load(workdir / "stock-full.json")
    pipelined = load(workdir / "pipelined.json")
    sol = load(workdir / "sol.json")

    for data in (quail_full, quail_lepard):
        if data["model"] != MODEL or data["sf"] != 0.1 or data["gpus"] != 1:
            raise ValueError("unexpected Quail configuration")
    if (stock_full["baseline"] != "stock_vllm"
            or stock_full["filter_submission"] != "stage-major"
            or stock_full["hf_name"] != "Qwen/Qwen3-4B-FP8"
            or stock_full["sf"] != 0.1
            or stock_full["checkpoint"] != "pre-quantized FP8"
            or stock_full.get("prompt_layout")
            != "canonical document-first Quail filter and join prompts"):
        raise ValueError("unexpected stock vLLM configuration")
    if (pipelined["baseline"] != "pipelined_vllm"
            or pipelined["filter_submission"] != "pipelined"
            or pipelined["hf_name"] != "Qwen/Qwen3-4B-FP8"
            or pipelined["sf"] != 0.1
            or pipelined["checkpoint"] != "pre-quantized FP8"
            or pipelined.get("prompt_layout")
            != "canonical document-first Quail filter and join prompts"):
        raise ValueError("unexpected pipelined vLLM configuration")
    if sol["scale_factor"] != 0.1 or sol["query_count"] != 35:
        raise ValueError("unexpected SoL configuration")

    order = stock_full["query_ids"]
    if len(order) != 35 or set(LEPARD) - set(order):
        raise ValueError("expected all 35 QuailB queries")

    quail = by_query(quail_full["passes"]["single"]["queries"])
    quail.update(by_query(quail_lepard["passes"]["single"]["queries"]))
    stock = by_query(stock_full["results"][0])
    pipelined_rows = by_query(pipelined["results"][0])
    if (set(order) != set(quail)
            or set(order) != set(stock)
            or set(order) != set(pipelined_rows)):
        raise ValueError("measured query coverage does not match")
    for baseline in (stock_full, pipelined):
        if baseline.get("paired_run", {}).get("missing_queries") != ["BIO-8"]:
            raise ValueError("expected BIO-8 to be the only missing baseline query")

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


def print_metrics(order, results, records):
    print("\nAggregate metrics")
    print("| System | Measured queries | Total time (s) | Mean time/query (s) | "
          "Mean $/query |")
    print("|---|---:|---:|---:|---:|")
    for system in SYSTEMS:
        values = np.asarray(results[system], dtype=float)
        measured = np.isfinite(values)
        total = values[measured].sum()
        mean = values[measured].mean()
        cost = mean * H100_USD_PER_HOUR / 3600
        print(f"| {system} | {measured.sum()} | {total:.2f} | {mean:.2f} | "
              f"${cost:.4f} |")

    print("\nAggregate throughput")
    print("| System | Query group | Measured queries | Throughput |")
    print("|---|---|---:|---:|")
    for group in ("filter only", "join only"):
        for system in ("Quail", "Stock vLLM", "Pipelined vLLM"):
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
            unit = "docs/s" if group == "filter only" else "pairs/s"
            print(f"| {system} | {group} | {measured} | "
                  f"{total_work / total_seconds:,.1f} {unit} |")

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


def main(workdir):
    order, results, records = load_inputs(workdir)
    colors = (GRAY, BLUE, ORANGE, TEAL)
    x = np.arange(len(order))
    width = 0.19

    fig, ax = plt.subplots(figsize=(20, 6))
    for index, (system, color) in enumerate(zip(SYSTEMS, colors)):
        values = results[system]
        offset = index - (len(SYSTEMS) - 1) / 2
        bars = ax.bar(x + offset * width, values, width,
                      color=color, label=system)
        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                continue
            ax.annotate(
                f"{value:.3g}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=5.5,
                rotation=90,
                color=color,
            )

    ax.set_yscale("log")
    ax.set_ylim(0.015, 900)
    ax.set_ylabel("time per query on one H100!, seconds (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.legend(frameon=False, loc="upper left", ncols=4)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    output = OUT / "quailb_sf01_4b_runtime.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"wrote {output}")
    print_metrics(order, results, records)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_quailb_sf01_4b_plots.py WORKDIR")
    main(Path(sys.argv[1]))
