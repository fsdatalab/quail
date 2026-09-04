"""Plot the 32-query QuailB SF 0.1 Qwen3 4B fp8 benchmark.

Pull the measured family files from the quail-results volume, then pass the
work directory to this script:

    W=<workdir>
    modal volume get quail-results benchmarks/quailb/families/20260831T070520Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/imdb.json $W/imdb.json
    modal volume get quail-results benchmarks/quailb/families/20260831T144252Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/biodex.json $W/biodex.json
    modal volume get quail-results benchmarks/quailb/families/20260831T070520Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/fever.json $W/fever.json
    modal volume get quail-results benchmarks/quailb/families/20260831T070520Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/lepard.json $W/lepard.json
    modal volume get quail-results benchmarks/quailb/families/20260831T070520Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/agent.json $W/agent.json
    modal volume get quail-results sol/sol_quailb_sf0.1.json $W/sol.json
    uv run --with matplotlib python reports/make_quailb_sf01_4b_plots.py $W

The SoL file is the ideal work estimate from reports/make_sol_quailb.py
for all 32 queries.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from quail.bench.evaluate import H100_USD_PER_HOUR

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"
plt.style.use(HERE / "quail.mplstyle")
sys.path.insert(0, str(HERE))
from plot_colors import BLUE, DARK, GRAY, ORANGE, TEAL  # noqa: E402

METHODS = ("Quail", "Stock vLLM", "Pipelined vLLM")
COLORS = (BLUE, ORANGE, TEAL)
FAMILY_FILES = ("imdb.json", "biodex.json", "fever.json", "lepard.json",
                "agent.json")
QUERY_ORDER = (
    "IMDB-1", "IMDB-2", "IMDB-3", "IMDB-4", "IMDB-5",
    "IMDB-6", "IMDB-7", "IMDB-8", "IMDB-9", "IMDB-10",
    "BIO-1", "BIO-2", "BIO-3",
    "FEV-1", "FEV-2", "FEV-3", "FEV-4", "FEV-5", "FEV-6",
    "FEV-7", "FEV-8", "FEV-9",
    "LEP-1", "LEP-2", "LEP-3", "LEP-4", "LEP-5", "LEP-6",
    "LEP-7", "LEP-8", "AGENT-1", "AGENT-2",
)


def load(path):
    """Load one JSON file."""
    with path.open() as source:
        return json.load(source)


def index_rows(rows):
    """Index unique result rows by query ID."""
    indexed = {row["query"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate query IDs")
    return indexed


def load_inputs(workdir):
    """Load and validate the five measured family files."""
    records = {method: {} for method in METHODS}
    for filename in FAMILY_FILES:
        family = load(workdir / filename)
        quail = family["quail"]
        if (quail["model"] != "qwen3-4b-fp8"
                or quail["sf"] != 0.1 or quail["gpus"] != 1):
            raise ValueError(f"unexpected Quail configuration in {filename}")
        records["Quail"].update(index_rows(
            quail["passes"]["single"]["queries"]))
        for method, key, submission in (
            ("Stock vLLM", "stock_vllm", "stage-major"),
            ("Pipelined vLLM", "pipelined_vllm", "pipelined"),
        ):
            report = family["baseline_reports"][key]
            if (report["hf_name"] != "Qwen/Qwen3-4B-FP8"
                    or report["sf"] != 0.1
                    or report["gpu_memory_utilization"] != 0.91
                    or report["max_num_batched_tokens"] != 25_305
                    or report["filter_submission"] != submission):
                raise ValueError(
                    f"unexpected {method} configuration in {filename}")
            if len(report["results"]) != 1:
                raise ValueError(f"expected one {method} repetition")
            records[method].update(index_rows(report["results"][0]))

    expected = set(QUERY_ORDER)
    for method, rows in records.items():
        if set(rows) != expected:
            raise ValueError(f"{method} query coverage does not match")
        errors = [query for query, row in rows.items() if "error" in row]
        if errors:
            raise ValueError(f"{method} has failed queries: {errors}")
        missing = [query for query, row in rows.items()
                   if not isinstance(row.get("regret_tokens"), int)]
        if missing:
            raise ValueError(f"{method} lacks KV regret: {missing}")
    return records


def load_sol(workdir):
    """Load the SoL estimate: seconds and modeled work per covered query."""
    sol = load(workdir / "sol.json")
    if sol["scale_factor"] != 0.1:
        raise ValueError("unexpected SoL scale factor")
    estimates = {}
    for query, record in sol["queries"].items():
        model = record["models"]["qwen3-4b-fp8"]
        pairs = int(model["join_pair_evaluations"])
        estimates[query] = {
            "seconds": float(model["sol_s"]),
            "work": pairs if pairs else int(model["input_document_rows"]),
        }
    return estimates


def wall_seconds(method, row):
    """Return query time without startup."""
    return float(row["wall_s"] if method == "Quail"
                 else row["total_wall_s"])


def stages(method, row):
    """Return the method's physical stage records."""
    return row["stages"] if method == "Quail" else row["steps"]


def is_join_stage(method, stage):
    """Return whether a physical stage evaluates document pairs."""
    return (stage["op"] == "join" if method == "Quail"
            else stage["kind"] == "join")


def has_join(method, row):
    """Return whether a query contains at least one join."""
    return any(is_join_stage(method, stage)
               for stage in stages(method, row))


def query_work(method, row):
    """Return input documents or evaluated document pairs."""
    physical = stages(method, row)
    if has_join(method, row):
        key = "tuples" if method == "Quail" else "n_pairs"
        return sum(int(stage[key]) for stage in physical
                   if is_join_stage(method, stage))
    if method == "Quail":
        return int(row["input_document_rows"])
    first_filter = next(
        stage for stage in physical
        if stage["kind"] in ("filter", "filter_chain"))
    return int(first_filter["n_in"])


def fresh_tokens(method, row):
    """Return measured fresh model input tokens."""
    if method == "Quail":
        return int(row["fresh_tokens"])
    return sum(int(stage.get("fresh_tokens") or 0)
               for stage in stages(method, row))


def answer_counts(row):
    """Return evaluated and correct answer counts."""
    answer = row.get("accuracy", {}).get("answer_accuracy")
    if not answer:
        return 0, 0
    return int(answer["evaluated"]), int(answer["correct"])


def aggregate(records, method):
    """Compute aggregate metrics for one method."""
    rows = records[method]
    total_seconds = sum(wall_seconds(method, rows[query])
                        for query in QUERY_ORDER)
    total_regret = sum(rows[query]["regret_tokens"]
                       for query in QUERY_ORDER)
    total_fresh = sum(fresh_tokens(method, rows[query])
                      for query in QUERY_ORDER)
    filter_queries = [query for query in QUERY_ORDER
                      if not has_join(method, rows[query])]
    join_queries = [query for query in QUERY_ORDER
                    if has_join(method, rows[query])]
    filter_throughput = (
        sum(query_work(method, rows[query]) for query in filter_queries)
        / sum(wall_seconds(method, rows[query])
              for query in filter_queries)
    )
    join_throughput = (
        sum(query_work(method, rows[query]) for query in join_queries)
        / sum(wall_seconds(method, rows[query])
              for query in join_queries)
    )
    evaluated, correct = map(sum, zip(*(
        answer_counts(rows[query]) for query in QUERY_ORDER)))
    return {
        "total_seconds": total_seconds,
        "total_cost": total_seconds * H100_USD_PER_HOUR / 3600,
        "filter_throughput": filter_throughput,
        "join_throughput": join_throughput,
        "total_regret": total_regret,
        "regret_fraction": total_regret / total_fresh,
        "accuracy": correct / evaluated,
    }


def family_name(query):
    """Return the displayed query family."""
    prefix = query.split("-", 1)[0]
    return {
        "IMDB": "IMDB",
        "BIO": "BioDEX",
        "FEV": "FEVER",
        "LEP": "LePaRD",
        "AGENT": "Agent",
    }[prefix]


def print_tables(records, sol):
    """Print the report's derived Markdown tables."""
    summaries = {method: aggregate(records, method) for method in METHODS}
    print("\nAggregate metrics")
    print("| Method | Queries | Total time (s) | Total cost | "
          "Filter throughput | Join throughput | KV regret | "
          "Regret / fresh tokens | Accuracy |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for method in METHODS:
        row = summaries[method]
        print(
            f"| {method} | 32 | {row['total_seconds']:,.2f} | "
            f"${row['total_cost']:.4f} | "
            f"{row['filter_throughput']:,.1f} docs/s | "
            f"{row['join_throughput']:,.1f} pairs/s | "
            f"{row['total_regret']:,} | "
            f"{row['regret_fraction']:.2%} | "
            f"{row['accuracy']:.2%} |"
        )

    print("\nResults by family")
    print("| Family | Queries | Quail (s) | Stock vLLM (s) | "
          "Pipelined vLLM (s) | Stock time / Quail time |")
    print("|---|---:|---:|---:|---:|---:|")
    for family in ("IMDB", "BioDEX", "FEVER", "LePaRD", "Agent"):
        queries = [query for query in QUERY_ORDER
                   if family_name(query) == family]
        values = {
            method: sum(wall_seconds(method, records[method][query])
                        for query in queries)
            for method in METHODS
        }
        print(
            f"| {family} | {len(queries)} | {values['Quail']:,.2f} | "
            f"{values['Stock vLLM']:,.2f} | "
            f"{values['Pipelined vLLM']:,.2f} | "
            f"{values['Stock vLLM'] / values['Quail']:.2f}x |"
        )

    print("\nNonzero KV regret")
    print("| Query | Quail | Stock vLLM | Pipelined vLLM |")
    print("|---|---:|---:|---:|")
    for query in QUERY_ORDER:
        values = [records[method][query]["regret_tokens"]
                  for method in METHODS]
        if any(values):
            print(f"| {query} | {values[0]:,} | {values[1]:,} | "
                  f"{values[2]:,} |")

    print("\nTime relative to SoL")
    print("| Method | Total time (s) | SoL total (s) | Time / SoL | "
          "Median time / SoL | Best query | Worst query |")
    print("|---|---:|---:|---:|---:|---|---|")
    covered = [query for query in QUERY_ORDER if query in sol]
    sol_total = sum(sol[query]["seconds"] for query in covered)
    for method in METHODS:
        ratios = {
            query: wall_seconds(method, records[method][query])
            / sol[query]["seconds"]
            for query in covered
        }
        total = sum(wall_seconds(method, records[method][query])
                    for query in covered)
        best = min(ratios, key=ratios.get)
        worst = max(ratios, key=ratios.get)
        print(
            f"| {method} | {total:,.2f} | {sol_total:,.2f} | "
            f"{total / sol_total:.2f}x | "
            f"{float(np.median(list(ratios.values()))):.2f}x | "
            f"{best} ({ratios[best]:.2f}x) | "
            f"{worst} ({ratios[worst]:.2f}x) |"
        )

    print("\nPer-query metrics")
    print("| Query | Unit | SoL estimate | Quail | Stock vLLM | "
          "Pipelined vLLM |")
    print("|---|---|---:|---:|---:|---:|")
    for query in QUERY_ORDER:
        cells = []
        unit = None
        estimate = sol.get(query)
        for method in METHODS:
            row = records[method][query]
            seconds = wall_seconds(method, row)
            joins = has_join(method, row)
            current_unit = "pairs/s" if joins else "docs/s"
            if unit is not None and current_unit != unit:
                raise ValueError(f"methods disagree on query type for {query}")
            unit = current_unit
            throughput = query_work(method, row) / seconds
            cost = seconds * H100_USD_PER_HOUR / 3600
            versus_sol = (
                f"; {seconds / estimate['seconds']:.2f}x SoL"
                if estimate else ""
            )
            cells.append(
                f"{seconds:.2f} s; {throughput:,.1f} {unit}; "
                f"${cost:.4f}; {row['regret_tokens']:,} regret{versus_sol}"
            )
        sol_cell = (
            f"{estimate['seconds']:.2f} s; "
            f"{estimate['work'] / estimate['seconds']:,.1f} {unit}; "
            f"${estimate['seconds'] * H100_USD_PER_HOUR / 3600:.4f}"
            if estimate else "none"
        )
        print(f"| {query} | {unit} | {sol_cell} | " + " | ".join(cells)
              + " |")


def annotate_bars(ax, bars, formatter, values):
    """Label aggregate bars directly."""
    maximum = max(values)
    for bar, value in zip(bars, values):
        ax.annotate(
            formatter(value),
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=DARK,
        )
    ax.set_ylim(0, maximum * 1.22)


def make_aggregate_plot(records):
    """Plot aggregate time, throughput, accuracy, and KV regret."""
    summaries = {method: aggregate(records, method) for method in METHODS}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))

    bar_panels = (
        (axes[0, 0], "Total query time and cost", "seconds",
         lambda summary: summary["total_seconds"],
         lambda value: (
             f"{value:,.0f} s\n"
             f"${value * H100_USD_PER_HOUR / 3600:.2f}"
         )),
        (axes[0, 1], "Filter-only queries", "documents per second",
         lambda summary: summary["filter_throughput"],
         lambda value: f"{value:,.1f}"),
        (axes[1, 1], "Queries with joins", "document pairs per second",
         lambda summary: summary["join_throughput"],
         lambda value: f"{value:,.0f}"),
        (axes[0, 2], "Recomputed prefix tokens",
         "million recomputed prefix tokens",
         lambda summary: summary["total_regret"] / 1e6,
         lambda value: f"{value:.2f}M"),
        (axes[1, 2], "Share of fresh tokens",
         "regret / fresh tokens (%)",
         lambda summary: 100 * summary["regret_fraction"],
         lambda value: f"{value:.2f}%"),
    )
    for ax, title, ylabel, value_fn, formatter in bar_panels:
        values = [value_fn(summaries[method]) for method in METHODS]
        bars = ax.bar(METHODS, values, color=COLORS, width=0.66)
        annotate_bars(ax, bars, formatter, values)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=14)

    accuracy_ax = axes[1, 0]
    accuracy = [100 * summaries[method]["accuracy"] for method in METHODS]
    x = np.arange(len(METHODS))
    accuracy_ax.scatter(x, accuracy, color=COLORS, s=55, zorder=3)
    for position, value in zip(x, accuracy):
        accuracy_ax.annotate(
            f"{value:.2f}%",
            (position, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=DARK,
        )
    accuracy_ax.set_xticks(x, METHODS, rotation=14)
    accuracy_ax.set_ylabel("weighted answer accuracy (%)")
    accuracy_ax.set_title("Answer accuracy")
    accuracy_margin = max(0.3, (max(accuracy) - min(accuracy)) * 0.8)
    accuracy_ax.set_ylim(
        max(0, min(accuracy) - accuracy_margin),
        min(100, max(accuracy) + accuracy_margin),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=3.0, w_pad=2.2)
    for column, heading in enumerate(
            ("Time, cost, and accuracy", "Throughput", "KV regret")):
        position = axes[0, column].get_position()
        fig.text(
            (position.x0 + position.x1) / 2,
            0.965,
            heading,
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
            color=DARK,
        )
    separator_bottom = min(ax.get_position().y0 for ax in axes[1])
    for column in (0, 1):
        left = axes[0, column].get_position().x1
        right = axes[0, column + 1].get_position().x0
        separator = Line2D(
            ((left + right) / 2, (left + right) / 2),
            (separator_bottom, 0.94),
            transform=fig.transFigure,
            color=GRAY,
            linewidth=0.8,
        )
        fig.add_artist(separator)
    OUT.mkdir(exist_ok=True)
    output = OUT / "quailb_sf01_4b_metrics.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"wrote {output}")


def grouped_bars(ax, queries, records, value_fn, sol=None, sol_fn=None):
    """Draw grouped bars for the requested queries, with SoL marks."""
    x = np.arange(len(queries))
    width = 0.25
    for index, (method, color) in enumerate(zip(METHODS, COLORS)):
        values = [value_fn(method, records[method][query])
                  for query in queries]
        ax.bar(x + (index - 1) * width, values, width,
               label=method, color=color)
    if sol is not None:
        positions = [position for position, query in enumerate(queries)
                     if query in sol]
        values = [sol_fn(sol[queries[position]]) for position in positions]
        # one horizontal mark across each query's three bars
        ax.scatter(positions, values, marker="_", s=420, linewidths=1.8,
                   color=DARK, zorder=4, label="SoL estimate")
    ax.set_xticks(x, queries, rotation=60, ha="right")
    for index in range(1, len(queries)):
        if family_name(queries[index - 1]) != family_name(queries[index]):
            ax.axvline(index - 0.5, color=GRAY, linewidth=0.8, zorder=0)


def make_per_query_plot(records, sol):
    """Plot runtime, throughput, cost, and KV regret per query."""
    filter_queries = [query for query in QUERY_ORDER
                      if not has_join("Quail", records["Quail"][query])]
    join_queries = [query for query in QUERY_ORDER
                    if has_join("Quail", records["Quail"][query])]
    fig, axes = plt.subplots(
        4, 1, figsize=(19, 19),
        gridspec_kw={"height_ratios": [1.4, 1.0, 1.25, 1.4]},
    )

    runtime_ax = axes[0]
    grouped_bars(runtime_ax, QUERY_ORDER, records, wall_seconds,
                 sol, lambda estimate: estimate["seconds"])
    runtime_ax.set_yscale("log")
    runtime_ax.set_ylabel("seconds per query (log scale)")
    runtime_ax.set_title("Query time and cost (dark mark: SoL estimate)")
    cost_rate = H100_USD_PER_HOUR / 3600
    cost_ax = runtime_ax.secondary_yaxis(
        "right",
        functions=(lambda value: value * cost_rate,
                   lambda value: value / cost_rate),
    )
    cost_ax.set_ylabel("USD per query (log scale)")

    filter_ax = axes[1]
    grouped_bars(
        filter_ax,
        filter_queries,
        records,
        lambda method, row: query_work(method, row)
        / wall_seconds(method, row),
        sol,
        lambda estimate: estimate["work"] / estimate["seconds"],
    )
    filter_ax.set_yscale("log")
    filter_ax.set_ylabel("documents per second (log scale)")
    filter_ax.set_title("Filter-only throughput")

    join_ax = axes[2]
    grouped_bars(
        join_ax,
        join_queries,
        records,
        lambda method, row: query_work(method, row)
        / wall_seconds(method, row),
        sol,
        lambda estimate: estimate["work"] / estimate["seconds"],
    )
    join_ax.set_ylabel("document pairs per second")
    join_ax.set_title("Throughput for queries with joins")

    regret_ax = axes[3]
    grouped_bars(
        regret_ax,
        QUERY_ORDER,
        records,
        lambda _method, row: row["regret_tokens"],
    )
    regret_ax.set_yscale("symlog", linthresh=1_000)
    regret_ax.set_ylabel("recomputed prefix tokens (symlog scale)")
    regret_ax.set_title("KV regret compared with unlimited KV")

    handles, labels = runtime_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.0))
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    separator_y = (
        axes[2].get_position().y0 + axes[3].get_position().y1
    ) / 2
    fig.add_artist(Line2D(
        (axes[3].get_position().x0, axes[3].get_position().x1),
        (separator_y, separator_y),
        transform=fig.transFigure,
        color=GRAY,
        linewidth=0.8,
    ))
    OUT.mkdir(exist_ok=True)
    output = OUT / "quailb_sf01_4b_per_query.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"wrote {output}")


def main():
    """Load inputs, print tables, and write both figures."""
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} WORKDIR")
    workdir = Path(sys.argv[1])
    records = load_inputs(workdir)
    sol = load_sol(workdir)
    print_tables(records, sol)
    make_aggregate_plot(records)
    make_per_query_plot(records, sol)


if __name__ == "__main__":
    main()
