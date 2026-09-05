"""Plot the QUAIL-B rerun: four methods, two SoL estimates, two KV regrets.

Every method ran through the same request interface on the refactored
engine. Pull the family run manifest and the per method suite files it
lists, plus the SoL file, into one work directory:

    W=<workdir>; R=benchmarks/quailb/family-runs/RUN_ID
    modal volume get quail-results $R/manifest.json $W/manifest.json
    for m in quail stock_vllm pipelined_vllm pipelined_sglang; do
      p=$(python3 -c "import json; print(json.load(open('$W/manifest.json'))['result_volume_paths']['$m'].removeprefix('/results/'))")
      modal volume get quail-results $p $W/$m.json
    done
    modal volume get quail-results sol/sol_quailb_sf0.1.json $W/sol.json
    uv run --with matplotlib python reports/make_quailb_two_regrets_plots.py $W

A method's file may be missing; it is then left out of every figure and
table. The SoL file carries two estimates per query: sol_s, where each
distinct token prefix in the corpus is computed once, and
per_document.sol_s, where each document is computed once and reused only
across its own questions. The figure marks the distinct prefix estimate;
the tables list both.

Two KV regrets follow the same split. regret_tokens is the per document
regret: a document's own prefix recomputed after an earlier request of
the query computed it. regret_distinct_tokens adds the shared prefix
tokens of every scanned document the engine recomputed and subtracts the
cached tokens it received from other documents' requests. Both come from
the run records; the evaluator computes them from the token store and the
backends' cache accounting.
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
from plot_colors import BLUE, DARK, GRAY, ORANGE, RED, TEAL  # noqa: E402

METHOD_FILES = (
    ("Quail", "quail", BLUE),
    ("Stock vLLM", "stock_vllm", ORANGE),
    ("Pipelined vLLM", "pipelined_vllm", TEAL),
    ("Pipelined SGLang", "pipelined_sglang", RED),
)
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


def load_inputs(workdir):
    """Load the per method suite files the manifest lists."""
    manifest = load(workdir / "manifest.json")
    records = {}
    methods = []
    for label, key, color in METHOD_FILES:
        path = workdir / f"{key}.json"
        if not path.exists():
            continue
        suite = load(path)
        if suite["backend"] != key or suite["sf"] != 0.1 \
                or suite["model"] != "qwen3-4b-fp8" or suite["gpus"] != 1:
            raise ValueError(f"unexpected configuration in {path.name}")
        rows = {}
        for row in suite["passes"]["single"]["queries"]:
            if "error" in row:
                raise ValueError(f"{key} failed {row['query']}: {row['error']}")
            rows[row["query"]] = row
        records[label] = rows
        methods.append((label, color))
    queries = [query for query in QUERY_ORDER
               if all(query in rows for rows in records.values())]
    if not queries:
        raise ValueError("no query is present for every method")
    return manifest, records, methods, queries


def load_sol(workdir):
    """Load both SoL estimates per query."""
    sol = load(workdir / "sol.json")
    if sol["scale_factor"] != 0.1:
        raise ValueError("unexpected SoL scale factor")
    estimates = {}
    for query, record in sol["queries"].items():
        model = record["models"]["qwen3-4b-fp8"]
        pairs = int(model["join_pair_evaluations"])
        estimates[query] = {
            "seconds": float(model["sol_s"]),
            "seconds_per_document": float(model["per_document"]["sol_s"]),
            "work": pairs if pairs else int(model["input_document_rows"]),
        }
    return estimates


def has_join(row) -> bool:
    return any(stage["op"] == "join" for stage in row["stages"])


def query_work(row) -> int:
    """Return input documents or evaluated document pairs."""
    if has_join(row):
        return sum(int(stage["tuples"]) for stage in row["stages"]
                   if stage["op"] == "join")
    return int(row["input_document_rows"])


def distinct_regret(row):
    value = row.get("regret_distinct_tokens")
    return None if value is None else int(value)


def answer_counts(row):
    answer = row.get("accuracy", {}).get("answer_accuracy")
    if not answer:
        return 0, 0
    return int(answer["evaluated"]), int(answer["correct"])


def aggregate(rows, queries, sol):
    """Aggregate one method over the queries every method ran."""
    seconds = {query: float(rows[query]["wall_s"]) for query in queries}
    total = sum(seconds.values())
    fresh = sum(int(rows[query]["fresh_tokens"]) for query in queries)
    regret = sum(int(rows[query]["regret_tokens"]) for query in queries)
    distinct = [distinct_regret(rows[query]) for query in queries]
    measured = [value for value in distinct if value is not None]
    filters = [query for query in queries if not has_join(rows[query])]
    joins = [query for query in queries if has_join(rows[query])]

    def throughput(subset):
        if not subset:
            return 0.0
        return (sum(query_work(rows[query]) for query in subset)
                / sum(seconds[query] for query in subset))

    evaluated, correct = map(sum, zip(*(
        answer_counts(rows[query]) for query in queries)))
    ratios = {query: seconds[query] / sol[query]["seconds"]
              for query in queries}
    ratios_per_document = {
        query: seconds[query] / sol[query]["seconds_per_document"]
        for query in queries}
    return {
        "total_seconds": total,
        "total_cost": total * H100_USD_PER_HOUR / 3600,
        "filter_throughput": throughput(filters),
        "join_throughput": throughput(joins),
        "total_regret": regret,
        "regret_fraction": regret / fresh if fresh else 0.0,
        "distinct_regret": sum(measured),
        "distinct_regret_queries": len(measured),
        "accuracy": correct / evaluated if evaluated else 0.0,
        "sol_ratio": total / sum(sol[query]["seconds"] for query in queries),
        "sol_ratio_per_document": total / sum(
            sol[query]["seconds_per_document"] for query in queries),
        "median_sol_ratio": float(np.median(list(ratios.values()))),
        "median_sol_ratio_per_document": float(
            np.median(list(ratios_per_document.values()))),
        "best": min(ratios, key=ratios.get),
        "worst": max(ratios, key=ratios.get),
        "ratios": ratios,
    }


def family_name(query):
    return {"IMDB": "IMDB", "BIO": "BioDEX", "FEV": "FEVER",
            "LEP": "LePaRD", "AGENT": "Agent"}[query.split("-", 1)[0]]


def print_tables(records, methods, queries, sol):
    """Print the report's derived Markdown tables."""
    summaries = {label: aggregate(records[label], queries, sol)
                 for label, _ in methods}
    print(f"\nAggregate metrics over {len(queries)} queries")
    print("| Method | Total time (s) | Total cost | Filter throughput | "
          "Join throughput | KV regret, per document | "
          "Regret / fresh tokens | KV regret, distinct prefix | Accuracy |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, _ in methods:
        row = summaries[label]
        print(
            f"| {label} | {row['total_seconds']:,.2f} | "
            f"${row['total_cost']:.4f} | "
            f"{row['filter_throughput']:,.1f} docs/s | "
            f"{row['join_throughput']:,.1f} pairs/s | "
            f"{row['total_regret']:,} | {row['regret_fraction']:.2%} | "
            f"{row['distinct_regret']:,} "
            f"({row['distinct_regret_queries']} queries) | "
            f"{row['accuracy']:.2%} |"
        )

    print("\nTime relative to SoL")
    print("| Method | Time / SoL, distinct prefix | Median | "
          "Time / SoL, per document | Median | Best query | Worst query |")
    print("|---|---:|---:|---:|---:|---|---|")
    for label, _ in methods:
        row = summaries[label]
        print(
            f"| {label} | {row['sol_ratio']:.2f}x | "
            f"{row['median_sol_ratio']:.2f}x | "
            f"{row['sol_ratio_per_document']:.2f}x | "
            f"{row['median_sol_ratio_per_document']:.2f}x | "
            f"{row['best']} ({row['ratios'][row['best']]:.2f}x) | "
            f"{row['worst']} ({row['ratios'][row['worst']]:.2f}x) |"
        )

    print("\nResults by family")
    header = " | ".join(f"{label} (s)" for label, _ in methods)
    print(f"| Family | Queries | {header} |")
    print("|---|---:|" + "---:|" * len(methods))
    for family in ("IMDB", "BioDEX", "FEVER", "LePaRD", "Agent"):
        subset = [query for query in queries if family_name(query) == family]
        if not subset:
            continue
        cells = " | ".join(
            f"{sum(float(records[label][query]['wall_s']) for query in subset):,.2f}"
            for label, _ in methods)
        print(f"| {family} | {len(subset)} | {cells} |")

    print("\nPer-query metrics")
    header = " | ".join(label for label, _ in methods)
    print(f"| Query | Unit | SoL, distinct prefix | SoL, per document | {header} |")
    print("|---|---|---:|---:|" + "---:|" * len(methods))
    for query in queries:
        unit = "pairs/s" if has_join(records["Quail"][query]) else "docs/s"
        estimate = sol[query]
        cells = []
        for label, _ in methods:
            row = records[label][query]
            seconds = float(row["wall_s"])
            distinct = distinct_regret(row)
            cells.append(
                f"{seconds:.2f} s; {query_work(row) / seconds:,.1f} {unit}; "
                f"${seconds * H100_USD_PER_HOUR / 3600:.4f}; regret "
                f"{int(row['regret_tokens']):,} per doc, "
                f"{'not measured' if distinct is None else f'{distinct:,}'} "
                f"distinct; {seconds / estimate['seconds']:.2f}x SoL"
            )
        sol_cell = (
            f"{estimate['seconds']:.2f} s; "
            f"{estimate['work'] / estimate['seconds']:,.1f} {unit}; "
            f"${estimate['seconds'] * H100_USD_PER_HOUR / 3600:.4f}"
        )
        print(f"| {query} | {unit} | {sol_cell} | "
              f"{estimate['seconds_per_document']:.2f} s | "
              + " | ".join(cells) + " |")

    print("\nQueries with nonzero per document KV regret")
    print(f"| Query | {header} |")
    print("|---|" + "---:|" * len(methods))
    for query in queries:
        values = [int(records[label][query]["regret_tokens"])
                  for label, _ in methods]
        if any(values):
            print(f"| {query} | " + " | ".join(f"{v:,}" for v in values)
                  + " |")

    print("\nDistinct prefix KV regret and its parts")
    print("| Query | Shared prefix tokens | "
          + " | ".join(f"{label} cross row | {label} distinct"
                       for label, _ in methods) + " |")
    print("|---|---:|" + "---:|---:|" * len(methods))
    for query in queries:
        shared = int(records["Quail"][query]["shared_prefix_tokens"])
        cells = []
        for label, _ in methods:
            row = records[label][query]
            cross = row.get("cross_row_cached_tokens")
            distinct = distinct_regret(row)
            cells.append(
                f"{'not measured' if cross is None else f'{int(cross):,}'}"
                f" | {'not measured' if distinct is None else f'{distinct:,}'}"
            )
        print(f"| {query} | {shared:,} | " + " | ".join(cells) + " |")

    print("\nFastest method per query")
    wins = {label: [] for label, _ in methods}
    for query in queries:
        best = min(methods, key=lambda m: float(records[m[0]][query]["wall_s"]))
        wins[best[0]].append(query)
    for label, _ in methods:
        print(f"- {label}: {len(wins[label])} "
              f"({', '.join(wins[label]) if wins[label] else 'none'})")
    return summaries


def grouped_bars(ax, queries, records, methods, value_fn, sol=None,
                 sol_fn=None):
    """Draw grouped bars per query, with an optional SoL mark."""
    x = np.arange(len(queries))
    width = 0.8 / len(methods)
    offset = (len(methods) - 1) / 2
    for index, (label, color) in enumerate(methods):
        values = [value_fn(records[label][query]) for query in queries]
        shown = [position for position, value in enumerate(values)
                 if value is not None]
        ax.bar(x[shown] + (index - offset) * width,
               [values[position] for position in shown], width,
               label=label, color=color)
    if sol is not None:
        ax.scatter(x, [sol_fn(sol[query]) for query in queries],
                   marker="_", s=420, linewidths=1.8, color=DARK,
                   zorder=4, label="SoL estimate")
    ax.set_xticks(x, queries, rotation=60, ha="right")
    boundaries = [position for position in range(1, len(queries))
                  if family_name(queries[position])
                  != family_name(queries[position - 1])]
    for position in boundaries:
        ax.axvline(position - 0.5, color=GRAY, linewidth=0.6)


def annotate_bars(ax, bars, formatter):
    for bar in bars:
        ax.annotate(formatter(bar.get_height()),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8,
                    xytext=(0, 2), textcoords="offset points")


def make_aggregate_plot(summaries, methods, query_count):
    """Bars per method for the totals the report leads with."""
    panels = (
        ("seconds, all queries", lambda s: s["total_seconds"], "{:,.0f}"),
        ("USD, all queries", lambda s: s["total_cost"], "${:.2f}"),
        ("answer accuracy (%)", lambda s: 100 * s["accuracy"], "{:.1f}"),
        ("documents per second, filter-only queries",
         lambda s: s["filter_throughput"], "{:,.0f}"),
        ("document pairs per second, join queries",
         lambda s: s["join_throughput"], "{:,.0f}"),
        ("time / SoL, distinct prefix", lambda s: s["sol_ratio"], "{:.2f}x"),
        ("million recomputed prefix tokens, per document",
         lambda s: s["total_regret"] / 1e6, "{:.2f}"),
        ("million recomputed prefix tokens, distinct prefix",
         lambda s: s["distinct_regret"] / 1e6, "{:.2f}"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    labels = [label for label, _ in methods]
    colors = [color for _, color in methods]
    for ax, (unit, value_fn, fmt) in zip(axes.flat, panels):
        values = [value_fn(summaries[label]) for label in labels]
        bars = ax.bar(range(len(labels)), values, color=colors)
        annotate_bars(ax, bars, fmt.format)
        ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
        ax.set_ylabel(unit)
    axes[1, 3].set_title("vLLM and SGLang: queries with a recorded value")
    fig.suptitle(f"QUAIL-B, {query_count} queries, Qwen3 4B fp8, one H100")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    output = OUT / "quailb_two_regrets_metrics.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"wrote {output}")


def make_per_query_plot(records, methods, queries, sol):
    """Runtime, throughput, and both regrets per query."""
    filters = [query for query in queries
               if not has_join(records["Quail"][query])]
    joins = [query for query in queries if has_join(records["Quail"][query])]
    fig, axes = plt.subplots(
        5, 1, figsize=(19, 23),
        gridspec_kw={"height_ratios": [1.4, 1.0, 1.25, 1.2, 1.2]},
    )
    runtime_ax = axes[0]
    grouped_bars(runtime_ax, queries, records, methods,
                 lambda row: float(row["wall_s"]),
                 sol, lambda estimate: estimate["seconds"])
    runtime_ax.set_yscale("log")
    runtime_ax.set_ylabel("seconds per query (log scale)")
    runtime_ax.set_title(
        "Query time and cost (dark mark: SoL estimate, distinct prefix)")
    rate = H100_USD_PER_HOUR / 3600
    cost_ax = runtime_ax.secondary_yaxis(
        "right", functions=(lambda v: v * rate, lambda v: v / rate))
    cost_ax.set_ylabel("USD per query (log scale)")

    grouped_bars(axes[1], filters, records, methods,
                 lambda row: query_work(row) / float(row["wall_s"]),
                 sol, lambda e: e["work"] / e["seconds"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("documents per second (log scale)")
    axes[1].set_title("Filter-only throughput")

    grouped_bars(axes[2], joins, records, methods,
                 lambda row: query_work(row) / float(row["wall_s"]),
                 sol, lambda e: e["work"] / e["seconds"])
    axes[2].set_ylabel("document pairs per second")
    axes[2].set_title("Throughput for queries with joins")

    grouped_bars(axes[3], queries, records, methods,
                 lambda row: int(row["regret_tokens"]))
    axes[3].set_yscale("symlog", linthresh=1_000)
    axes[3].set_ylabel("recomputed prefix tokens (symlog scale)")
    axes[3].set_title(
        "KV regret, per document: a document's own prefix recomputed")

    grouped_bars(axes[4], queries, records, methods, distinct_regret)
    axes[4].set_yscale("symlog", linthresh=1_000)
    axes[4].set_ylabel("recomputed prefix tokens (symlog scale)")
    axes[4].set_title(
        "KV regret, distinct prefix: any prefix recomputed that another "
        "document had computed")

    handles, labels = runtime_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.0))
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    separator_y = (axes[2].get_position().y0 + axes[3].get_position().y1) / 2
    fig.add_artist(Line2D(
        (axes[3].get_position().x0, axes[3].get_position().x1),
        (separator_y, separator_y), transform=fig.transFigure,
        color=GRAY, linewidth=0.8))
    OUT.mkdir(exist_ok=True)
    output = OUT / "quailb_two_regrets_per_query.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"wrote {output}")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} WORKDIR")
    workdir = Path(sys.argv[1])
    manifest, records, methods, queries = load_inputs(workdir)
    sol = load_sol(workdir)
    print(f"family run {manifest['run_id']}: "
          f"{len(methods)} methods, {len(queries)} queries")
    summaries = print_tables(records, methods, queries, sol)
    make_aggregate_plot(summaries, methods, len(queries))
    make_per_query_plot(records, methods, queries, sol)


if __name__ == "__main__":
    main()
