"""Generate the QUAIL-B overview and dataset plots without inference.

Pull the original suite manifest and its four result files:

    W=/tmp/quail-shared-comparison; mkdir -p "$W"
    uv run modal volume get quail-results \
      benchmarks/quailb/family-runs/20260905T021527Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/manifest.json \
      "$W/manifest.json"
    for method in quail stock_vllm pipelined_vllm pipelined_sglang; do
      source_path=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["result_volume_paths"][sys.argv[2]].removeprefix("/results/"))' "$W/manifest.json" "$method")
      uv run modal volume get quail-results "$source_path" "$W/$method.json"
    done
    uv run modal volume get quail-results \
      ground_truth/quailb/schema_v1/corpora/c_1aa2c4f0d0b6c816fd37aa5748c33341/manifest.json \
      "$W/corpus.json"
    uv run modal volume get quail-results \
      /ablations/shared-kv-retention-20260906T054932Z "$W"
    uv run modal volume get quail-results \
      /sol/2026-09-06-quailb-prefix-reuse.json "$W/sol.json"
    uv run --with matplotlib python reports/make_quailb_comparison_plots.py "$W"

Use --retention-dir to point to an already pulled retention comparison.
The current FEV-9 replaces the old Quail measurement. The old baseline FEV-9
measurements are excluded because they used a different query definition.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_colors import BLUE, DARK, GRAY, GREEN, ORANGE
from quail.bench.evaluate import H100_USD_PER_HOUR


HERE = Path(__file__).resolve().parent
METHODS = [
    ("quail", "Quail", BLUE),
    ("stock_vllm", "Stock vLLM", GRAY),
    ("pipelined_vllm", "Pipelined vLLM", ORANGE),
    ("pipelined_sglang", "Pipelined SGLang", GREEN),
]


def row_metrics(row):
    """Derive throughput, GPU cost, and accuracy from saved counts."""
    joins = [stage for stage in row["stages"] if stage["op"] == "join"]
    count = sum(stage["tuples"] for stage in joins) if joins else row["input_document_rows"]
    answers = row["accuracy"]["answer_accuracy"]
    output = row["accuracy"]["output_accuracy"]
    matched = output["matching_rows"]
    predicted = output["predicted_rows"]
    expected = output["expected_rows"]
    return {
        "seconds": row["wall_s"],
        "recomputed": row["regret_tokens"],
        "fresh": row["fresh_tokens"],
        "throughput": count / row["wall_s"],
        "unit": "pairs/s" if joins else "docs/s",
        "cost": row["wall_s"] / 3600 * H100_USD_PER_HOUR,
        "agreement": 100 * answers["correct"] / answers["evaluated"],
        "precision": 100 * matched / predicted if predicted else (0 if expected else 100),
        "recall": 100 * matched / expected if expected else (0 if predicted else 100),
    }


METRICS = (
    ("seconds", "Latency", "seconds"),
    ("recomputed", "Recomputed KV", "tokens"),
    ("fresh", "Fresh input tokens", "tokens"),
    ("agreement", "Answer agreement", "percent"),
)


def input_relations(queries, corpus):
    """Resolve each query alias to its saved input table count."""
    import pyarrow as pa

    import quail
    from quail.bench.evaluate import CORPUS_COLUMNS
    from quail.bench.quailb import queries as builders
    from quail.catalog import DocumentProvider
    from quail.planner import collect_operators

    relations = {}
    with quail.Session(tokenizer=lambda text: []) as session:
        for name, columns in CORPUS_COLUMNS.items():
            session.register(name, DocumentProvider.from_table(
                pa.table({column: [] for column in columns}), id_col="id"))
        available = builders(session)
        for query in queries:
            scans, _, _ = collect_operators(available[query][1]().logical)
            relations[query] = [(scan.alias, scan.provider, corpus["tables"][scan.provider]["rows"])
                                for scan in scans]
    return relations


def load_sol(root, queries, rows, corpus):
    """Load compatible estimates with prefix reuse across requests."""
    source = json.loads((root / "sol.json").read_text())
    assert source["corpus_id"] == corpus["corpus_id"]
    assert source["scale_factor"] == 0.1
    assert source["optimizer"]["persistent_kv_capacity"] == "unlimited"
    assert "across documents" in source["estimates"]["sol_s"]
    estimates = {}
    for query in queries:
        estimate = source["queries"][query]["models"]["qwen3-4b-fp8"]
        measured = rows["quail"][query]
        expected_filters = [(p["alias"], p["predicate_key"])
                            for p in measured["accuracy"]["per_predicate"] if p["op"] == "filter"]
        expected_joins = [p["predicate_key"] for p in measured["accuracy"]["per_predicate"]
                          if p["op"] == "join"]
        assert sorted((s["alias"], s["code"]) for s in estimate["filter_stages"]) == sorted(expected_filters), query
        assert sorted(s["code"] for s in estimate["join_stages"]) == sorted(expected_joins), query
        assert estimate["input_document_rows"] == measured["input_document_rows"], query
        assert estimate["tokens"] <= estimate["per_document"]["tokens"], query
        estimates[query] = estimate
    return estimates


def series_value(rows, sol, method, query, metric):
    """Return a measured value or an explicitly modeled value."""
    if method == "sol":
        return {"seconds": sol[query]["sol_s"], "fresh": sol[query]["tokens"],
                "recomputed": 0, "agreement": None}[metric]
    if query not in rows[method]:
        return None
    return row_metrics(rows[method][query])[metric]


def metric_bars(axis, queries, rows, sol, metric, overview):
    """Draw measured bars and a SoL line across each query group."""
    methods = METHODS + ([] if metric == "agreement" else [("sol", "SoL estimate", DARK)])
    positive = [value for key, _, _ in methods for query in queries
                if (value := series_value(rows, sol, key, query, metric)) is not None and value > 0]
    maximum = max(positive, default=0)
    logarithmic = metric != "agreement" and positive and maximum / min(positive) > 10
    if metric == "agreement":
        axis.set_ylim(0, 122)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.set_ylabel("percent")
    elif logarithmic:
        if metric == "recomputed":
            axis.set_yscale("symlog", linthresh=1)
            axis.set_ylim(0, maximum * 30)
            axis.set_ylabel("tokens (linear to 1, then log)")
        else:
            axis.set_yscale("log")
            axis.set_ylim(min(positive) / 2, maximum * 8)
            axis.set_ylabel("seconds (log scale)" if metric == "seconds" else "tokens (log scale)")
    else:
        axis.set_ylim(0, maximum * 1.6 if maximum else 1)
        axis.set_ylabel("seconds" if metric == "seconds" else "tokens")
        if not maximum:
            axis.set_yticks([0])
    width = 0.82 / len(METHODS)
    floor = axis.get_ylim()[0]
    for method_index, (key, label, color) in enumerate(METHODS):
        xs, values = [], []
        for index, query in enumerate(queries):
            position = index - 0.41 + (method_index + 0.5) * width
            value = series_value(rows, sol, key, query, metric)
            if value is None or value == 0:
                axis.plot(position, 0.015, marker="x" if value is None else "_",
                          color=color, markersize=4, linestyle="none",
                          transform=axis.get_xaxis_transform(), clip_on=False)
                continue
            xs.append(position)
            values.append(value)
            if not overview:
                shown = f"{value / 1e6:.2f}M" if value >= 1e6 else (
                    f"{value / 1e3:.1f}k" if value >= 1000 else f"{value:.2f}")
                if metric in ("fresh", "recomputed") and value < 1000:
                    shown = f"{value:.0f}"
                axis.annotate(shown, (position, value), xytext=(0, 3),
                              textcoords="offset points", rotation=90, ha="center", va="bottom",
                              fontsize=8)
        axis.bar(xs, [value - floor for value in values], bottom=floor, width=width * 0.9,
                 color=color, label=label, edgecolor="none")
    if metric != "agreement":
        for index, query in enumerate(queries):
            axis.hlines(series_value(rows, sol, "sol", query, metric),
                        index - 0.41, index + 0.41, color=DARK, linewidth=1.7,
                        zorder=3, clip_on=False)
    axis.set_xlim(-0.7, len(queries) - 0.3)
    axis.set_xticks(range(len(queries)),
                   [query + ("*" if query == "FEV-9" else "") for query in queries],
                   rotation=55 if overview else 35, ha="right")
    axis.tick_params(axis="x", labelsize=9)
    titles = {"seconds": "Latency", "recomputed": "Recomputed KV tokens",
              "fresh": "Fresh input tokens", "agreement": "Answer agreement with Qwen3 32B"}
    axis.set_title(titles[metric], fontsize=13)


def document_page(title, queries, relations):
    """Create a readable input-count page for the PDF."""
    figure = plt.figure(figsize=(14, 9))
    figure.suptitle(f"{title}: input documents before filtering", y=0.96, fontsize=16)
    axis = figure.add_axes((0.045, 0.12, 0.91, 0.77))
    axis.axis("off")
    cells = [[query, "; ".join(f"{alias} ({provider}): {count:,}"
                               for alias, provider, count in relations[query])]
             for query in queries]
    table = axis.table(cellText=cells, colLabels=["Query", "Relation alias (set): input documents"],
                       colWidths=[0.10, 0.90], cellLoc="left", colLoc="left", loc="upper left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, _), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_height(min(0.065, 0.95 / (len(cells) + 1)))
        cell.PAD = 0.02
        if row == 0:
            cell.set_text_props(weight="bold")
    figure.text(0.055, 0.075,
                "Each alias lists its full input before filters. Repeated aliases can refer to the same underlying set.\n"
                "Counts come from the saved corpus manifest. SoL uses the same inputs and the saved reference labels for survivors.",
                fontsize=11, linespacing=1.5)
    return figure


def plot_comparison(title, queries, rows, relations, sol, name, overview=False):
    """Export vector PDF pages and a first-page PNG preview."""
    destination = HERE / "plots" / name
    groups = [[metric] for metric, _, _ in METRICS] if overview else [
        ["seconds", "fresh", "recomputed", "agreement"]]
    with PdfPages(destination.with_suffix(".pdf")) as pdf:
        for page, metrics in enumerate(groups):
            figure, axes = plt.subplots(1, 1, figsize=(14, 8.5)) if overview else plt.subplots(
                2, 2, figsize=(14, 10))
            axes = [axes] if overview else list(axes.flat)
            for axis, metric in zip(axes, metrics):
                metric_bars(axis, queries, rows, sol, metric, overview)
            figure.suptitle(f"{title}, Qwen3 4B FP8, sf=0.1, one H100", y=0.97, fontsize=16)
            handles = [Patch(facecolor=color, label=label) for _, label, color in METHODS]
            if metrics != ["agreement"]:
                handles.append(Line2D([0], [0], color=DARK, linewidth=1.7, label="SoL estimate"))
            figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925),
                          ncol=5, fontsize=11, frameon=False)
            footer = ("SoL estimates ideal work with unlimited prefix KV reuse across requests and reference-label survivors. "
                      "It has no measured accuracy.\n"
                      "Fresh tokens include recomputed KV. A dash marks zero; x marks unavailable. "
                      "Stock vLLM uses operator-at-a-time submission.")
            if "FEV-9" in queries:
                footer += "\n* FEV-9 uses four filters and shared retention. Older baseline measurements used a different query and are omitted."
            figure.text(0.055, 0.035, footer, fontsize=10, linespacing=1.5)
            figure.subplots_adjust(left=0.075, right=0.97, top=0.83,
                                   bottom=0.24 if overview else 0.20, hspace=0.60, wspace=0.28)
            pdf.savefig(figure, bbox_inches=None)
            if page == 0:
                figure.savefig(destination, dpi=300, bbox_inches=None)
            plt.close(figure)
        figure = document_page(title, queries, relations)
        pdf.savefig(figure, bbox_inches=None)
        plt.close(figure)
    return destination.name


def load_rows(root, retention_root):
    """Combine saved measurements while excluding obsolete FEV-9 baselines."""
    manifest = json.loads((root / "manifest.json").read_text())
    corpus = json.loads((root / "corpus.json").read_text())
    rows = {}
    for key, _, _ in METHODS:
        suite = json.loads((root / f"{key}.json").read_text())
        assert (suite["model"], suite["sf"], suite["lf"], suite["gpus"]) == (
            "qwen3-4b-fp8", 0.1, 1, 1)
        assert suite["corpus_id"] == corpus["corpus_id"]
        rows[key] = {row["query"]: row for row in suite["passes"]["single"]["queries"]
                     if row["query"] != "FEV-9"}
    current = json.loads((retention_root / "shared" / "summary.json").read_text())
    accuracy = json.loads((retention_root / "accuracy.json").read_text())
    assert accuracy["corpus_tables"] == {
        name: corpus["tables"][name] for name in ("claims", "evidence")}
    assert (current["query"], current["model"], current["sf"], current["lf"], current["gpus"]) == (
        "FEV-9", "qwen3-4b-fp8", 0.1, 1, 1)
    assert set(current["estimated_plan"]["filter_order"]) == {"c1", "c2", "e1", "e2"}
    score = accuracy["configurations"]["shared"]
    assert score["output_accuracy"]["predicted_rows"] == current["rows"]
    rows["quail"]["FEV-9"] = {
        **current["report"], "query": "FEV-9", "accuracy": score,
        "input_document_rows": score["input_document_rows"],
    }
    return manifest, rows, accuracy["source_volume_path"]


def main(workdir, retention_dir=None):
    """Regenerate figures and a report from the original suite files."""
    root = Path(workdir)
    retention_root = Path(retention_dir) if retention_dir else root / "shared-kv-retention-20260906T054932Z"
    manifest, rows, retention_source = load_rows(root, retention_root)
    queries = manifest["query_ids"]
    comparable = [query for query in queries if all(query in rows[key] for key in rows)]
    assert len(queries) == 32 and len(comparable) == 31
    assert all("error" not in row for method in rows.values() for row in method.values())
    faster = sum(rows["quail"][query]["wall_s"] < rows["stock_vllm"][query]["wall_s"]
                 for query in comparable)
    plt.style.use(HERE / "quail.mplstyle")
    plt.rcParams.update({"pdf.fonttype": 42, "figure.autolayout": False, "savefig.bbox": None})
    corpus = json.loads((root / "corpus.json").read_text())
    relations = input_relations(queries, corpus)
    for method in rows.values():
        for query, row in method.items():
            assert sum(count for _, _, count in relations[query]) == row["input_document_rows"]
    sol = load_sol(root, queries, rows, corpus)
    overview = plot_comparison("QUAIL-B", queries, rows, relations, sol, "quailb_main.png", overview=True)
    fev = row_metrics(rows["quail"]["FEV-9"])
    output = rows["quail"]["FEV-9"]["accuracy"]["output_accuracy"]
    lines = [
        "# QUAIL-B comparison from saved results", "",
        "- The main PDF covers all 32 queries with grouped bars and one metric per page.",
        "  Its final page lists input document counts. Each dataset PDF has a page",
        "  of four bar charts and a separate input-count page. Text and marks remain",
        "  vector content when zoomed. The PNGs below are first-page previews.",
        "- The five dataset plots use the same",
        "  method colors and definitions for latency, recomputed KV tokens, fresh",
        "  input tokens, accuracy, and input document counts for every relation alias.",
        "- The other 31 queries reuse the original measurements from September 5, 2026.",
        "  No inference was rerun for this report. These are historical measurements,",
        "  not a measurement of shared retention on every query.",
        "- The setup was Qwen3 4B FP8, sf=0.1, lf=1, and one H100 per configuration.",
        "  Quail and the vLLM configurations shared a physical GPU within each family.",
        "  SGLang used a separate GPU. Stock vLLM used operator-at-a-time submission.",
        "- FEV-9 now has four filters. The old suite had only one filter for FEV-9,",
        "  so its old measurements are excluded. The current Quail measurement appears",
        "  in both the main plot and the FEVER plot. Missing baselines are labeled.",
        "  The [retention report](2026-09-05-shared-kv-retention.md) gives the change details.",
        "- The prediction for this update was that scoring and plotting would need no",
        "  inference. We reused all 124 saved configurations for the other 31 queries.",
        f"- In these saved measurements, Quail was faster than stock vLLM on {faster}",
        f"  of {len(comparable)} comparable queries.",
        "- A horizontal line across each query's bar group shows its SoL estimate.",
        "  SoL models ideal computation and memory traffic with unlimited prefix KV.",
        "  It credits matching token prefixes across requests, documents, and aliases.",
        "  It uses exact reference-label survivors and searches supported left-deep",
        "  join plans. Different answers can change the work done by measured runs,",
        "  so the gap from SoL is not purely execution overhead.",
        "- SoL uses the distinct-prefix estimate, not the per-document-only estimate.",
        "  Its latency, fresh-token count, and zero prefix recomputation are estimates.",
        "  No accuracy is assigned to SoL because it is not a measured model run.",
        "  Matching document prefixes are reusable; a partner suffix after a different",
        "  anchor context is not an identical prefix and is still computed.",
        "- The earlier SoL file used the old FEV-9 definition. We recalculated only",
        "  FEV-9 on the CPU from saved labels and corpus rows. The other 31 estimates",
        "  are unchanged. No GPU inference was run.",
        f"- FEV-9 SoL is {sol['FEV-9']['sol_s']:.3f} seconds with shared-prefix reuse,",
        f"  compared with {sol['FEV-9']['per_document']['sol_s']:.3f} seconds with reuse only",
        "  within each document. These estimates use reference-label survivors.",
        "- Answer agreement measures evaluated calls against saved Qwen3 32B labels.",
        "  Each method can evaluate different calls after its filters and joins.",
        "  Output precision is the fraction of returned rows matching the reference.",
        "  Output recall is the fraction of reference rows returned. High answer",
        "  agreement can coexist with poor final output precision.",
        "- Query time excludes startup. Throughput counts input documents for filters",
        "  and evaluated document pairs across all stages for joins. GPU cost is query",
        f"  seconds divided by 3,600 and multiplied by ${H100_USD_PER_HOUR:.4f}.",
        "- Fresh input tokens are input token positions processed by a model forward",
        "  pass instead of read from existing KV. They include document and prompt",
        "  suffix tokens, and any repeated computation after KV becomes unavailable.",
        "  A repeated token counts again. This is not a count of unique text or",
        "  generated answers. Recomputed KV tokens are part of the fresh-token total.",
        "- Recomputed KV is the saved `regret_tokens` total for reusable prefixes",
        "  of documents or anchors already computed earlier in the query. This uses",
        "  per-document accounting, not the separate distinct-prefix metric.",
        "  Token and latency plots use a log scale when positive values span more",
        "  than one order of magnitude. Recomputed KV retains a linear region to",
        "  include zero. A dash marks zero; x marks an unavailable measurement.",
        "- Document counts come from the saved corpus manifest and describe inputs",
        "  before filtering. Repeated aliases each list their full input count.",
        "  The report tables also show throughput, GPU cost, and final output quality.",
        f"- FEV-9 agrees with the reference on {fev['agreement']:.2f}% of evaluated answers. Its final",
        f"  output matches only {output['matching_rows']:,} reference rows out of {output['predicted_rows']:,} returned rows.",
        f"  The reference has {output['expected_rows']:,} rows, so output precision is approximately {fev['precision']:.8f}%",
        f"  and recall is {fev['recall']:.2f}%. The retention change preserved all answers.", "",
        "[Open the main vector PDF](plots/quailb_main.pdf)", "",
        f"[![QUAIL-B latency preview](plots/{overview})](plots/quailb_main.pdf)", "",
        f"Figure: plots/{overview}", "",
        f"Source manifest on `quail-results`: `{manifest['manifest_volume_path']}`.", "",
        f"Current FEV-9 source on `quail-results`: `{retention_source}/`.", "",
        "SoL estimates on `quail-results`: `/results/sol/2026-09-06-quailb-prefix-reuse.json`.", "",
        "The FEV-9 recalculation is also saved separately at `/results/sol/2026-09-06-fev9-prefix-reuse.json`.", "",
        f"Corpus counts on `quail-results`: `/results/ground_truth/quailb/schema_v1/corpora/{corpus['corpus_id']}/manifest.json`.", "",
        "The manifest lists all four source suite paths. The download commands are",
        "in `reports/make_quailb_comparison_plots.py`.", "",
    ]
    for family in ("IMDB", "BIO", "FEV", "LEP", "AGENT"):
        selected = [query for query in queries if query.startswith(family + "-")]
        name = plot_comparison(f"QUAIL-B {family}", selected, rows, relations, sol, f"quailb_{family.lower()}.png")
        pdf_name = Path(name).with_suffix(".pdf").name
        lines.extend([f"## {family}", "", f"[Open the {family} vector PDF](plots/{pdf_name})", "",
                      f"[![{family} preview](plots/{name})](plots/{pdf_name})", "",
                      f"Figure: plots/{name}", "",
                      "| Query | Input documents by alias and set |", "|---|---|"])
        for query in selected:
            counts = ", ".join(f"{alias} ({provider}) = {count:,}"
                               for alias, provider, count in relations[query])
            lines.append(f"| {query} | {counts} |")
        lines.extend(["",
                      "| Query | Method | Seconds | Recomputed KV tokens | Fresh input tokens | Throughput | Unit | $/query | Answer agreement (%) | Output precision (%) | Output recall (%) |",
                      "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"])
        for query in selected:
            for key, label, _ in METHODS:
                if query not in rows[key]:
                    lines.append(f"| {query} | {label} | Not measured for this query definition | | | | | | | | |")
                    continue
                m = row_metrics(rows[key][query])
                lines.append(
                    f"| {query} | {label} | {m['seconds']:.2f} | {m['recomputed']:,} | {m['fresh']:,} | {m['throughput']:,.2f} "
                    f"| {m['unit']} | {m['cost']:.5f} | {m['agreement']:.2f} "
                    f"| {m['precision']:.5g} | {m['recall']:.5g} |")
            estimate = sol[query]
            throughput = estimate["document_pairs_per_second_at_sol"]
            unit = "pairs/s"
            if throughput is None:
                throughput = estimate["documents_per_second_at_sol"]
                unit = "docs/s"
            lines.append(
                f"| {query} | SoL estimate | {estimate['sol_s']:.3f} | 0 (assumed) "
                f"| {estimate['tokens']:,.0f} | {throughput:,.2f} | {unit} "
                f"| {estimate['cost_usd_per_query_at_sol']:.5f} | Not measured | Not measured | Not measured |")
        lines.append("")
    report = HERE / "2026-09-05-quailb-saved-results.md"
    report.write_text("\n".join(lines))
    print(f"Updated {report}, the main figure, and five dataset figures from 125 saved configurations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir")
    parser.add_argument("--retention-dir")
    args = parser.parse_args()
    main(args.workdir, args.retention_dir)
