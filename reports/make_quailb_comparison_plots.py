"""Generate the QUAIL-B overview and dataset plots without inference.

Pull the measured runs, the corpus manifest, and the SoL estimates:

    W=/tmp/quail-comparison; mkdir -p "$W/run" "$W/fev10" "$W/saved" "$W/ablation"
    RUNS=benchmarks/quailb/family-runs
    RUN=$RUNS/20260912T225100Z-902686c5
    uv run modal volume get quail-results "$RUN/manifest.json" "$W/run/manifest.json"
    uv run modal volume get quail-results "$RUN/measurements.parquet" \
      "$W/run/measurements.parquet"
    FEV10=$RUNS/20260911T201441Z-d16f87d8
    uv run modal volume get quail-results "$FEV10/manifest.json" \
      "$W/fev10/manifest.json"
    uv run modal volume get quail-results "$FEV10/measurements.parquet" \
      "$W/fev10/measurements.parquet"
    result_path() {
      python3 -c "import json, sys; m = json.load(open(sys.argv[1])); \
        print(m['result_volume_paths'][sys.argv[2]].removeprefix('/results/'))" "$@"
    }
    uv run modal volume get quail-results \
      "$RUNS/20260905T021527Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/manifest.json" \
      "$W/saved/manifest.json"
    uv run modal volume get quail-results \
      "$RUNS/20260906T211500Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/manifest.json" \
      "$W/saved/fev9_manifest.json"
    for method in quail pipelined_vllm pipelined_sglang; do
      uv run modal volume get quail-results \
        "$(result_path "$W/saved/manifest.json" $method)" "$W/saved/$method.json"
    done
    for method in quail pipelined_vllm; do
      uv run modal volume get quail-results \
        "$(result_path "$W/saved/fev9_manifest.json" $method)" \
        "$W/saved/fev9_$method.json"
    done
    uv run modal volume get quail-results \
      ablations/streamed-filter-join-regret.json "$W/ablation/regret.json"
    CORPUS=ground_truth/quailb/schema_v1/corpora/c_1aa2c4f0d0b6c816fd37aa5748c33341
    uv run modal volume get quail-results "$CORPUS/manifest.json" "$W/corpus.json"
    uv run modal volume get quail-results \
      sol/2026-09-11-quailb-prefix-reuse.json "$W/sol.json"
    uv run --with matplotlib python reports/make_quailb_comparison_plots.py "$W"

Every method and query comes from the September 12 run of all four
methods, with three exceptions that run's FEVER container did not
produce (it failed on FEV-10 in the request backends, fixed since, and
was not rerun): pipelined vLLM FEV-1 to FEV-9 come from the saved
September 5 suite (FEV-9 from its September 6 rerun), and the three
baselines' FEV-10 from the September 11 FEV-10 run. The saved Quail
suite and the ablation file feed the before-and-after section.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from plot_colors import BLUE, DARK, GRAY, GREEN, ORANGE

from quail.specs import H100_USD_PER_HOUR

HERE = Path(__file__).resolve().parent
METHODS = [
    ("quail", "Quail", BLUE),
    ("stock_vllm", "Stock vLLM", GRAY),
    ("pipelined_vllm", "Pipelined vLLM", ORANGE),
    ("pipelined_sglang", "Pipelined SGLang", GREEN),
]
QUERY_ORDER = (
    [f"IMDB-{n}" for n in range(1, 11)] + [f"BIO-{n}" for n in range(1, 4)]
    + [f"FEV-{n}" for n in range(1, 11)] + [f"LEP-{n}" for n in range(1, 9)]
    + ["AGENT-1", "AGENT-2"])


def token_cell(value):
    """Format a token count for a report table cell."""
    return "Not measured" if value is None else f"{value:,}"


def row_metrics(row):
    """Derive throughput, GPU cost, and accuracy from one measurement row."""
    pairs = row["evaluated_document_pairs"]
    count = row["input_rows"] if pairs is None else pairs
    matched = row["matching_rows"]
    predicted = row["predicted_rows"]
    expected = row["expected_rows"]
    return {
        "seconds": row["runtime_s"],
        "recomputed": row["regret_tokens"],
        "fresh": row["fresh_tokens"],
        "throughput": count / row["runtime_s"],
        "unit": "docs/s" if pairs is None else "pairs/s",
        "cost": row["runtime_s"] / 3600 * H100_USD_PER_HOUR,
        "agreement": 100 * row["answers_correct"] / row["answers_evaluated"],
        "precision": (100 * matched / predicted if predicted
                      else (0 if expected else 100)),
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
    from quail_b.queries import queries as query_specs

    specs = query_specs()
    return {
        query: [
            (alias.alias, alias.table, corpus["tables"][alias.table]["rows"])
            for alias in specs[query].aliases]
        for query in queries
    }


def load_sol(root, queries, rows, corpus):
    """Load compatible estimates with prefix reuse across requests."""
    from quail_b.queries import queries as query_specs

    specs = query_specs()
    source = json.loads((root / "sol.json").read_text())
    assert source["corpus_id"] == corpus["corpus_id"]
    assert source["scale_factor"] == 0.1
    assert source["optimizer"]["persistent_kv_capacity"] == "unlimited"
    assert "across documents" in source["estimates"]["sol_s"]
    estimates = {}
    for query in queries:
        estimate = source["queries"][query]["models"]["qwen3-4b-fp8"]
        spec = specs[query]
        filters = sorted(alias.alias for alias in spec.aliases
                         for _ in alias.filters)
        assert sorted(s["alias"] for s in estimate["filter_stages"]) == filters
        assert len(estimate["join_stages"]) == len(spec.joins), query
        assert estimate["input_document_rows"] == rows["quail"][query]["input_rows"]
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
    methods = METHODS + ([] if metric == "agreement"
                         else [("sol", "SoL estimate", DARK)])
    positive = [value for key, _, _ in methods for query in queries
                if (value := series_value(rows, sol, key, query, metric)) is not None
                and value > 0]
    maximum = max(positive, default=0)
    logarithmic = (metric != "agreement" and positive
                   and maximum / min(positive) > 10)
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
            axis.set_ylabel("seconds (log scale)" if metric == "seconds"
                            else "tokens (log scale)")
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
                              textcoords="offset points", rotation=90,
                              ha="center", va="bottom", fontsize=8)
        axis.bar(xs, [value - floor for value in values], bottom=floor,
                 width=width * 0.9, color=color, label=label, edgecolor="none")
    if metric != "agreement":
        for index, query in enumerate(queries):
            axis.hlines(series_value(rows, sol, "sol", query, metric),
                        index - 0.41, index + 0.41, color=DARK, linewidth=1.7,
                        zorder=3, clip_on=False)
    axis.set_xlim(-0.7, len(queries) - 0.3)
    axis.set_xticks(range(len(queries)),
                   queries,
                   rotation=55 if overview else 35, ha="right")
    axis.tick_params(axis="x", labelsize=9)
    titles = {"seconds": "Latency", "recomputed": "Recomputed KV tokens",
              "fresh": "Fresh input tokens",
              "agreement": "Answer agreement with Qwen3 32B"}
    axis.set_title(titles[metric], fontsize=13)


def document_page(title, queries, relations):
    """Create a readable input-count page for the PDF."""
    figure = plt.figure(figsize=(14, 9))
    figure.suptitle(f"{title}: input documents before filtering", y=0.96,
                    fontsize=16)
    axis = figure.add_axes((0.045, 0.12, 0.91, 0.77))
    axis.axis("off")
    cells = [[query, "; ".join(f"{alias} ({provider}): {count:,}"
                               for alias, provider, count in relations[query])]
             for query in queries]
    table = axis.table(cellText=cells,
                       colLabels=["Query", "Relation alias (set): input documents"],
                       colWidths=[0.10, 0.90], cellLoc="left", colLoc="left",
                       loc="upper left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, _), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_height(min(0.065, 0.95 / (len(cells) + 1)))
        cell.PAD = 0.02
        if row == 0:
            cell.set_text_props(weight="bold")
    note_text = (
        "Each alias lists its full input before filters. Repeated aliases can "
        "refer to the same underlying set.\n"
        "Counts come from the saved corpus manifest. SoL uses the same inputs "
        "and the saved reference labels for survivors.")
    figure.text(0.055, 0.075, note_text, fontsize=11, linespacing=1.5)
    return figure


def plot_comparison(title, queries, rows, relations, sol, name, overview=False):
    """Export vector PDF pages and a first-page PNG preview."""
    destination = HERE / "plots" / name
    groups = [[metric] for metric, _, _ in METRICS] if overview else [
        ["seconds", "fresh", "recomputed", "agreement"]]
    with PdfPages(destination.with_suffix(".pdf")) as pdf:
        for page, metrics in enumerate(groups):
            figure, axes = (plt.subplots(1, 1, figsize=(14, 8.5)) if overview
                            else plt.subplots(2, 2, figsize=(14, 10)))
            axes = [axes] if overview else list(axes.flat)
            for axis, metric in zip(axes, metrics):
                metric_bars(axis, queries, rows, sol, metric, overview)
            figure.suptitle(f"{title}, Qwen3 4B FP8, sf=0.1, one H100", y=0.97,
                            fontsize=16)
            handles = [Patch(facecolor=color, label=label)
                       for _, label, color in METHODS]
            if metrics != ["agreement"]:
                handles.append(Line2D([0], [0], color=DARK, linewidth=1.7,
                                      label="SoL estimate"))
            figure.legend(handles=handles, loc="upper center",
                          bbox_to_anchor=(0.5, 0.925), ncol=5, fontsize=11,
                          frameon=False)
            footer_text = (
                "SoL estimates ideal work with unlimited prefix KV reuse across "
                "requests and reference-label survivors. "
                "It has no measured accuracy.\n"
                "Fresh tokens include recomputed KV. A dash marks zero, "
                "a cross a value that was not measured. "
                "Stock vLLM uses operator-at-a-time submission.")
            figure.text(0.055, 0.035, footer_text, fontsize=10, linespacing=1.5)
            figure.subplots_adjust(left=0.075, right=0.97, top=0.83,
                                   bottom=0.24 if overview else 0.20, hspace=0.60,
                                   wspace=0.28)
            pdf.savefig(figure, bbox_inches=None)
            if page == 0:
                figure.savefig(destination, dpi=300, bbox_inches=None)
            plt.close(figure)
        figure = document_page(title, queries, relations)
        pdf.savefig(figure, bbox_inches=None)
        plt.close(figure)
    return destination.name


def measurement_rows(path):
    """Read a run's measurements.parquet as {method: {query: row}}."""
    rows = {}
    for row in pq.read_table(path).to_pylist():
        rows.setdefault(row["method"], {})[row["query"]] = row
    return rows


def suite_row(row):
    """Shape a saved suite row (the September 5 format) like a measurement row."""
    joins = [stage for stage in row["stages"] if stage["op"] == "join"]
    answers = row["accuracy"]["answer_accuracy"]
    output = row["accuracy"]["output_accuracy"]
    return {
        "query": row["query"],
        "runtime_s": row["wall_s"],
        "fresh_tokens": row["fresh_tokens"],
        "minimum_tokens": None,
        "regret_tokens": None,
        "evaluated_document_pairs": (
            sum(stage["tuples"] for stage in joins) if joins else None),
        "input_rows": row["input_document_rows"],
        "answers_evaluated": answers["evaluated"],
        "answers_correct": answers["correct"],
        "predicted_rows": output["predicted_rows"],
        "expected_rows": output["expected_rows"],
        "matching_rows": output["matching_rows"],
        "cost_usd": row["wall_s"] / 3600 * H100_USD_PER_HOUR,
    }


def suite_rows(root, method, corpus):
    """The saved suite's rows of one method, FEV-9 from its rerun if pulled."""
    sources = [json.loads((root / "saved" / f"{method}.json").read_text())]
    rerun = root / "saved" / f"fev9_{method}.json"
    if rerun.exists():
        sources.append(json.loads(rerun.read_text()))
    rows = {}
    for source in sources:
        assert (source["model"], source["sf"], source["lf"], source["gpus"]) == (
            "qwen3-4b-fp8", 0.1, 1, 1)
        assert source["corpus_id"] == corpus["corpus_id"]
        assert source["backend"] == method
        for row in source["passes"]["single"]["queries"]:
            assert "error" not in row, row
            rows[row["query"]] = suite_row(row)
    return rows


def check_manifest(manifest):
    """Assert a family run's manifest is the expected configuration."""
    from quail_b.queries import SELECTIVITY_ESTIMATE_COLLECTION

    assert manifest["model"] == "qwen3-4b-fp8"
    assert manifest["sf"] == 0.1
    assert manifest["collection_id"] == SELECTIVITY_ESTIMATE_COLLECTION


def load_rows(root, corpus):
    """Return {method: {query: row}} and the source of every cell.

    The September 12 run supplies every cell it has. The FEV-10 run
    fills the three baselines' FEV-10, and the saved suite fills any
    other cell that run did not produce, with no recomputed KV figure
    since the suite saved no answer tables.
    """
    manifest = json.loads((root / "run" / "manifest.json").read_text())
    fev10_manifest = json.loads((root / "fev10" / "manifest.json").read_text())
    check_manifest(manifest)
    check_manifest(fev10_manifest)
    assert fev10_manifest["query_ids"] == ["FEV-10"]
    rows = measurement_rows(root / "run" / "measurements.parquet")
    sources = {(method, query): "run" for method in rows for query in rows[method]}
    fev10 = measurement_rows(root / "fev10" / "measurements.parquet")
    saved = {}
    for key, _, _ in METHODS:
        for query in QUERY_ORDER:
            if query in rows.setdefault(key, {}):
                continue
            if query == "FEV-10" and query in fev10.get(key, {}):
                rows[key][query] = fev10[key][query]
                sources[(key, query)] = "fev10"
            elif (root / "saved" / f"{key}.json").exists():
                if key not in saved:
                    saved[key] = suite_rows(root, key, corpus)
                rows[key][query] = saved[key][query]
                sources[(key, query)] = "saved"
    return rows, sources, manifest, fev10_manifest


def before_after_lines(root, rows, corpus):
    """The section comparing this run's Quail rows with the saved Quail rows."""
    saved = suite_rows(root, "quail", corpus)
    ablation = {(item["query"], item["config"]): item for item in json.loads(
        (root / "ablation" / "regret.json").read_text())}
    lines = [
        "## Quail before and after this branch", "",
        "Time before is the saved Quail suite (September 5, FEV-9 from September",
        "6); time after is the September 12 run. Recomputed KV before comes from",
        "the ablation cells that ran the `main` engine on the six queries whose",
        "join anchors it recomputed (`/results/ablations/streamed-filter-join-*`),",
        "computed by quail-bench from their saved answer tables; the other",
        "queries make the same requests with the same fresh tokens before and",
        "after, so their recomputed KV is the same number. FEV-10 did not exist",
        "before the branch.", "",
        "| Query | Time before, s | Time after, s | Change | Recomputed KV before "
        "| Recomputed KV after | Fresh before | Fresh after | Rows before / after |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for query in QUERY_ORDER:
        after = rows["quail"][query]
        if query not in saved:
            lines.append(
                f"| {query} | new | {after['runtime_s']:.2f} | | "
                f"| {token_cell(after['regret_tokens'])} | | {after['fresh_tokens']:,} "
                f"| / {after['predicted_rows']:,} |")
            continue
        before = saved[query]
        change = 100 * (after["runtime_s"] - before["runtime_s"]) / before["runtime_s"]
        if (query, "materialized") in ablation:
            earlier = ablation[(query, "materialized")]
            assert earlier["fresh"] == before["fresh_tokens"], query
            regret_before = earlier["regret"]
        else:
            assert before["fresh_tokens"] == after["fresh_tokens"], query
            regret_before = after["regret_tokens"]
        lines.append(
            f"| {query} | {before['runtime_s']:.2f} | {after['runtime_s']:.2f} "
            f"| {change:+.1f}% | {token_cell(regret_before)} "
            f"| {token_cell(after['regret_tokens'])} "
            f"| {before['fresh_tokens']:,} | {after['fresh_tokens']:,} "
            f"| {before['predicted_rows']:,} / {after['predicted_rows']:,} |")
    return lines + [""]


def main(workdir):
    """Regenerate the figures and the report from the pulled files."""
    root = Path(workdir)
    corpus = json.loads((root / "corpus.json").read_text())
    rows, sources, manifest, fev10_manifest = load_rows(root, corpus)
    queries = list(QUERY_ORDER)
    assert all(query in rows[key] for key, _, _ in METHODS for query in queries)
    faster = sum(rows["quail"][query]["runtime_s"]
                 < rows["stock_vllm"][query]["runtime_s"] for query in queries)
    plt.style.use(HERE / "quail.mplstyle")
    plt.rcParams.update({"pdf.fonttype": 42, "figure.autolayout": False,
                         "savefig.bbox": None})
    relations = input_relations(queries, corpus)
    for method in rows.values():
        for query, row in method.items():
            assert (sum(count for _, _, count in relations[query])
                    == row["input_rows"])
    sol = load_sol(root, queries, rows, corpus)
    overview = plot_comparison("QUAIL-B", queries, rows, relations, sol,
                               "quailb_main.png", overview=True)
    fev = row_metrics(rows["quail"]["FEV-9"])
    output = rows["quail"]["FEV-9"]
    calls = manifest["function_call_ids"]
    borrowed = sorted(key for key, source in sources.items() if source != "run")
    lines = [
        "# QUAIL-B comparison", "",
        "- The main PDF covers all 33 queries with grouped bars and one metric "
        "per page.",
        "  Its final page lists input document counts. Each dataset PDF has a page",
        "  of four bar charts and a separate input-count page. Text and marks remain",
        "  vector content when zoomed. The PNGs below are first-page previews.",
        "- The setup was Qwen3 4B FP8, sf=0.1, lf=1, and one H100 per configuration.",
        "  Quail and the vLLM configurations shared a physical GPU within each",
        "  family. SGLang used a separate GPU. Stock vLLM used operator-at-a-time",
        "  submission; the other three pipeline their requests.",
        "- All four methods were run on September 12, 2026: "
        f"`/results/benchmarks/quailb/family-runs/{manifest['run_id']}/`,",
        "  function calls " + ", ".join(f"`{call}`" for call in calls.values()) + ".",
        f"  {len(borrowed)} of the 132 cells come from earlier runs because that",
        "  run's FEVER container failed on FEV-10 in the request backends (an",
        "  equality join's key columns were not passed to them; fixed on this",
        "  branch) and was not rerun: pipelined vLLM FEV-1 to FEV-9 from the",
        "  saved September 5 suite (FEV-9 from its September 6 rerun), and the",
        "  three baselines' FEV-10 from the September 11 FEV-10 run",
        f"  (`/results/benchmarks/quailb/family-runs/{fev10_manifest['run_id']}/`).",
        "  The saved suite kept no answer tables, so its recomputed KV is not",
        "  measured; the FEV-10 run's is.",
        f"- Quail was faster than stock vLLM on {faster} of {len(queries)} queries.",
        "- A horizontal line across each query's bar group shows its SoL estimate.",
        "  SoL models ideal computation and memory traffic with unlimited prefix KV.",
        "  It credits matching token prefixes across requests, documents, and aliases.",
        "  It uses exact reference-label survivors and searches supported left-deep",
        "  join plans. Different answers can change the work done by measured runs,",
        "  so the gap from SoL is not purely execution overhead. SoL uses the",
        "  distinct-prefix estimate, not the per-document-only estimate. No",
        "  accuracy is assigned to SoL because it is not a measured model run.",
        "  SoL was recalculated for all 33 queries on the CPU on September 11,",
        "  2026, from saved labels and corpus rows.",
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
        "- Recomputed KV is `regret_tokens`: fresh tokens minus the fewest input",
        "  tokens the run's requests needed with unlimited KV, where every",
        "  distinct prefix across the requests is computed once. quail-bench",
        "  derives it on the CPU after the run from the saved answer tables and",
        "  the prompt token pieces the runner reports (`quail_b.minimum`); the",
        "  engine tracks nothing. A run saved without that minimum is not measured.",
        "  Token and latency plots use a log scale when positive values span more",
        "  than one order of magnitude. Recomputed KV retains a linear region to",
        "  include zero. A dash marks zero.",
        "- Document counts come from the saved corpus manifest and describe inputs",
        "  before filtering. Repeated aliases each list their full input count.",
        "  The report tables also show throughput, GPU cost, and final output quality.",
        f"- FEV-9 agrees with the reference on {fev['agreement']:.2f}% of "
        "evaluated answers. Its final",
        f"  output matches only {output['matching_rows']:,} reference rows out of "
        f"{output['predicted_rows']:,} returned rows.",
        f"  The reference has {output['expected_rows']:,} rows, so output "
        f"precision is approximately {fev['precision']:.8f}%",
        f"  and recall is {fev['recall']:.2f}%.", "",
        "[Open the main vector PDF](plots/quailb_main.pdf)", "",
        f"[![QUAIL-B latency preview](plots/{overview})](plots/quailb_main.pdf)", "",
        f"Figure: plots/{overview}", "",
        "SoL estimates on `quail-results`: "
        "`/results/sol/2026-09-11-quailb-prefix-reuse.json`.", "",
        "Corpus counts on `quail-results`: `/results/ground_truth/quailb/"
        f"schema_v1/corpora/{corpus['corpus_id']}/manifest.json`.", "",
        "The download commands are in `reports/make_quailb_comparison_plots.py`.",
        "",
    ]
    for family in ("IMDB", "BIO", "FEV", "LEP", "AGENT"):
        selected = [query for query in queries if query.startswith(family + "-")]
        name = plot_comparison(f"QUAIL-B {family}", selected, rows, relations,
                               sol, f"quailb_{family.lower()}.png")
        pdf_name = Path(name).with_suffix(".pdf").name
        lines.extend([f"## {family}", "",
                      f"[Open the {family} vector PDF](plots/{pdf_name})", "",
                      f"[![{family} preview](plots/{name})](plots/{pdf_name})", "",
                      f"Figure: plots/{name}", "",
                      "| Query | Input documents by alias and set |", "|---|---|"])
        for query in selected:
            counts = ", ".join(f"{alias} ({provider}) = {count:,}"
                               for alias, provider, count in relations[query])
            lines.append(f"| {query} | {counts} |")
        lines.extend(["",
                      "| Query | Method | Seconds | Recomputed KV tokens "
                      "| Fresh input tokens | Throughput | Unit | $/query "
                      "| Answer agreement (%) | Output precision (%) "
                      "| Output recall (%) | Source |",
                      "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|"])
        for query in selected:
            for key, label, _ in METHODS:
                m = row_metrics(rows[key][query])
                source = {"run": "September 12", "fev10": "FEV-10 run",
                          "saved": "saved suite"}[sources[(key, query)]]
                lines.append(
                    f"| {query} | {label} | {m['seconds']:.2f} "
                    f"| {token_cell(m['recomputed'])} "
                    f"| {m['fresh']:,} | {m['throughput']:,.2f} "
                    f"| {m['unit']} | {m['cost']:.5f} | {m['agreement']:.2f} "
                    f"| {m['precision']:.5g} | {m['recall']:.5g} | {source} |")
            estimate = sol[query]
            throughput = estimate["document_pairs_per_second_at_sol"]
            unit = "pairs/s"
            if throughput is None:
                throughput = estimate["documents_per_second_at_sol"]
                unit = "docs/s"
            lines.append(
                f"| {query} | SoL estimate | {estimate['sol_s']:.3f} | 0 (assumed) "
                f"| {estimate['tokens']:,.0f} | {throughput:,.2f} | {unit} "
                f"| {estimate['cost_usd_per_query_at_sol']:.5f} "
                "| Not measured | Not measured | Not measured | estimate |")
        lines.append("")
    lines.extend(before_after_lines(root, rows, corpus))
    report = HERE / "quailb-comparison.md"
    report.write_text("\n".join(lines))
    print(f"Updated {report}, the main figure, and five dataset figures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir")
    main(parser.parse_args().workdir)
