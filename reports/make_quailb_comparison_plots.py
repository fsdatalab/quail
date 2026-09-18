"""Generate the QUAIL-B overview and dataset plots without inference.

Pull the measured runs, the corpus manifest, and the SoL estimates:

    W=/tmp/quail-comparison; mkdir -p "$W/run" "$W/fev10" "$W/bio"
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
    CORPUS=ground_truth/quailb/schema_v1/corpora/c_1aa2c4f0d0b6c816fd37aa5748c33341
    uv run modal volume get quail-results "$CORPUS/manifest.json" "$W/corpus.json"
    uv run modal volume get quail-results \
      sol/2026-09-11-quailb-prefix-reuse.json "$W/sol.json"
    BIO=$RUNS/20260918T060700Z-biodex-chat
    uv run modal volume get quail-results "$BIO/manifest.json" "$W/bio/manifest.json"
    uv run modal volume get quail-results "$BIO/measurements.parquet" \
      "$W/bio/measurements.parquet"
    for METHOD in quail pipelined_vllm; do
      mkdir -p "$W/bio/$METHOD"
      uv run modal volume get quail-results "$BIO/$METHOD/run.json" \
        "$W/bio/$METHOD/run.json"
    done
    uv run modal volume get quail-results \
      sol/2026-09-18-biodex-chat/sol_quailb_sf0.1_BIO-1_BIO-2_BIO-3.json \
      "$W/bio-sol.json"
    BENCH=git+https://github.com/fsdatalab/quail-bench.git
    REV=fc27f35188f0fcc6a1f3b8fe3bfbb9e12d6842eb
    uv run --with matplotlib --with "quail-b@$BENCH@$REV" \
      python reports/make_quailb_comparison_plots.py "$W"

BioDEX uses the September 18 chat-format run. Other queries come from
September 12, except the two vLLM configurations' FEV-10, which come
from the September 11 FEV-10 run. The September 12 FEVER container failed in the
request backends (fixed since) and was not rerun, so pipelined vLLM
has no FEV-1 to FEV-9 measurement: those cells are marked missing,
not filled from older runs. SGLang is not reported.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from plot_colors import BLUE, DARK, GRAY, ORANGE

from quail.specs import H100_USD_PER_HOUR

HERE = Path(__file__).resolve().parent
METHODS = [
    ("quail", "Quail", BLUE),
    ("stock_vllm", "Stock vLLM", GRAY),
    ("pipelined_vllm", "Pipelined vLLM", ORANGE),
]
QUERY_ORDER = (
    [f"IMDB-{n}" for n in range(1, 11)] + [f"BIO-{n}" for n in range(1, 4)]
    + [f"FEV-{n}" for n in range(1, 11)] + [f"LEP-{n}" for n in range(1, 9)]
    + ["AGENT-1", "AGENT-2"])
BIO_QUERIES = ("BIO-1", "BIO-2", "BIO-3")


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
    from quail.bench.substrait import read_plan
    from quail_b.queries import get_query

    return {
        query: [
            (relation.alias, relation.table,
             corpus["tables"][relation.table]["rows"])
            for relation in read_plan(get_query(query).plan).relations]
        for query in queries
    }


def load_sol(root, queries, rows, corpus):
    """Load compatible estimates with prefix reuse across requests."""
    from quail.bench.substrait import read_plan
    from quail_b.queries import get_query
    from quail_b.rendering import PROMPT_FORMAT

    source = json.loads((root / "sol.json").read_text())
    bio_source = json.loads((root / "bio-sol.json").read_text())
    bio_manifest = json.loads((root / "bio" / "manifest.json").read_text())
    assert bio_source["corpus_id"] == corpus["corpus_id"]
    assert bio_source["collection_id"] == bio_manifest["collection_id"]
    assert bio_source["scale_factor"] == 0.1
    assert bio_source["optimizer"]["persistent_kv_capacity"] == "unlimited"
    assert set(bio_source["queries"]) == set(BIO_QUERIES)
    for query, record in bio_source["queries"].items():
        assert record["prompt_format"] == PROMPT_FORMAT
        assert record["plan_sha256"] == hashlib.sha256(
            get_query(query).plan_bytes).hexdigest()
    source["queries"].update(bio_source["queries"])
    assert source["corpus_id"] == corpus["corpus_id"]
    assert source["scale_factor"] == 0.1
    assert source["optimizer"]["persistent_kv_capacity"] == "unlimited"
    assert "across documents" in source["estimates"]["sol_s"]
    estimates = {}
    for query in queries:
        estimate = source["queries"][query]["models"]["qwen3-4b-fp8"]
        plan = read_plan(get_query(query).plan)
        filters = sorted(item.alias for item in plan.filters)
        assert sorted(s["alias"] for s in estimate["filter_stages"]) == filters
        assert len(estimate["join_stages"]) == len(plan.joins), query
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
    measured = [item for item in METHODS
                if item[0] != "stock_vllm" or set(queries) != set(BIO_QUERIES)]
    methods = measured + ([] if metric == "agreement"
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
    width = 0.82 / len(measured)
    floor = axis.get_ylim()[0]
    for method_index, (key, label, color) in enumerate(measured):
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
    if metric == "seconds" and not overview:
        for index, query in enumerate(queries):
            if query not in BIO_QUERIES:
                continue
            quail_time = rows["quail"][query]["runtime_s"]
            baseline_time = rows["pipelined_vllm"][query]["runtime_s"]
            top = max(quail_time, baseline_time)
            axis.text(index, top * (3 if logarithmic else 1.3),
                      f"{baseline_time / quail_time:.2f}x faster", ha="center",
                      va="bottom", fontsize=9)
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
    """Export the vector PDF: one metric per page, then the input counts."""
    destination = HERE / "plots" / name
    groups = [[metric] for metric, _, _ in METRICS] if overview else [
        ["seconds", "fresh", "recomputed", "agreement"]]
    with PdfPages(destination.with_suffix(".pdf")) as pdf:
        for metrics in groups:
            figure, axes = (plt.subplots(1, 1, figsize=(14, 8.5)) if overview
                            else plt.subplots(2, 2, figsize=(14, 10)))
            axes = [axes] if overview else list(axes.flat)
            for axis, metric in zip(axes, metrics):
                metric_bars(axis, queries, rows, sol, metric, overview)
            figure.suptitle(f"{title}, Qwen3 4B FP8, sf=0.1, one H100", y=0.97,
                            fontsize=16)
            handles = [Patch(facecolor=color, label=label)
                       for key, label, color in METHODS
                       if key != "stock_vllm" or set(queries) != set(BIO_QUERIES)]
            if metrics != ["agreement"]:
                handles.append(Line2D([0], [0], color=DARK, linewidth=1.7,
                                      label="SoL estimate"))
            figure.legend(handles=handles, loc="upper center",
                          bbox_to_anchor=(0.5, 0.925), ncol=5, fontsize=11,
                          frameon=False)
            figure.subplots_adjust(left=0.075, right=0.97, top=0.83,
                                   bottom=0.14 if overview else 0.10, hspace=0.60,
                                   wspace=0.28)
            pdf.savefig(figure, bbox_inches=None)
            plt.close(figure)
        figure = document_page(title, queries, relations)
        pdf.savefig(figure, bbox_inches=None)
        plt.close(figure)
    return destination.with_suffix(".pdf").name


def measurement_rows(path):
    """Read a run's measurements.parquet as {method: {query: row}}."""
    rows = {}
    for row in pq.read_table(path).to_pylist():
        rows.setdefault(row["method"], {})[row["query"]] = row
    return rows


def check_manifest(manifest):
    """Assert a family run's manifest is the expected configuration."""
    from quail_b.queries import SELECTIVITY_ESTIMATE_COLLECTION

    assert manifest["model"] == "qwen3-4b-fp8"
    assert manifest["sf"] == 0.1
    assert manifest["collection_id"] == SELECTIVITY_ESTIMATE_COLLECTION


def load_rows(root, corpus):
    """Return {method: {query: row}} and the source of every cell.

    BioDEX uses the chat-format run. Other cells come from September 12,
    with the baselines' FEV-10 filled from its separate run.
    """
    manifest = json.loads((root / "run" / "manifest.json").read_text())
    fev10_manifest = json.loads((root / "fev10" / "manifest.json").read_text())
    check_manifest(manifest)
    check_manifest(fev10_manifest)
    assert fev10_manifest["query_ids"] == ["FEV-10"]
    rows = measurement_rows(root / "run" / "measurements.parquet")
    sources = {(method, query): "run" for method in rows for query in rows[method]}
    fev10 = measurement_rows(root / "fev10" / "measurements.parquet")
    for key, _, _ in METHODS:
        if "FEV-10" not in rows.setdefault(key, {}) and "FEV-10" in fev10.get(key, {}):
            rows[key]["FEV-10"] = fev10[key]["FEV-10"]
            sources[(key, "FEV-10")] = "fev10"
    from quail_b.queries import get_query
    from quail_b.rendering import PROMPT_FORMAT
    from quail_b.run import _query_hash

    bio_manifest = json.loads((root / "bio" / "manifest.json").read_text())
    assert bio_manifest["status"] == "complete"
    assert bio_manifest["model"] == "qwen3-4b-fp8"
    assert bio_manifest["sf"] == 0.1
    assert set(bio_manifest["query_ids"]) == set(BIO_QUERIES)
    assert set(bio_manifest["methods"]) == {"quail", "pipelined_vllm"}
    bio_rows = measurement_rows(root / "bio" / "measurements.parquet")
    for method, _, _ in METHODS:
        for query in BIO_QUERIES:
            rows[method].pop(query, None)
            sources.pop((method, query), None)
    for method in bio_manifest["methods"]:
        suite = json.loads((root / "bio" / method / "run.json").read_text())
        assert suite["corpus_id"] == corpus["corpus_id"]
        assert suite["collection_id"] == bio_manifest["collection_id"]
        assert suite["metadata"]["prompt_format"] == PROMPT_FORMAT
        assert set(bio_rows[method]) == set(BIO_QUERIES)
        for item in suite["queries"]:
            assert item["status"] == "complete"
            assert item["definition_hash"] == _query_hash(get_query(item["id"]))
        rows[method].update(bio_rows[method])
        sources.update({(method, query): "bio" for query in BIO_QUERIES})
    return rows, sources, manifest, fev10_manifest, bio_manifest


def main(workdir):
    """Regenerate the figures and the report from the pulled files."""
    root = Path(workdir)
    corpus = json.loads((root / "corpus.json").read_text())
    rows, sources, manifest, fev10_manifest, bio_manifest = load_rows(root, corpus)
    queries = list(QUERY_ORDER)
    assert all(query in rows["quail"] for query in queries)
    compared = [query for query in queries if query in rows["stock_vllm"]]
    faster = sum(rows["quail"][query]["runtime_s"]
                 < rows["stock_vllm"][query]["runtime_s"] for query in compared)
    plt.style.use(HERE / "quail.mplstyle")
    plt.rcParams.update({"pdf.fonttype": 42, "figure.autolayout": False,
                         "savefig.bbox": None})
    relations = input_relations(queries, corpus)
    for method in rows.values():
        for query, row in method.items():
            assert (sum(count for _, _, count in relations[query])
                    == row["input_rows"])
    sol = load_sol(root, queries, rows, corpus)
    bio_baseline = json.loads(
        (root / "bio" / "pipelined_vllm" / "run.json").read_text())
    bio_capacity = bio_baseline["queries"][0]["measurements"][
        "backend_metrics"]["capacity"]
    overview = plot_comparison("QUAIL-B", queries, rows, relations, sol,
                               "quailb_main.pdf", overview=True)
    fev = row_metrics(rows["quail"]["FEV-9"])
    bio2 = row_metrics(rows["quail"]["BIO-2"])
    bio3 = row_metrics(rows["quail"]["BIO-3"])
    output = rows["quail"]["FEV-9"]
    calls = {key: call for key, call in manifest["function_call_ids"].items()
             if not key.endswith(":sglang") and not key.startswith("bio:")}
    labels = {key: label for key, label, _ in METHODS}

    def cells_text(cells):
        by_method = {}
        for method, query in sorted(cells):
            by_method.setdefault(method, []).append(query)
        return "; ".join(
            f"{labels[method]} " + ", ".join(
                query for query in QUERY_ORDER if query in queries_of)
            for method, queries_of in by_method.items())

    borrowed = [key for key, source in sources.items() if source == "fev10"]
    missing = [(key, query) for key, _, _ in METHODS for query in queries
               if query not in rows[key]]

    lines = [
        "# QUAIL-B comparison", "",
        "- The main PDF covers all 33 queries with grouped bars and one metric "
        "per page.",
        "  Its final page lists input document counts. Each dataset PDF has a page",
        "  of four bar charts and a separate input-count page. Text and marks remain",
        "  vector content when zoomed.",
        "- The setup was Qwen3 4B FP8, sf=0.1, lf=1, and one H100 per configuration.",
        "  Quail and the vLLM configurations shared a physical GPU within each",
        "  family. Stock vLLM used operator-at-a-time submission; the other two",
        "  pipeline their requests.",
        "- BioDEX uses non-thinking chat prompts and newly generated Qwen3 32B",
        "  reference labels. BIO-1 and BIO-3 filter for serious adverse events.",
        "  Quail and pipelined stock vLLM were rerun on September 18:",
        f"  `/results/benchmarks/quailb/family-runs/{bio_manifest['run_id']}/`.",
        "  The older BioDEX measurements are omitted. Other datasets retain",
        "  their earlier prompts and measurements.",
        "  Quail's planned limits were 110,376 tokens per chunk and 362,250",
        "  resident KV tokens. Pipelined stock vLLM used prefix caching,",
        f"  {bio_capacity['max_num_batched_tokens']:,} batched tokens,",
        f"  {bio_capacity['max_num_seqs']:,} sequences, and GPU memory utilization",
        f"  {bio_capacity['gpu_memory_utilization']:.2f}. Its measured KV capacity",
        f"  was {bio_capacity['kv_cache_size_tokens']:,} tokens. Both methods used",
        "  the same planner's filter and join ordering rules.",
        "- Other datasets use the September 12, 2026 run: "
        f"`/results/benchmarks/quailb/family-runs/{manifest['run_id']}/`,",
        "  function calls " + ", ".join(f"`{call}`" for call in calls.values()) + ".",
        "  That run's FEVER container failed on FEV-10 in the request backends (an",
        "  equality join's key columns were not passed to them; fixed since) and",
        f"  was not rerun. {len(borrowed)} of the 99 cells come from the September 11",
        f"  FEV-10 run (`/results/benchmarks/quailb/family-runs/"
        f"{fev10_manifest['run_id']}/`): {cells_text(borrowed)}.",
        f"  {len(missing)} cells have no current measurement. The main plot marks",
        "  them with an x below the axis:",
        f"  {cells_text(missing)}.",
        f"- Quail was faster than stock vLLM on {faster} of {len(compared)} queries",
        "  where both were measured.",
        "- A horizontal line across each query's bar group shows its SoL estimate.",
        "  SoL models ideal computation and memory traffic with unlimited prefix KV.",
        "  It credits matching token prefixes across requests, documents, and aliases.",
        "  It uses exact reference-label survivors and searches supported left-deep",
        "  join plans. Different answers can change the work done by measured runs,",
        "  so the gap from SoL is not purely execution overhead. SoL uses the",
        "  distinct-prefix estimate, not the per-document-only estimate. No",
        "  accuracy is assigned to SoL because it is not a measured model run.",
        "  SoL was calculated from saved labels and corpus rows on the CPU:",
        "  BioDEX on September 18, and the other datasets on September 11.",
        "- Answer agreement measures evaluated calls against saved Qwen3 32B labels.",
        "  Each method can evaluate different calls after its filters and joins.",
        "  Output precision is the fraction of returned rows matching the reference.",
        "  Output recall is the fraction of reference rows returned. High answer",
        "  agreement can coexist with poor final output precision.",
        "- The prediction was 20 to 35 seconds for BIO-1 and faster joins in Quail",
        "  than in pipelined stock vLLM. The measured times support both predictions.",
        f"- Quail's output recall is {bio2['recall']:.2f}% on BIO-2 and",
        f"  {bio3['recall']:.2f}% on BIO-3 against the saved 32B references.",
        f"  Per-answer agreement is {bio2['agreement']:.2f}% and",
        f"  {bio3['agreement']:.2f}%, respectively. Most evaluated pairs are negative.",
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
        f"[Open the main vector PDF](plots/{overview})", "",
        f"Figure: plots/{overview}", "",
        "SoL estimates on `quail-results`: "
        "`/results/sol/2026-09-11-quailb-prefix-reuse.json`.", "",
        "BioDEX SoL: `/results/sol/2026-09-18-biodex-chat/"
        "sol_quailb_sf0.1_BIO-1_BIO-2_BIO-3.json`.", "",
        "Corpus counts on `quail-results`: `/results/ground_truth/quailb/"
        f"schema_v1/corpora/{corpus['corpus_id']}/manifest.json`.", "",
        "The download commands are in `reports/make_quailb_comparison_plots.py`.",
        "",
    ]
    for family in ("IMDB", "BIO", "FEV", "LEP", "AGENT"):
        selected = [query for query in queries if query.startswith(family + "-")]
        name = plot_comparison(f"QUAIL-B {family}", selected, rows, relations,
                               sol, f"quailb_{family.lower()}.pdf")
        lines.extend([f"## {family}", "",
                      f"[Open the {family} vector PDF](plots/{name})", "",
                      f"Figure: plots/{name}", "",
                      "| Query | Input documents by alias and set |", "|---|---|"])
        for query in selected:
            counts = ", ".join(f"{alias} ({provider}) = {count:,}"
                               for alias, provider, count in relations[query])
            lines.append(f"| {query} | {counts} |")
        if family == "BIO":
            terms = corpus["tables"]["terms"]["rows"]
            survivors = {}
            for method in ("quail", "pipelined_vllm"):
                pairs = rows[method]["BIO-3"]["evaluated_document_pairs"]
                assert pairs % terms == 0
                survivors[method] = pairs // terms
            ratios = [
                rows["pipelined_vllm"][query]["runtime_s"]
                / rows["quail"][query]["runtime_s"] for query in BIO_QUERIES]
            lines.extend([
                "", "Quail was " + ", ".join(f"{ratio:.2f}x" for ratio in ratios)
                + " faster than pipelined stock vLLM on BIO-1, BIO-2, and BIO-3,"
                " respectively.",
                "", "BIO-3 filter survivors: "
                f"{sol['BIO-3']['documents_after_filters']:,} in the reference, "
                f"{survivors['quail']:,} in Quail, and "
                f"{survivors['pipelined_vllm']:,} in pipelined stock vLLM."])
        lines.extend(["",
                      "| Query | Method | Seconds | Recomputed KV tokens "
                      "| Fresh input tokens | Throughput | Unit | $/query "
                      "| Answer agreement (%) | Output precision (%) "
                      "| Output recall (%) | Source |",
                      "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|"])
        for query in selected:
            for key, label, _ in METHODS:
                if family == "BIO" and key == "stock_vllm":
                    continue
                if query not in rows[key]:
                    lines.append(f"| {query} | {label} | missing | | | | | | | | "
                                 "| not run |")
                    continue
                m = row_metrics(rows[key][query])
                source = {"run": "September 12",
                          "bio": "September 18 BioDEX",
                          "fev10": "FEV-10 run"}[sources[(key, query)]]
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
    report = HERE / "quailb-comparison.md"
    report.write_text("\n".join(lines))
    print(f"Updated {report}, the main figure, and five dataset figures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir")
    main(parser.parse_args().workdir)
