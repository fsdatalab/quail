"""Generate the QUAIL-B overview and dataset plots without inference.

Pull the original suite manifest and its four result files:

    W=/tmp/quail-shared-comparison; mkdir -p "$W"
    result_path() {
      python3 -c "import json, sys; m = json.load(open(sys.argv[1])); \
        print(m['result_volume_paths'][sys.argv[2]].removeprefix('/results/'))" "$@"
    }
    RUNS=benchmarks/quailb/family-runs
    FAMILIES=benchmarks/quailb/families
    CORPUS=ground_truth/quailb/schema_v1/corpora/c_1aa2c4f0d0b6c816fd37aa5748c33341
    uv run modal volume get quail-results \
      "$RUNS/20260905T021527Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/manifest.json" \
      "$W/manifest.json"
    for method in quail stock_vllm pipelined_vllm pipelined_sglang; do
      source_path=$(result_path "$W/manifest.json" "$method")
      uv run modal volume get quail-results "$source_path" "$W/$method.json"
    done
    uv run modal volume get quail-results "$CORPUS/manifest.json" "$W/corpus.json"
    mkdir -p "$W/fev9"
    uv run modal volume get quail-results \
      "$RUNS/20260906T211500Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/manifest.json" \
      "$W/fev9/manifest.json"
    for method in quail stock_vllm pipelined_vllm pipelined_sglang; do
      source_path=$(result_path "$W/fev9/manifest.json" "$method")
      uv run modal volume get quail-results "$source_path" "$W/fev9/$method.json"
    done
    uv run modal volume get quail-results \
      "$FAMILIES/20260906T220559Z-sglang-baseline-redesign/fever-sglang-process.json" \
      "$W/fev9/sglang_anchor_major.json"
    uv run modal volume get quail-results \
      "$FAMILIES/20260906T222629Z-sglang-suffix-major/fever-sglang-process.json" \
      "$W/fev9/sglang_update.json"
    mkdir -p "$W/fev10"
    FEV10=$RUNS/20260911T201441Z-d16f87d8
    uv run modal volume get quail-results \
      "$FEV10/manifest.json" "$W/fev10/manifest.json"
    for method in quail stock_vllm pipelined_vllm pipelined_sglang; do
      uv run modal volume get quail-results \
        "$FEV10/$method/run.json" "$W/fev10/$method.json"
    done
    uv run modal volume get quail-results \
      /sol/2026-09-11-quailb-prefix-reuse.json "$W/sol.json"
    mkdir -p "$W/quail3" "$W/quail4"
    QUAIL3=$RUNS/20260912T023323Z-e689d27e
    uv run modal volume get quail-results \
      "$QUAIL3/manifest.json" "$W/quail3/manifest.json"
    for family in imdb biodex lepard agent; do
      uv run modal volume get quail-results \
        "$QUAIL3/quail/$family/run.json" "$W/quail3/$family.json"
    done
    QUAIL4=$RUNS/20260912T032335Z-609d6410
    uv run modal volume get quail-results \
      "$QUAIL4/manifest.json" "$W/quail4/manifest.json"
    uv run modal volume get quail-results \
      "$QUAIL4/quail/fever/run.json" "$W/quail4/fever.json"
    uv run --with matplotlib python reports/make_quailb_comparison_plots.py "$W" \
      --quail-dir "$W/quail3" --quail-dir "$W/quail4"

Use --fev9-dir and --fev10-dir to point to already pulled runs, and
--quail-dir (repeatable) to Quail-only reruns: each directory holds a
run's manifest.json and any of its report files (a run's quail/run.json
or a family's quail/<family>/run.json). All four current FEV-9
measurements replace the old query definition. FEV-10 joined the
benchmark after the suite run, so its four measurements come from its
own run. The Quail-only reruns replace every Quail row they cover, and
the report compares them with the saved Quail rows.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
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


def recomputed_kv(row):
    """Return a row's regret tokens, or None for a run saved without them.

    Regret is fresh tokens minus the fewest the run's requests needed
    with unlimited KV, derived from the saved answer tables by
    quail-bench's scoring. A run saved without that minimum has no
    figure.
    """
    return row.get("regret_tokens")


def token_cell(value):
    """Format a token count for a report table cell."""
    return "Not measured" if value is None else f"{value:,}"


def row_metrics(row):
    """Derive throughput, GPU cost, and accuracy from saved counts."""
    joins = [stage for stage in row["stages"] if stage["op"] == "join"]
    count = (sum(stage["tuples"] for stage in joins) if joins
             else row["input_document_rows"])
    answers = row["accuracy"]["answer_accuracy"]
    output = row["accuracy"]["output_accuracy"]
    matched = output["matching_rows"]
    predicted = output["predicted_rows"]
    expected = output["expected_rows"]
    return {
        "seconds": row["wall_s"],
        "recomputed": recomputed_kv(row),
        "fresh": row["fresh_tokens"],
        "throughput": count / row["wall_s"],
        "unit": "pairs/s" if joins else "docs/s",
        "cost": row["wall_s"] / 3600 * H100_USD_PER_HOUR,
        "agreement": 100 * answers["correct"] / answers["evaluated"],
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
    source = json.loads((root / "sol.json").read_text())
    assert source["corpus_id"] == corpus["corpus_id"]
    assert source["scale_factor"] == 0.1
    assert source["optimizer"]["persistent_kv_capacity"] == "unlimited"
    assert "across documents" in source["estimates"]["sol_s"]
    estimates = {}
    for query in queries:
        estimate = source["queries"][query]["models"]["qwen3-4b-fp8"]
        measured = rows["quail"][query]
        predicates = measured["accuracy"]["per_predicate"]
        expected_filters = [(p["alias"], p["predicate_key"])
                            for p in predicates if p["op"] == "filter"]
        expected_joins = [p["predicate_key"] for p in predicates
                          if p["op"] == "join"]
        filter_stages = sorted((s["alias"], s["code"])
                               for s in estimate["filter_stages"])
        assert filter_stages == sorted(expected_filters), query
        join_stages = sorted(s["code"] for s in estimate["join_stages"])
        assert join_stages == sorted(expected_joins), query
        assert (estimate["input_document_rows"]
                == measured["input_document_rows"]), query
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
            footer_text += (
                "\nFEV-9 and FEV-10 use the revised SGLang adapter; other SGLang "
                "results use "
                "the earlier adapter."
                if "FEV-9" in queries
                else "\nSGLang measurements use the earlier adapter.")
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


def current_rows(record):
    """Every query of a benchmark run.json, shaped like saved-suite rows."""
    return {query["id"]: current_row(query) for query in record["queries"]}


def current_row(query):
    """One query of a benchmark run.json, shaped like a saved-suite row."""
    measured = query["measurements"]
    return {
        "query": query["id"],
        "wall_s": measured["wall_s"],
        "fresh_tokens": measured["fresh_tokens"],
        **{key: query["metrics"][key] for key in ("minimum_tokens", "regret_tokens")
           if query["metrics"].get(key) is not None},
        "stages": measured["stages"],
        "backend_metrics": measured["backend_metrics"],
        "accuracy": query["metrics"]["accuracy"],
        "input_document_rows": query["metrics"]["accuracy"]["input_document_rows"],
    }


def check_run(record, key, corpus, suite, manifest):
    """Assert one benchmark run.json is the expected configuration."""
    configuration = record["metadata"]["configuration"]
    assert (configuration["model"], record["scale_factor"],
            configuration["gpus"]) == ("qwen3-4b-fp8", 0.1, 1)
    assert configuration["backend"] == key
    assert record["corpus_id"] == corpus["corpus_id"]
    assert (record["collection_id"]
            == suite["ground_truth"]["fever"]["collection_id"])
    # a family's own report carries no run id; the run's report does
    assert record.get("run_id") in (None, manifest["run_id"])
    for query in record["queries"]:
        assert query["status"] == "complete", query


def load_rows(root, fev9_root, fev10_root, quail_roots=()):
    """Combine the saved suite with the current FEV-9, FEV-10, and Quail runs."""
    manifest = json.loads((root / "manifest.json").read_text())
    corpus = json.loads((root / "corpus.json").read_text())
    fev9_manifest = json.loads((fev9_root / "manifest.json").read_text())
    fev10_manifest = json.loads((fev10_root / "manifest.json").read_text())
    assert fev10_manifest["query_ids"] == ["FEV-10"]
    sglang_update = json.loads((fev9_root / "sglang_update.json").read_text())
    assert sglang_update["methods"] == ["pipelined_sglang"]
    cleanup = sglang_update["process_cleanup"]
    assert all(value < 1024 for value in cleanup["gpu_memory_used_mib_after_exit"])
    assert fev9_manifest["query_ids"] == ["FEV-9"]
    rows = {}
    predicates = None
    for key, _, _ in METHODS:
        suite = json.loads((root / f"{key}.json").read_text())
        current = (sglang_update["suites"][key] if key == "pipelined_sglang" else
                   json.loads((fev9_root / f"{key}.json").read_text()))
        for source in (suite, current):
            assert (source["model"], source["sf"], source["lf"], source["gpus"]) == (
                "qwen3-4b-fp8", 0.1, 1, 1)
            assert source["corpus_id"] == corpus["corpus_id"]
            assert source["backend"] == key
        if key == "pipelined_sglang":
            assert current["ground_truth"] == suite["ground_truth"]["fever"]
        else:
            assert (current["aggregate_volume_path"]
                    == fev9_manifest["result_volume_paths"][key])
            assert current["ground_truth"]["fever"] == suite["ground_truth"]["fever"]
        measured = current["passes"]["single"]["queries"]
        assert len(measured) == 1 and measured[0]["query"] == "FEV-9"
        row = measured[0]
        assert "error" not in row, row
        if key == "pipelined_sglang":
            assert all(step["submission"] == "suffix-major" for step in
                       row["backend_metrics"]["steps"] if step["kind"] == "join")
        filters = [stage for stage in row["stages"] if stage["op"] == "filter"]
        assert len(filters) == 4
        assert {stage["alias"] for stage in filters} == {"c1", "c2", "e1", "e2"}
        assert len([stage for stage in row["stages"] if stage["op"] == "join"]) == 3
        signature = sorted((p["op"], p.get("alias", ""), p["predicate_key"])
                           for p in row["accuracy"]["per_predicate"])
        if predicates is None:
            predicates = signature
        assert signature == predicates
        rows[key] = {row["query"]: row for row in suite["passes"]["single"]["queries"]
                     if row["query"] != "FEV-9"}
        rows[key]["FEV-9"] = row
        added = json.loads((fev10_root / f"{key}.json").read_text())
        check_run(added, key, corpus, suite, fev10_manifest)
        (added_row,) = current_rows(added).values()
        assert added_row["query"] == "FEV-10"
        if key == "pipelined_sglang":
            assert all(step["submission"] == "suffix-major" for step in
                       added_row["backend_metrics"]["steps"]
                       if step["kind"] == "join")
        stages = added_row["stages"]
        assert {(s["op"], s["alias"]) for s in stages if s["op"] == "filter"} \
            == {("filter", "c"), ("filter", "e")}
        assert len([s for s in stages if s["op"] == "join"]) == 1
        rows[key]["FEV-10"] = added_row
    saved_quail, rerun_manifests = {}, []
    suite = json.loads((root / "quail.json").read_text())
    for quail_root in quail_roots:
        rerun_manifest = json.loads((quail_root / "manifest.json").read_text())
        rerun_manifests.append(rerun_manifest)
        for path in sorted(quail_root.glob("*.json")):
            if path.name == "manifest.json":
                continue
            rerun = json.loads(path.read_text())
            check_run(rerun, "quail", corpus, suite, rerun_manifest)
            for query, row in current_rows(rerun).items():
                saved_quail.setdefault(query, rows["quail"][query])
                rows["quail"][query] = row
    return (manifest, rows, fev9_manifest, fev10_manifest,
            sglang_update["result_volume_path"], saved_quail, rerun_manifests)


def rerun_lines(saved, rows, manifests, queries):
    """The report section comparing the Quail reruns with the saved rows."""
    if not saved:
        return []
    runs = "; ".join(
        f"`/results/benchmarks/quailb/family-runs/{manifest['run_id']}/` "
        "(function calls "
        + ", ".join(f"`{call}`" for call in manifest["function_call_ids"].values())
        + ")" for manifest in manifests)
    lines = [
        "## Quail rerun against the saved Quail rows", "",
        f"Quail only, {len(saved)} queries, from {runs}. The Quail bars and "
        "the Quail rows above come from these runs; the baseline rows are "
        "the saved runs. Recomputed KV is fresh tokens minus the fewest tokens",
        "the run's requests needed, derived from its saved answer tables; a",
        "run saved without them shows as not measured.", "",
        "| Query | Saved seconds | Rerun seconds | Change | Saved recomputed KV "
        "| Rerun recomputed KV | Saved fresh tokens | Rerun fresh tokens "
        "| Agreement saved / rerun, % | Rows saved / rerun |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for query in (query for query in queries if query in saved):
        before, after = row_metrics(saved[query]), row_metrics(rows["quail"][query])
        change = 100 * (after["seconds"] - before["seconds"]) / before["seconds"]
        old_rows = saved[query]["accuracy"]["output_accuracy"]["predicted_rows"]
        new_rows = rows["quail"][query]["accuracy"]["output_accuracy"]["predicted_rows"]
        lines.append(
            f"| {query} | {before['seconds']:.2f} | {after['seconds']:.2f} "
            f"| {change:+.1f}% | {token_cell(before['recomputed'])} "
            f"| {token_cell(after['recomputed'])} "
            f"| {before['fresh']:,} | {after['fresh']:,} "
            f"| {before['agreement']:.2f} / {after['agreement']:.2f} "
            f"| {old_rows:,} / {new_rows:,} |")
    return lines + [""]


def main(workdir, fev9_dir=None, fev10_dir=None, quail_dirs=()):
    """Regenerate figures from the saved suite and the current runs."""
    root = Path(workdir)
    fev9_root = Path(fev9_dir) if fev9_dir else root / "fev9"
    fev10_root = Path(fev10_dir) if fev10_dir else root / "fev10"
    quail_roots = [Path(path) for path in quail_dirs or []]
    (manifest, rows, fev9_manifest, fev10_manifest, sglang_source,
     saved_quail, rerun_manifests) = load_rows(
        root, fev9_root, fev10_root, quail_roots)
    queries = list(manifest["query_ids"])
    queries.insert(queries.index("FEV-9") + 1, "FEV-10")
    comparable = [query for query in queries if all(query in rows[key] for key in rows)]
    assert len(queries) == 33 and len(comparable) == 33
    assert all("error" not in row
               for method in rows.values() for row in method.values())
    faster = sum(rows["quail"][query]["wall_s"] < rows["stock_vllm"][query]["wall_s"]
                 for query in comparable)
    plt.style.use(HERE / "quail.mplstyle")
    plt.rcParams.update({"pdf.fonttype": 42, "figure.autolayout": False,
                         "savefig.bbox": None})
    corpus = json.loads((root / "corpus.json").read_text())
    relations = input_relations(queries, corpus)
    for method in rows.values():
        for query, row in method.items():
            assert (sum(count for _, _, count in relations[query])
                    == row["input_document_rows"])
    sol = load_sol(root, queries, rows, corpus)
    overview = plot_comparison("QUAIL-B", queries, rows, relations, sol,
                               "quailb_main.png", overview=True)
    fev = row_metrics(rows["quail"]["FEV-9"])
    sglang = row_metrics(rows["pipelined_sglang"]["FEV-9"])
    previous_sglang = row_metrics(json.loads(
        (fev9_root / "pipelined_sglang.json").read_text()
    )["passes"]["single"]["queries"][0])
    anchor_major_sglang = row_metrics(json.loads(
        (fev9_root / "sglang_anchor_major.json").read_text()
    )["suites"]["pipelined_sglang"]["passes"]["single"]["queries"][0])
    output = rows["quail"]["FEV-9"]["accuracy"]["output_accuracy"]
    fev10 = {key: row_metrics(rows[key]["FEV-10"]) for key, _, _ in METHODS}
    fev5 = {key: row_metrics(rows[key]["FEV-5"]) for key, _, _ in METHODS}
    lines = [
        "# QUAIL-B comparison from saved results", "",
        "- The main PDF covers all 33 queries with grouped bars and one metric "
        "per page.",
        "  Its final page lists input document counts. Each dataset PDF has a page",
        "  of four bar charts and a separate input-count page. Text and marks remain",
        "  vector content when zoomed. The PNGs below are first-page previews.",
        "- The five dataset plots use the same",
        "  method colors and definitions for latency, recomputed KV tokens, fresh",
        "  input tokens, accuracy, and input document counts for every relation alias.",
        "- 31 queries reuse the original measurements from September 5, 2026.",
        "  FEV-9 was rerun on September 6, 2026, with all four methods. FEV-10",
        "  joined the benchmark on September 11, 2026, and was measured that day",
        "  with all four methods.",
        "  FEV-9 and FEV-10 use SGLang with all anchors submitted per partner, "
        "without client tiles or request slices.",
        "  Other queries retain historical SGLang measurements with the earlier "
        "adapter.",
        "  The other queries are not new measurements of shared retention.",
        "- The setup was Qwen3 4B FP8, sf=0.1, lf=1, and one H100 per configuration.",
        "  Quail and the vLLM configurations shared a physical GPU within each family.",
        "  SGLang used a separate GPU. Stock vLLM used operator-at-a-time submission.",
        "- FEV-9 has four filters and three joins. All four methods now use that",
        "  query definition in both the main plot and the FEVER plot.",
        "  The [FEV-9 comparison](2026-09-06-sglang-baseline.md) records the new run.",
        "  The [retention report](2026-09-05-shared-kv-retention.md) records the "
        "earlier ablation.",
        f"- The revised SGLang adapter took {sglang['seconds']:.2f} seconds "
        "on FEV-9,",
        f"  compared with {previous_sglang['seconds']:.2f} seconds using its "
        "earlier submission policy.",
        f"  Fresh computation was {sglang['fresh']:,} tokens, compared with "
        f"{previous_sglang['fresh']:,} before.",
        "  The intermediate run with vLLM's pair order took "
        f"{anchor_major_sglang['seconds']:.2f} seconds",
        f"  and computed {anchor_major_sglang['fresh']:,} fresh tokens. "
        "The current SGLang order",
        "  separates requests sharing an anchor so earlier requests can populate "
        "reusable KV.",
        "- We predicted Quail would remain near 39 seconds and beat the baselines.",
        f"  It took {fev['seconds']:.2f} seconds in the new run. We reused all "
        "124 saved",
        "  configurations for the other 31 queries.",
        f"- In these saved measurements, Quail was faster than stock vLLM on {faster}",
        f"  of {len(comparable)} comparable queries.",
        "- A horizontal line across each query's bar group shows its SoL estimate.",
        "  SoL models ideal computation and memory traffic with unlimited prefix KV.",
        "  It credits matching token prefixes across requests, documents, and aliases.",
        "  It uses exact reference-label survivors and searches supported left-deep",
        "  join plans. Different answers can change the work done by measured runs,",
        "  so the gap from SoL is not purely execution overhead.",
        "- SoL uses the distinct-prefix estimate, not the per-document-only estimate.",
        "  Its latency, fresh-token count, and zero prefix recomputation are "
        "estimates.",
        "  No accuracy is assigned to SoL because it is not a measured model run.",
        "  Matching document prefixes are reusable; a partner suffix after a different",
        "  anchor context is not an identical prefix and is still computed.",
        "- SoL was recalculated for all 33 queries on the CPU on September 11, "
        "2026,",
        "  from saved labels and corpus rows. The 32 earlier estimates are "
        "unchanged",
        "  to the printed precision. Calculating SoL required no GPU inference.",
        "- FEV-10 is FEV-5 with one ordinary equality in the join: SUPPORT is "
        "asked",
        "  only of a claim and its own Wikipedia page. It is the only query whose",
        "  join has an equality. Predicted before its run: 3 to 5 seconds on stock",
        "  and pipelined vLLM and 4 to 7 on pipelined SGLang, against FEV-5's",
        f"  {fev5['stock_vllm']['seconds']:.2f}, "
        f"{fev5['pipelined_vllm']['seconds']:.2f}, and "
        f"{fev5['pipelined_sglang']['seconds']:.2f}. Measured: Quail "
        f"{fev10['quail']['seconds']:.2f} seconds,",
        f"  stock vLLM {fev10['stock_vllm']['seconds']:.2f}, pipelined vLLM "
        f"{fev10['pipelined_vllm']['seconds']:.2f}, and pipelined SGLang",
        f"  {fev10['pipelined_sglang']['seconds']:.2f}, with "
        f"{fev10['quail']['fresh']:,}, {fev10['stock_vllm']['fresh']:,}, "
        f"{fev10['pipelined_vllm']['fresh']:,}, and",
        f"  {fev10['pipelined_sglang']['fresh']:,} fresh tokens. The "
        "[feature note](shipped_features/"
        "2026-09-11-streamed-edges-pair-joins-plan-edits.md)",
        "  records the Quail run and the prediction for the baselines.",
        f"- FEV-9 SoL is {sol['FEV-9']['sol_s']:.3f} seconds with shared-prefix reuse,",
        f"  compared with {sol['FEV-9']['per_document']['sol_s']:.3f} seconds "
        "with reuse only",
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
        "- Recomputed KV is `regret_tokens`: fresh tokens minus the fewest input",
        "  tokens the run's requests needed with unlimited KV, where every",
        "  distinct prefix across the requests is computed once. It is derived",
        "  on the CPU after the run from the saved answer tables by quail-bench's",
        "  scoring (`quail_b.minimum`); the engine tracks nothing. A run saved",
        "  without that minimum is not measured.",
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
        "Source manifest on `quail-results`: "
        f"`{manifest['manifest_volume_path']}`.", "",
        "FEV-9 Quail and vLLM manifest on `quail-results`: "
        f"`{fev9_manifest['manifest_volume_path']}`.", "",
        f"Current FEV-9 SGLang result on `quail-results`: `{sglang_source}`.", "",
        "FEV-10 run on `quail-results`: "
        f"`/results/benchmarks/quailb/family-runs/{fev10_manifest['run_id']}/` "
        f"(function call `{fev10_manifest['function_call_ids']['fever:quail_vllm']}` "
        "for Quail and vLLM, "
        f"`{fev10_manifest['function_call_ids']['fever:sglang']}` for SGLang).", "",
        "SoL estimates on `quail-results`: "
        "`/results/sol/2026-09-11-quailb-prefix-reuse.json`.", "",
        "The FEV-10 estimate is also saved separately at "
        "`/results/sol/2026-09-11-fev10-prefix-reuse.json`, and the FEV-9 one at "
        "`/results/sol/2026-09-06-fev9-prefix-reuse.json`.", "",
        "Corpus counts on `quail-results`: `/results/ground_truth/quailb/"
        f"schema_v1/corpora/{corpus['corpus_id']}/manifest.json`.", "",
        "The manifest lists all four source suite paths. The download commands are",
        "in `reports/make_quailb_comparison_plots.py`.", "",
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
                      "| Output recall (%) |",
                      "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"])
        for query in selected:
            for key, label, _ in METHODS:
                if query not in rows[key]:
                    lines.append(f"| {query} | {label} | Not measured for this "
                                 "query definition | | | | | | | | |")
                    continue
                m = row_metrics(rows[key][query])
                lines.append(
                    f"| {query} | {label} | {m['seconds']:.2f} "
                    f"| {token_cell(m['recomputed'])} "
                    f"| {m['fresh']:,} | {m['throughput']:,.2f} "
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
                f"| {estimate['cost_usd_per_query_at_sol']:.5f} "
                "| Not measured | Not measured | Not measured |")
        lines.append("")
    lines.extend(rerun_lines(saved_quail, rows, rerun_manifests, queries))
    report = HERE / "2026-09-05-quailb-saved-results.md"
    report.write_text("\n".join(lines))
    print(f"Updated {report}, the main figure, and five dataset figures "
          "from 132 saved configurations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir")
    parser.add_argument("--fev9-dir")
    parser.add_argument("--fev10-dir")
    parser.add_argument("--quail-dir", action="append")
    args = parser.parse_args()
    main(args.workdir, args.fev9_dir, args.fev10_dir, args.quail_dir)
