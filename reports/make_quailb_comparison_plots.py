"""Generate the QUAIL-B overview and dataset plots without inference.

Pull the saved measurements, corpus manifest, and SoL estimates:

    W=/tmp/quail-comparison; mkdir -p "$W/run"
    RUN=benchmarks/quailb/family-runs/20260918T071546Z-all-chat
    uv run modal volume get quail-results "$RUN/manifest.json" "$W/run/manifest.json"
    uv run modal volume get quail-results "$RUN/measurements.parquet" \
      "$W/run/measurements.parquet"
    for METHOD in quail pipelined_vllm; do
      mkdir -p "$W/run/$METHOD"
      uv run modal volume get quail-results "$RUN/$METHOD/run.json" \
        "$W/run/$METHOD/run.json"
    done
    CORPUS=ground_truth/quailb/schema_v1/corpora/c_1aa2c4f0d0b6c816fd37aa5748c33341
    uv run modal volume get quail-results "$CORPUS/manifest.json" "$W/corpus.json"
    uv run modal volume get quail-results \
      sol/2026-09-18-all-chat/sol_quailb_sf0.1.json "$W/sol.json"
    BENCH=git+https://github.com/fsdatalab/quail-bench.git
    REV=fc27f35188f0fcc6a1f3b8fe3bfbb9e12d6842eb
    uv run --with matplotlib --with "quail-b@$BENCH@$REV" \
      python reports/make_quailb_comparison_plots.py "$W"

All queries use non-thinking chat prompts. The saved run reuses the completed
September 18 BioDEX measurements after checking its reference label identities.
"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from plot_colors import BLUE, DARK, ORANGE

from quail.specs import H100_USD_PER_HOUR

HERE = Path(__file__).resolve().parent
METHODS = [
    ("quail", "Quail", BLUE),
    ("pipelined_vllm", "Pipelined vLLM", ORANGE),
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
        "tokens_per_second": row["fresh_tokens"] / row["runtime_s"],
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
    ("tokens_per_second", "Total fresh input tokens per second", "tokens/second"),
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


def load_sol(root, queries, rows, corpus, manifest):
    """Load estimates matching the measured query and reference identities."""
    from quail_b.queries import get_query
    from quail_b.rendering import PROMPT_FORMAT

    source = json.loads((root / "sol.json").read_text())
    assert source["corpus_id"] == corpus["corpus_id"]
    assert source["collection_id"] == manifest["collection_id"]
    assert source["scale_factor"] == 0.1
    assert source["optimizer"]["persistent_kv_capacity"] == "unlimited"
    assert set(source["queries"]) == set(queries)
    estimates = {}
    for query, record in source["queries"].items():
        assert record["prompt_format"] == PROMPT_FORMAT
        assert record["plan_sha256"] == hashlib.sha256(
            get_query(query).plan_bytes).hexdigest()
        estimate = record["models"]["qwen3-4b-fp8"]
        assert estimate["input_document_rows"] == rows["quail"][query]["input_rows"]
        assert estimate["tokens"] <= estimate["per_document"]["tokens"], query
        estimates[query] = estimate
    return estimates


def series_value(rows, sol, method, query, metric):
    """Return a measured value or an explicitly modeled value."""
    if method == "sol":
        return {"seconds": sol[query]["sol_s"], "fresh": sol[query]["tokens"],
                "tokens_per_second": sol[query]["tokens"] / sol[query]["sol_s"],
                "recomputed": 0, "agreement": None}[metric]
    if query not in rows[method]:
        return None
    return row_metrics(rows[method][query])[metric]


def metric_bars(axis, queries, rows, sol, metric, overview):
    """Draw measured bars and a SoL line across each query group."""
    measured = METHODS
    unit = next(unit for key, _, unit in METRICS if key == metric)
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
            axis.set_ylabel(f"{unit} (log scale)")
    else:
        axis.set_ylim(0, maximum * 1.6 if maximum else 1)
        axis.set_ylabel(unit)
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
            quail_time = rows["quail"][query]["runtime_s"]
            baseline_time = rows["pipelined_vllm"][query]["runtime_s"]
            top = max(quail_time, baseline_time)
            ratio = baseline_time / quail_time
            axis.annotate(f"{ratio:.2f}x", (index, top), xytext=(0, 35),
                          textcoords="offset points", ha="center",
                          va="bottom", fontsize=8)
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
    latency_title = "Latency" if overview else "Latency (ratios: vLLM / Quail)"
    titles = {"seconds": latency_title, "recomputed": "Recomputed KV tokens",
              "fresh": "Fresh input tokens",
              "tokens_per_second": "Total fresh input tokens per second",
              "agreement": "Answer agreement with reference labels"}
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
        ["seconds", "fresh", "recomputed", "agreement"],
        ["tokens_per_second"]]
    with PdfPages(destination.with_suffix(".pdf")) as pdf:
        for metrics in groups:
            single = len(metrics) == 1
            figure, axes = (plt.subplots(1, 1, figsize=(14, 8.5)) if single
                            else plt.subplots(2, 2, figsize=(14, 10)))
            axes = [axes] if single else list(axes.flat)
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
            figure.subplots_adjust(left=0.075, right=0.97, top=0.83,
                                   bottom=0.14 if overview else 0.10, hspace=0.60,
                                   wspace=0.28)
            figure.text(0.075, 0.02,
                        "Reference labels: Qwen3 32B FP8 and dataset annotations.",
                        fontsize=9)
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


def load_rows(root, corpus):
    """Load complete measurements for both methods and all 33 queries."""
    from quail_b.queries import get_query
    from quail_b.rendering import PROMPT_FORMAT
    from quail_b.run import _query_hash

    manifest = json.loads((root / "run" / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["model"] == "qwen3-4b-fp8"
    assert manifest["sf"] == 0.1
    assert set(manifest["query_ids"]) == set(QUERY_ORDER)
    assert set(manifest["methods"]) == {key for key, _, _ in METHODS}
    rows = measurement_rows(root / "run" / "measurements.parquet")
    suites = {}
    for method, _, _ in METHODS:
        suite = json.loads((root / "run" / method / "run.json").read_text())
        assert suite["corpus_id"] == corpus["corpus_id"]
        assert suite["collection_id"] == manifest["collection_id"]
        assert suite["metadata"]["prompt_format"] == PROMPT_FORMAT
        assert set(rows[method]) == set(QUERY_ORDER)
        assert {item["id"] for item in suite["queries"]} == set(QUERY_ORDER)
        for item in suite["queries"]:
            assert item["status"] == "complete"
            assert item["definition_hash"] == _query_hash(get_query(item["id"]))
            assert rows[method][item["id"]]["regret_tokens"] is not None
        suites[method] = suite
    return rows, manifest, suites


def main(workdir):
    """Regenerate the report and figures from the downloaded measurements."""
    root = Path(workdir)
    corpus = json.loads((root / "corpus.json").read_text())
    rows, manifest, suites = load_rows(root, corpus)
    queries = list(QUERY_ORDER)
    plt.style.use(HERE / "quail.mplstyle")
    plt.rcParams.update({"pdf.fonttype": 42, "figure.autolayout": False,
                         "savefig.bbox": None})
    relations = input_relations(queries, corpus)
    for method in rows.values():
        for query, row in method.items():
            assert sum(count for _, _, count in relations[query]) == row["input_rows"]
    sol = load_sol(root, queries, rows, corpus, manifest)
    overview = plot_comparison("QUAIL-B", queries, rows, relations, sol,
                               "quailb_main.pdf", overview=True)
    faster = sum(rows["quail"][q]["runtime_s"]
                 < rows["pipelined_vllm"][q]["runtime_s"] for q in queries)
    speedups = {q: rows["pipelined_vllm"][q]["runtime_s"]
                / rows["quail"][q]["runtime_s"] for q in queries}
    fastest = max(speedups, key=speedups.get)
    settings = [item["measurements"]["backend_metrics"]["capacity"]
                for item in suites["pipelined_vllm"]["queries"]]
    batch_tokens = sorted({item["max_num_batched_tokens"] for item in settings})
    sequences = sorted({item["max_num_seqs"] for item in settings})
    kv_capacity = [item["kv_cache_size_tokens"] for item in settings]
    bio2 = row_metrics(rows["quail"]["BIO-2"])
    bio3 = row_metrics(rows["quail"]["BIO-3"])
    checks = manifest["reference_checks"]
    kv_range = (f"{min(kv_capacity):,} tokens"
                if min(kv_capacity) == max(kv_capacity)
                else f"{min(kv_capacity):,} to {max(kv_capacity):,} tokens")
    imdb9 = row_metrics(rows["quail"]["IMDB-9"])
    fev8 = row_metrics(rows["quail"]["FEV-8"])
    lines = [
        "# QUAIL-B comparison", "",
        "- All 33 queries use Qwen3's chat format with thinking disabled.",
        "  Both methods use Qwen3 4B FP8, sf=0.1, lf=1, and one H100.",
        "  Quail and pipelined stock vLLM share a physical GPU within each family.",
        "  Pipelined stock vLLM advances documents through filter stages",
        "  independently, then starts joins after filtering finishes.",
        "- The measured run is on `quail-results`:",
        f"  `/results/benchmarks/quailb/family-runs/{manifest['run_id']}/`.",
        "  It reuses the completed BioDEX run after checking that its corpus,",
        "  query definitions, prompt format, and reference answers match.",
        f"  BioDEX source: `{manifest['reused_biodex']}`.",
        "  All other measurements are new. Earlier prompt formats are omitted.",
        "- Model references use Qwen3 32B FP8 with thinking disabled.",
        f"  The collection is `{manifest['collection_id']}`.",
        "  Reference join prompts put the first argument first. FEVER joins",
        "  were regenerated after fixing automatic prompt reordering. Saved",
        "  benchmark answers were rescored without changing timings.",
        "  The existing FEVER annotation and LePaRD citation rules still apply.",
        f"  Saved-answer checks repeated {checks['compared']} answers",
        f"  with {checks['answer_differences']} differences.",
        "- Quail's planned limits are 110,376 tokens per chunk and 362,250",
        "  resident KV tokens. Pipelined stock vLLM uses prefix caching,",
        f"  {', '.join(f'{n:,}' for n in batch_tokens)} batched tokens, and",
        f"  {', '.join(f'{n:,}' for n in sequences)} sequences.",
        f"  Its measured KV capacity is {kv_range}.",
        "  Both methods use the same planner's filter and join ordering rules.",
        f"- Quail is faster on {faster} of {len(queries)} queries.",
        f"  The arithmetic mean speedup is {mean(speedups.values()):.2f}x,",
        f"  the median is {median(speedups.values()):.2f}x, and the maximum is",
        f"  {speedups[fastest]:.2f}x on {fastest}. Each query has equal weight.",
        "  Speedup is pipelined stock vLLM time divided by Quail time.",
        "  Query time excludes startup and result collection. Table throughput counts",
        "  input documents for filters and evaluated pairs across stages for joins.",
        f"  GPU cost is query seconds / 3,600 * ${H100_USD_PER_HOUR:.4f}.",
        "- SoL means speed of light. It estimates ideal GPU time by dividing",
        "  arithmetic and memory traffic by the hardware's peak rates. For each",
        "  model component, it takes the larger time, then adds component times.",
        "  It assumes ideal batching and unlimited retained KV. Matching token",
        "  prefixes are computed once across requests, documents, and aliases.",
        "  It excludes startup and software scheduling overhead and uses exact",
        "  reference-label survivors. The supported join search uses left-deep",
        "  plans. Different measured answers change the work, so the gap from",
        "  SoL is not purely execution overhead. SoL has no measured accuracy.",
        "- Fresh input tokens count every input position processed by a model",
        "  forward pass. Repeated computation counts again. Recomputed KV tokens",
        "  are fresh tokens minus the minimum for the run's actual requests with",
        "  unlimited KV. They are included in fresh tokens, not added to them.",
        "  The benchmark computes this minimum from saved answers after the run.",
        "  Token throughput is total fresh input tokens divided by query seconds.",
        "  It includes recomputation and excludes generated answer tokens.",
        "  More recomputation can raise this rate without making a query faster.",
        "- Answer agreement counts matching evaluated predicate answers. Output",
        "  precision is the fraction of returned rows matching the reference.",
        "  Output recall is the fraction of reference rows returned. Most join",
        "  pairs can be negative, so high agreement can coexist with low recall.",
        f"  Quail's BIO-2 and BIO-3 recall is {bio2['recall']:.2f}% and",
        f"  {bio3['recall']:.2f}%, respectively.",
        f"  Its IMDB-9 output recall is {imdb9['recall']:.4f}%, and its FEV-8",
        f"  output precision is {fev8['precision']:.4f}%.",
        "- The main PDF shows all queries with one metric per page. Each dataset",
        "  PDF includes every query in that dataset. Input counts list each alias",
        "  separately, before filtering. SoL is a horizontal line, not a measured",
        "  bar. Latency labels show vLLM time divided by Quail time.",
        "  Log scales are labeled; a dash marks zero.", "",
        f"[Open the main vector PDF](plots/{overview})", "",
        f"Figure: plots/{overview}", "",
        "SoL estimates on `quail-results`:",
        "`/results/sol/2026-09-18-all-chat/sol_quailb_sf0.1.json`.", "",
        "The download commands are in `reports/make_quailb_comparison_plots.py`.", "",
    ]
    for family in ("IMDB", "BIO", "FEV", "LEP", "AGENT"):
        selected = [q for q in queries if q.startswith(family + "-")]
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
            for method, _, _ in METHODS:
                pairs = rows[method]["BIO-3"]["evaluated_document_pairs"]
                assert pairs % terms == 0
                survivors[method] = pairs // terms
            lines.extend([
                "", "BIO-3 filter survivors: "
                f"{sol['BIO-3']['documents_after_filters']:,} in the reference, "
                f"{survivors['quail']:,} in Quail, and "
                f"{survivors['pipelined_vllm']:,} in pipelined stock vLLM.",
                "", "The cause of vLLM's lower BIO-2 time than its earlier raw-prompt",
                "run remains unknown. The runs did not isolate prompt changes",
                "from other execution changes."])
        lines.extend(["",
                      "| Query | Method | Seconds | Recomputed KV tokens "
                      "| Fresh input tokens | Tokens/second | Throughput "
                      "| Unit | $/query "
                      "| Answer agreement (%) | Output precision (%) "
                      "| Output recall (%) |",
                      "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"])
        for query in selected:
            for key, label, _ in METHODS:
                m = row_metrics(rows[key][query])
                lines.append(
                    f"| {query} | {label} | {m['seconds']:.2f} "
                    f"| {token_cell(m['recomputed'])} "
                    f"| {m['fresh']:,} | {m['tokens_per_second']:,.2f} "
                    f"| {m['throughput']:,.2f} "
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
                f"| {estimate['tokens']:,.0f} "
                f"| {estimate['tokens'] / estimate['sol_s']:,.2f} "
                f"| {throughput:,.2f} | {unit} "
                f"| {estimate['cost_usd_per_query_at_sol']:.5f} "
                "| Not measured | Not measured | Not measured |")
        lines.append("")
    report = HERE / "quailb-comparison.md"
    report.write_text("\n".join(lines))
    print(f"Updated {report}, the main figure, and five dataset figures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir")
    main(parser.parse_args().workdir)
