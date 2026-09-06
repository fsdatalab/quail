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
    uv run --with matplotlib python reports/make_quailb_comparison_plots.py "$W"

Use --retention-dir to point to an already pulled retention comparison.
The current FEV-9 replaces the old Quail measurement. The old baseline FEV-9
measurements are excluded because they used a different query definition.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from plot_colors import BLUE, GRAY, GREEN, ORANGE
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


def set_metric_scale(axis, metric, values):
    """Choose a scale that preserves zero recomputation counts."""
    if metric == "agreement":
        axis.set_xlim(0, 115)
        axis.set_xlabel("percent")
    elif metric == "recomputed" and max(values) > 10:
        axis.set_xscale("symlog", linthresh=1)
        axis.set_xlim(0, max(values) * 15)
        axis.set_xlabel("tokens (linear to 1, then log)")
    elif min(values) > 0 and max(values) / min(values) > 10:
        axis.set_xscale("log")
        axis.set_xlim(min(values) / 2, max(values) * 6)
        axis.set_xlabel("seconds (log scale)" if metric == "seconds" else "tokens (log scale)")
    else:
        axis.set_xlim(0, max(values) * 1.7 if max(values) else 1)
        axis.set_xlabel("seconds" if metric == "seconds" else "tokens")
        if not max(values):
            axis.set_xticks([0])


def plot_comparison(title, queries, rows, relations, name, overview=False):
    """Plot the standard metrics and per-alias input counts."""
    height = 19 if overview else 1.25 * len(queries) + 2.8
    figure, axes = plt.subplots(1, 5, figsize=(23, height), sharey=True,
                               gridspec_kw={"width_ratios": [1.0, 1.7, 1.7, 1.7, 1.7]})
    spacing = 1 if overview else 5
    offsets = [(index - 1.5) * (0.18 if overview else 1) for index in range(4)]
    positions = [spacing * index for index in range(len(queries))]
    labels = [query + ("*" if query == "FEV-9" else "") for query in queries]
    axes[0].set_yticks(positions, labels)
    axes[0].set_xticks([])
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(positions[-1] + spacing * 0.65, -spacing * 0.65)
    axes[0].set_title("Input documents\nper relation alias")
    for query, position in zip(queries, positions):
        counts = [f"{alias}={count:,}" for alias, _, count in relations[query]]
        shown = "\n".join(", ".join(counts[index:index + 2]) for index in range(0, len(counts), 2))
        axes[0].text(0.03, position, shown, va="center", fontsize=9)
    for axis, (metric, metric_title, _) in zip(axes[1:], METRICS):
        all_values = []
        for index, (key, label, color) in enumerate(METHODS):
            available = [(q, query) for q, query in enumerate(queries) if query in rows[key]]
            ys = [q * spacing + offsets[index] for q, _ in available]
            values = [row_metrics(rows[key][query])[metric] for _, query in available]
            all_values.extend(values)
            axis.scatter(values, ys, color=color, s=25 if overview else 30,
                         label=label, clip_on=False)
            for (_, query), y, value in zip(available, ys, values):
                if overview and query != "FEV-9":
                    continue
                shown = f"{value:,.0f}" if metric in ("recomputed", "fresh") else f"{value:.2f}"
                if (not overview and metric == "seconds" and key == "quail"
                        and query in rows["stock_vllm"]):
                    delta = 100 * (value / rows["stock_vllm"][query]["wall_s"] - 1)
                    shown += f" ({delta:+.0f}%)"
                axis.annotate(shown, (value, y), xytext=(5, 0),
                              textcoords="offset points", va="center", fontsize=9)
        set_metric_scale(axis, metric, all_values)
        axis.set_title(metric_title + ("\nQwen3 32B reference" if metric == "agreement" else ""))
    handles, labels = axes[1].get_legend_handles_labels()
    figure.suptitle(f"{title}, Qwen3 4B FP8, sf=0.1, one H100", y=0.995, fontsize=15)
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.97),
                  ncol=4, frameon=False, fontsize=11)
    footer = "Fresh input tokens include recomputed KV tokens. Counts are before filters. Stock vLLM uses operator-at-a-time submission."
    if not overview:
        footer += " Quail latency labels show the change relative to stock vLLM."
    if "FEV-9" in queries:
        footer += "\n* FEV-9 uses shared retention and four filters. Baselines are unavailable for this definition; their missing points are not zeros."
        footer += " Other queries reuse the saved suite."
    figure.text(0.5, 0.015, footer, ha="center", fontsize=9, linespacing=1.6)
    figure.tight_layout(rect=(0, 0.06 if overview else 0.09, 1, 0.935), w_pad=3)
    destination = HERE / "plots" / name
    figure.savefig(destination, dpi=300)
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
    corpus = json.loads((root / "corpus.json").read_text())
    relations = input_relations(queries, corpus)
    for method in rows.values():
        for query, row in method.items():
            assert sum(count for _, _, count in relations[query]) == row["input_document_rows"]
    overview = plot_comparison("QUAIL-B", queries, rows, relations, "quailb_main.png", overview=True)
    fev = row_metrics(rows["quail"]["FEV-9"])
    output = rows["quail"]["FEV-9"]["accuracy"]["output_accuracy"]
    lines = [
        "# QUAIL-B comparison from saved results", "",
        "- The main plot covers all 32 queries. The five dataset plots use the same",
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
        f"  of {len(comparable)} comparable queries. Dataset figures annotate Quail's change in time",
        "  relative to stock vLLM. Positive percentages mean Quail took longer.",
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
        "  Its plots use a linear scale from 0 to 1 token and a log scale above 1",
        "  when counts span a large range. Zero recomputation remains visible.",
        "- Document counts come from the saved corpus manifest and describe inputs",
        "  before filtering. Repeated aliases each list their full input count.",
        "  The report tables also show throughput, GPU cost, and final output quality.",
        f"- FEV-9 agrees with the reference on {fev['agreement']:.2f}% of evaluated answers. Its final",
        f"  output matches only {output['matching_rows']:,} reference rows out of {output['predicted_rows']:,} returned rows.",
        f"  The reference has {output['expected_rows']:,} rows, so output precision is approximately {fev['precision']:.8f}%",
        f"  and recall is {fev['recall']:.2f}%. The retention change preserved all answers.", "",
        f"![QUAIL-B main comparison](plots/{overview})", "",
        f"Figure: plots/{overview}", "",
        f"Source manifest on `quail-results`: `{manifest['manifest_volume_path']}`.", "",
        f"Current FEV-9 source on `quail-results`: `{retention_source}/`.", "",
        f"Corpus counts on `quail-results`: `/results/ground_truth/quailb/schema_v1/corpora/{corpus['corpus_id']}/manifest.json`.", "",
        "The manifest lists all four source suite paths. The download commands are",
        "in `reports/make_quailb_comparison_plots.py`.", "",
    ]
    for family in ("IMDB", "BIO", "FEV", "LEP", "AGENT"):
        selected = [query for query in queries if query.startswith(family + "-")]
        name = plot_comparison(f"QUAIL-B {family}", selected, rows, relations, f"quailb_{family.lower()}.png")
        lines.extend([f"## {family}", "", f"![{family} saved results](plots/{name})", "",
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
