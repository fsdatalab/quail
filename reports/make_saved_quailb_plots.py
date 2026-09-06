"""Plot the saved suite without rerunning inference.

Pull the original suite manifest and its four result files:

    W=/tmp/quail-shared-comparison; mkdir -p "$W"
    uv run modal volume get quail-results \
      benchmarks/quailb/family-runs/20260905T021527Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/manifest.json \
      "$W/manifest.json"
    for method in quail stock_vllm pipelined_vllm pipelined_sglang; do
      source_path=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["result_volume_paths"][sys.argv[2]].removeprefix("/results/"))' "$W/manifest.json" "$method")
      uv run modal volume get quail-results "$source_path" "$W/$method.json"
    done
    uv run --with matplotlib python reports/make_saved_quailb_plots.py "$W"

FEV-9 is omitted because the old suite measured a different query. Its current
measurements and accuracy are in 2026-09-05-shared-kv-retention.md.
"""

import json
import sys
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
        "throughput": count / row["wall_s"],
        "unit": "pairs/s" if joins else "docs/s",
        "cost": row["wall_s"] / 3600 * H100_USD_PER_HOUR,
        "agreement": 100 * answers["correct"] / answers["evaluated"],
        "precision": 100 * matched / predicted if predicted else (0 if expected else 100),
        "recall": 100 * matched / expected if expected else (0 if predicted else 100),
    }


def plot_family(family, queries, rows):
    """Plot each query's saved runtime and reference answer agreement."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 1.35 * len(queries) + 2.2), sharey=True)
    for index, (key, label, color) in enumerate(METHODS):
        positions = [5 * q + index for q in range(len(queries))]
        for axis, metric in zip(axes, ("seconds", "agreement")):
            values = [row_metrics(rows[key][query])[metric] for query in queries]
            axis.barh(positions, values, height=0.75, color=color, label=label)
            for query, position, value in zip(queries, positions, values):
                text = f"{value:.2f}"
                if key == "quail" and metric == "seconds":
                    delta = 100 * (value / rows["stock_vllm"][query]["wall_s"] - 1)
                    text += f" ({delta:+.0f}%)"
                axis.annotate(text, (value, position), xytext=(4, 0),
                              textcoords="offset points", va="center", fontsize=9)
    axes[0].set_yticks([5 * index + 1.5 for index in range(len(queries))], queries)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("seconds")
    axes[0].set_title(f"{family} query time\nQuail time change vs. stock vLLM")
    times = [rows[key][query]["wall_s"] for key, _, _ in METHODS for query in queries]
    if max(times) / min(times) > 10:
        axes[0].set_xscale("log")
        axes[0].set_xlim(min(times) / 2, max(times) * 3)
        axes[0].set_xlabel("seconds (log scale)")
    else:
        axes[0].set_xlim(0, max(times) * 1.4)
    axes[1].set_xlim(0, 112)
    axes[1].set_xlabel("percent")
    axes[1].set_title(f"{family} answer agreement with Qwen3 32B")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.95), w_pad=4)
    destination = HERE / "plots" / f"saved_quailb_{family.lower()}.png"
    figure.savefig(destination, dpi=300)
    plt.close(figure)
    return destination.name


def main(workdir):
    """Regenerate figures and a report from the original suite files."""
    root = Path(workdir)
    manifest = json.loads((root / "manifest.json").read_text())
    rows = {}
    for key, _, _ in METHODS:
        suite = json.loads((root / f"{key}.json").read_text())
        assert (suite["model"], suite["sf"], suite["lf"], suite["gpus"]) == (
            "qwen3-4b-fp8", 0.1, 1, 1)
        rows[key] = {row["query"]: row for row in suite["passes"]["single"]["queries"]}
    queries = [query for query in manifest["query_ids"] if query != "FEV-9"]
    assert len(queries) == 31
    assert all("error" not in rows[key][query] for key in rows for query in queries)
    faster = sum(rows["quail"][query]["wall_s"] < rows["stock_vllm"][query]["wall_s"]
                 for query in queries)
    plt.style.use(HERE / "quail.mplstyle")
    lines = [
        "# QUAIL-B comparison from saved results", "",
        "- The 31 queries below reuse the original measurements from September 5, 2026.",
        "  No inference was rerun for this report. These are historical measurements,",
        "  not a measurement of shared retention on every query.",
        "- The setup was Qwen3 4B FP8, sf=0.1, lf=1, and one H100 per configuration.",
        "  Quail and the vLLM configurations shared a physical GPU within each family.",
        "  SGLang used a separate GPU. Stock vLLM used operator-at-a-time submission.",
        "- FEV-9 now has four filters. The old suite had only one filter for FEV-9,",
        "  so its old measurements are excluded. The current query is reported in",
        "  [the shared retention comparison](2026-09-05-shared-kv-retention.md).",
        "- The prediction for this update was that scoring and plotting would need no",
        "  inference. We reused all 124 saved configurations for the other 31 queries.",
        f"- In these saved measurements, Quail was faster than stock vLLM on {faster}",
        f"  of {len(queries)} queries. The figures annotate Quail's change in time",
        "  relative to stock vLLM. Positive percentages mean Quail took longer.",
        "- Answer agreement measures evaluated calls against saved Qwen3 32B labels.",
        "  Each method can evaluate different calls after its filters and joins.",
        "  Output precision is the fraction of returned rows matching the reference.",
        "  Output recall is the fraction of reference rows returned. High answer",
        "  agreement can coexist with poor final output precision.",
        "- Query time excludes startup. Throughput counts input documents for filters",
        "  and evaluated document pairs across all stages for joins. GPU cost is query",
        f"  seconds divided by 3,600 and multiplied by ${H100_USD_PER_HOUR:.4f}.",
        "- All figures show runtime and answer agreement. The tables also show",
        "  throughput, cost, and final output precision and recall.", "",
        f"Source manifest on `quail-results`: `{manifest['manifest_volume_path']}`.", "",
        "The manifest lists all four source suite paths. The download commands are",
        "in `reports/make_saved_quailb_plots.py`.", "",
    ]
    for family in ("IMDB", "BIO", "FEV", "LEP", "AGENT"):
        selected = [query for query in queries if query.startswith(family + "-")]
        name = plot_family(family, selected, rows)
        lines.extend([f"## {family}", "", f"![{family} saved results](plots/{name})", "",
                      f"Figure: plots/{name}", "",
                      "| Query | Method | Seconds | Throughput | Unit | $/query | Answer agreement (%) | Output precision (%) | Output recall (%) |",
                      "|---|---|---:|---:|---|---:|---:|---:|---:|"])
        for query in selected:
            for key, label, _ in METHODS:
                m = row_metrics(rows[key][query])
                lines.append(
                    f"| {query} | {label} | {m['seconds']:.2f} | {m['throughput']:,.2f} "
                    f"| {m['unit']} | {m['cost']:.5f} | {m['agreement']:.2f} "
                    f"| {m['precision']:.5g} | {m['recall']:.5g} |")
        lines.append("")
    report = HERE / "2026-09-05-quailb-saved-results.md"
    report.write_text("\n".join(lines))
    print(f"Updated {report} and five family figures from 124 saved configurations.")


if __name__ == "__main__":
    main(sys.argv[1])
