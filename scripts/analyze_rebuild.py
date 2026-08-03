#!/usr/bin/env python3
"""Summarize rebuilt runtime, cost, and non-regression results."""

import argparse
import gzip
import json
from pathlib import Path
import random
import statistics


def load_results(root: Path):
    rows = []
    for metadata_path in sorted(root.glob("*/metadata.json")):
        result_path = metadata_path.parent / "result.json.gz"
        if not result_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        with gzip.open(result_path, "rt") as handle:
            result = json.load(handle)
        rows.append((metadata, result))
    return rows


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[index]


def bootstrap_median_upper(values, seed=20260803, repetitions=50_000):
    rng = random.Random(seed)
    medians = []
    for _ in range(repetitions):
        sample = [rng.choice(values) for _ in values]
        medians.append(statistics.median(sample))
    return percentile(medians, 0.95)


def build_summary(rows, cost_catalog):
    comparisons = []
    for metadata, result in rows:
        if metadata["phase"] != "packing-compare":
            continue
        custom = result["custom"]
        stock = result["stock"]
        shared = set(custom["answers"]) & set(stock["answers"])
        flips = sorted(
            key for key in shared
            if custom["answers"][key] != stock["answers"][key]
        )
        gpu_ns = sum(
            step["duration_ns"] for step in custom.get("trace", [])
        )
        comparisons.append({
            "run_id": metadata["run_id"],
            "custom_seconds": custom["wall_ns"] / 1e9,
            "stock_seconds": stock["wall_ns"] / 1e9,
            "custom_to_stock": custom["wall_ns"] / stock["wall_ns"],
            "answer_flips": flips,
            "lengths_match": (
                custom.get("body_token_lengths")
                == stock.get("body_token_lengths")
            ),
            "model_runner_seconds": gpu_ns / 1e9,
            "runtime_overhead_seconds": (
                custom["wall_ns"] - gpu_ns
            ) / 1e9,
        })
    ratios = [row["custom_to_stock"] for row in comparisons]
    non_regression = {
        "count": len(comparisons),
        "median_custom_to_stock": (
            statistics.median(ratios) if ratios else None
        ),
        "upper_95_bootstrap_median": (
            bootstrap_median_upper(ratios) if ratios else None
        ),
        "answer_flips": sum(
            len(row["answer_flips"]) for row in comparisons
        ),
        "length_vectors_match": all(
            row["lengths_match"] for row in comparisons
        ),
        "rows": comparisons,
    }
    non_regression["passes"] = bool(
        ratios
        and non_regression["median_custom_to_stock"] <= 1.0
        and non_regression["upper_95_bootstrap_median"] <= 1.02
        and non_regression["answer_flips"] == 0
        and non_regression["length_vectors_match"]
    )

    fixed_shapes = {}
    for metadata, result in rows:
        config = metadata.get("config", {})
        if (
            metadata["phase"] != "custom-smoke"
            or config.get("short_circuit") is not False
            or config.get("n_docs") != 1
            or config.get("document_tokens") not in {300, 3000, 30000}
        ):
            continue
        key = (config["document_tokens"], config["k"])
        fixed_shapes[key] = {
            "run_id": metadata["run_id"],
            "seconds": result["wall_ns"] / 1e9,
            "steps": result["steps"],
            "body_token_lengths": result.get("body_token_lengths"),
            "accuracy": result["accuracy"],
        }
    shape_rows = [
        {
            "document_tokens": length,
            "k": k,
            **value,
        }
        for (length, k), value in sorted(fixed_shapes.items())
    ]

    kernel_rows = []
    for row in cost_catalog.get("comparisons", []):
        kernel_rows.append({
            "run_id": row["run_id"],
            "groups": row["groups"],
            "k": row["k"],
            "prefix_tokens": row["prefix_tokens"],
            "tail_tokens": row["tail_tokens"],
            "cascade_microseconds": row["cascade_ns"] / 1000,
            "standard_microseconds": row["standard_ns"] / 1000,
            "speedup": row["speedup"],
        })
    return {
        "non_regression": non_regression,
        "fixed_work_shapes": shape_rows,
        "attention_cost_validation": cost_catalog.get("validation", {}),
        "attention_kernel_shapes": kernel_rows,
    }


def render_markdown(summary):
    gate = summary["non_regression"]
    cost = summary["attention_cost_validation"]
    lines = [
        "# Rebuild results",
        "",
        "## What was tested",
        "",
        "The logical query is an ordered conjunction. Ground truth stayed outside the runtime. The custom path used FP8 KV pages and called the pinned vLLM model runner without the vLLM scheduler or KV manager.",
        "",
        "## Custom runtime versus stock vLLM",
        "",
        f"- Repetitions: {gate['count']}",
        f"- Median custom time divided by stock time: {gate['median_custom_to_stock']:.4f}" if gate["median_custom_to_stock"] is not None else "- No runs",
        f"- 95 percent upper bootstrap bound: {gate['upper_95_bootstrap_median']:.4f}" if gate["upper_95_bootstrap_median"] is not None else "- No bound",
        f"- Answer flips: {gate['answer_flips']}",
        f"- Gate: {'passed' if gate['passes'] else 'failed'}",
        "",
        "| Run | Custom | Stock | Custom / stock | Runtime overhead | Flips |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in gate["rows"]:
        lines.append(
            f"| `{row['run_id']}` | {row['custom_seconds']:.4f} s | "
            f"{row['stock_seconds']:.4f} s | {row['custom_to_stock']:.4f} | "
            f"{row['runtime_overhead_seconds']:.4f} s | "
            f"{len(row['answer_flips'])} |"
        )
    lines.extend([
        "",
        "## Fixed-work fused k",
        "",
        "Each row executes all four filters. These are one-document operator checks.",
        "",
        "| Target document tokens | k | Time | Steps | Accuracy |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in summary["fixed_work_shapes"]:
        lines.append(
            f"| {row['document_tokens']} | {row['k']} | "
            f"{row['seconds']:.4f} s | {row['steps']} | "
            f"{100 * row['accuracy']:.2f}% |"
        )
    lines.extend([
        "",
        "## Attention cost estimator",
        "",
        f"- Held-out median absolute error: {100 * cost.get('median_absolute_error', 0):.2f}%",
        f"- Held-out 95th-percentile absolute error: {100 * cost.get('p95_absolute_error', 0):.2f}%",
        f"- Gate: {cost.get('status', 'unknown')}",
        "",
        "Cascade attention is not always faster. The planner must use the measured shape table and choose ordinary attention when cascade merge overhead is larger than the saved prefix work.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/runs")
    parser.add_argument(
        "--catalog",
        default="results/BATCH_COST_CATALOG.json",
    )
    parser.add_argument("--json", default="results/REBUILD_SUMMARY.json")
    parser.add_argument("--markdown", default="notes/REBUILD_RESULTS.md")
    args = parser.parse_args()
    rows = load_results(Path(args.root))
    catalog = json.loads(Path(args.catalog).read_text())
    summary = build_summary(rows, catalog)
    Path(args.json).write_text(json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    ))
    Path(args.markdown).write_text(render_markdown(summary))
    print(
        f"summarized {summary['non_regression']['count']} comparison runs"
    )


if __name__ == "__main__":
    main()
