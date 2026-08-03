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
        "median_runtime_overhead_fraction": (
            statistics.median(
                row["runtime_overhead_seconds"]
                / row["model_runner_seconds"]
                for row in comparisons
            )
            if comparisons else None
        ),
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
            "answers": result["answers"],
        }
    shape_rows = [
        {
            "document_tokens": length,
            "k": k,
            **value,
        }
        for (length, k), value in sorted(fixed_shapes.items())
    ]
    for row in shape_rows:
        baseline = fixed_shapes.get((row["document_tokens"], 1))
        row["answer_flips_from_k1"] = (
            sum(
                answer != baseline["answers"].get(key)
                for key, answer in row["answers"].items()
            )
            if baseline is not None else None
        )
        del row["answers"]

    headline_by_key = {}
    for metadata, result in rows:
        config = metadata.get("config", {})
        n_docs = config.get("n_docs")
        if n_docs not in {2000, 10000}:
            continue
        if metadata["phase"] == "stock-smoke":
            key = (n_docs, "stock")
        elif (
            metadata["phase"] == "custom-smoke"
            and config.get("k") == 1
        ):
            key = (n_docs, "custom-k1")
        else:
            continue
        runner_ns = sum(
            step["duration_ns"] for step in result.get("trace", [])
        )
        headline_by_key[key] = {
            "run_id": metadata["run_id"],
            "seconds": result["wall_ns"] / 1e9,
            "answers": result["answers"],
            "body_token_lengths": result.get("body_token_lengths"),
            "model_runner_seconds": runner_ns / 1e9,
        }
    headlines = []
    for n_docs in (2000, 10000):
        custom = headline_by_key.get((n_docs, "custom-k1"))
        stock = headline_by_key.get((n_docs, "stock"))
        if custom is None or stock is None:
            continue
        flips = sum(
            custom["answers"][key] != stock["answers"].get(key)
            for key in custom["answers"]
        )
        headlines.append({
            "n_docs": n_docs,
            "custom_run_id": custom["run_id"],
            "stock_run_id": stock["run_id"],
            "custom_seconds": custom["seconds"],
            "stock_seconds": stock["seconds"],
            "stock_to_custom_speedup": (
                stock["seconds"] / custom["seconds"]
            ),
            "answer_count": len(custom["answers"]),
            "answer_flips": flips,
            "lengths_match": (
                custom["body_token_lengths"]
                == stock["body_token_lengths"]
            ),
            "model_runner_seconds": custom["model_runner_seconds"],
            "runtime_overhead_seconds": (
                custom["seconds"] - custom["model_runner_seconds"]
            ),
            "runtime_overhead_fraction": (
                (
                    custom["seconds"]
                    - custom["model_runner_seconds"]
                ) / custom["model_runner_seconds"]
                if custom["model_runner_seconds"] else None
            ),
        })

    fused_scale = []
    k1_2000 = headline_by_key.get((2000, "custom-k1"))
    if k1_2000 is not None:
        for metadata, result in rows:
            config = metadata.get("config", {})
            if not (
                metadata["phase"] == "custom-smoke"
                and config.get("n_docs") == 2000
                and config.get("k") in {2, 4}
                and config.get("multigroup_cascade") is False
            ):
                continue
            shared = (
                set(k1_2000["answers"])
                & set(result["answers"])
            )
            fused_scale.append({
                "run_id": metadata["run_id"],
                "k": config["k"],
                "seconds": result["wall_ns"] / 1e9,
                "steps": result["steps"],
                "shared_answers": len(shared),
                "answer_flips_from_k1": sum(
                    k1_2000["answers"][key] != result["answers"][key]
                    for key in shared
                ),
                "survivors": len(result["survivors"]),
            })
    fused_scale.sort(key=lambda row: row["k"])

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
        "headline_results": headlines,
        "fused_scale_validation": fused_scale,
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
        f"- Median runtime overhead above model-runner time: {100 * gate['median_runtime_overhead_fraction']:.2f}%" if gate["median_runtime_overhead_fraction"] is not None else "- No overhead measurement",
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
        "## Headline scale results",
        "",
        "| Documents | Custom k=1 | Stock vLLM | Stock / custom | Answers | Flips |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["headline_results"]:
        lines.append(
            f"| {row['n_docs']} | {row['custom_seconds']:.2f} s | "
            f"{row['stock_seconds']:.2f} s | "
            f"{row['stock_to_custom_speedup']:.3f} | "
            f"{row['answer_count']} | {row['answer_flips']} |"
        )
    lines.extend([
        "",
        "Model-runner time is the measured lower reference for the same chosen batches. It is not a hardware lower bound over all possible schedules.",
        "",
        "## Fixed-work fused k",
        "",
        "Each row executes all four filters. These are one-document operator checks.",
        "The verified fused implementation runs one prefix group per forward. Multi-group full-model runs are excluded because they changed Boolean answers.",
        "",
        "| Target document tokens | k | Time | Steps | Accuracy | Flips from k=1 |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["fixed_work_shapes"]:
        lines.append(
            f"| {row['document_tokens']} | {row['k']} | "
            f"{row['seconds']:.4f} s | {row['steps']} | "
            f"{100 * row['accuracy']:.2f}% | "
            f"{row['answer_flips_from_k1']} |"
        )
    fused_flips = sum(
        row["answer_flips_from_k1"]
        for row in summary["fixed_work_shapes"]
        if row["k"] > 1
    )
    if fused_flips:
        lines.extend([
            "",
            f"Fused k changed {fused_flips} fixed-work answers. It does not pass the semantic shipping gate.",
        ])
    if summary["fused_scale_validation"]:
        lines.extend([
            "",
            "### Fused 2,000-document semantic check",
            "",
            "| k | Time | Steps | Shared answers | Flips from k=1 | Survivors |",
            "|---:|---:|---:|---:|---:|---:|",
        ])
        for row in summary["fused_scale_validation"]:
            lines.append(
                f"| {row['k']} | {row['seconds']:.2f} s | "
                f"{row['steps']} | {row['shared_answers']} | "
                f"{row['answer_flips_from_k1']} | "
                f"{row['survivors']} |"
            )
        lines.extend([
            "",
            "These verified one-prefix-group runs fail semantics and performance. Fused FP8 cascade must not ship.",
        ])
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
