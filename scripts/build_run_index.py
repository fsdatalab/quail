#!/usr/bin/env python3
"""Build a readable index from immutable run directories."""

import argparse
import gzip
import json
from pathlib import Path


def load_runs(root: Path) -> list[dict]:
    rows = []
    for metadata_path in sorted(root.glob("*/metadata.json")):
        directory = metadata_path.parent
        result_path = directory / "result.json.gz"
        if not result_path.exists():
            result_path = directory / "result.json"
        if not result_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        opener = gzip.open if result_path.suffix == ".gz" else open
        with opener(result_path, "rt", encoding="utf-8") as handle:
            result = json.load(handle)
        config = metadata.get("config", {})
        rows.append({
            "run_id": metadata["run_id"],
            "created_at_utc": metadata["created_at_utc"],
            "git_commit": metadata["git_commit"],
            "phase": metadata["phase"],
            "n_docs": config.get("n_docs", result.get("n_docs")),
            "n_filters": config.get("n_filters", result.get("n_filters")),
            "k": config.get("k", result.get("k")),
            "document_tokens": config.get(
                "document_tokens",
                result.get("target_document_tokens"),
            ),
            "short_circuit": config.get(
                "short_circuit",
                result.get("short_circuit"),
            ),
            "wall_ns": result.get("wall_ns"),
            "steps": result.get("steps"),
            "accuracy": result.get("accuracy"),
            "ground_truth_used_by_runtime": result.get(
                "ground_truth_used_by_runtime"
            ),
            "artifact": str(result_path),
        })
    return rows


def markdown(rows: list[dict]) -> str:
    lines = [
        "# Experiment runs",
        "",
        "Each row is one immutable run. Ground truth is used only by the evaluator.",
        "",
        "| Run | Phase | Documents | Filters | k | Target document tokens | Time | Steps | Accuracy | Commit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        seconds = (
            f"{row['wall_ns'] / 1e9:.4f} s"
            if row["wall_ns"] is not None else ""
        )
        accuracy = (
            f"{100 * row['accuracy']:.2f}%"
            if row["accuracy"] is not None else ""
        )
        lines.append(
            f"| `{row['run_id']}` | {row['phase']} | "
            f"{row['n_docs'] or ''} | {row['n_filters'] or ''} | "
            f"{row['k'] or ''} | {row['document_tokens'] or ''} | "
            f"{seconds} | {row['steps'] or ''} | {accuracy} | "
            f"`{row['git_commit'][:8]}` |"
        )
    lines.extend([
        "",
        "## Interpretation rules",
        "",
        "1. A timing is a performance result only when the run excludes model loading and kernel warmup.",
        "2. Runs with one shared-prefix group per forward are correctness checks, not throughput results.",
        "3. Accuracy against synthetic labels measures the model and prompt as well as execution.",
        "4. Compare physical implementations only when they execute the same required filter work.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/runs")
    parser.add_argument("--json", default="results/RUN_INDEX.json")
    parser.add_argument("--markdown", default="notes/EXPERIMENTS.md")
    args = parser.parse_args()
    rows = load_runs(Path(args.root))
    Path(args.json).write_text(json.dumps(rows, indent=2, sort_keys=True))
    Path(args.markdown).write_text(markdown(rows))
    print(f"indexed {len(rows)} runs")


if __name__ == "__main__":
    main()
