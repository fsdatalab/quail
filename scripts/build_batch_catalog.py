#!/usr/bin/env python3
"""Build measured attention-shape tables from immutable Modal runs."""

import argparse
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docengine.runtime.batch_cost import (
    AttentionShapePoint,
    AttentionShapeTable,
    ValidationRow,
    summarize_validation,
)


def load_kernel_rows(root: Path) -> list[dict]:
    rows = []
    for metadata_path in sorted(root.glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("phase") != "cascade-kernel":
            continue
        result_path = metadata_path.parent / "result.json.gz"
        with gzip.open(result_path, "rt") as handle:
            result = json.load(handle)
        rows.append({
            "run_id": metadata["run_id"],
            "groups": result["groups"],
            "k": result["k"],
            "prefix_tokens": result["prefix_tokens"],
            "tail_tokens": result["tail_tokens"],
            "cascade_ns": round(result["cascade_ms"] * 1e6),
            "standard_ns": round(result["standard_ms"] * 1e6),
            "speedup": result["speedup"],
            "max_absolute_difference": result["max_absolute_difference"],
        })
    return rows


def build_catalog(rows: list[dict]) -> dict:
    calibration = [
        row for row in rows
        if row["tail_tokens"] not in {24, 40}
    ]
    held_out = [
        row for row in rows
        if row not in calibration
    ]
    cascade = AttentionShapeTable([
        AttentionShapePoint(
            groups=row["groups"],
            k=row["k"],
            prefix_tokens=row["prefix_tokens"],
            tail_tokens=row["tail_tokens"],
            time_ns=row["cascade_ns"],
        )
        for row in calibration
    ])
    standard = AttentionShapeTable([
        AttentionShapePoint(
            groups=row["groups"],
            k=row["k"],
            prefix_tokens=row["prefix_tokens"],
            tail_tokens=row["tail_tokens"],
            time_ns=row["standard_ns"],
        )
        for row in calibration
    ])
    validation_rows = []
    for row in held_out:
        predicted = cascade.estimate_ns(
            groups=row["groups"],
            k=row["k"],
            prefix_tokens=row["prefix_tokens"],
            tail_tokens=row["tail_tokens"],
        )
        validation_rows.append({
            "run_id": row["run_id"],
            "predicted_ns": predicted,
            "measured_ns": row["cascade_ns"],
            "absolute_error_fraction": abs(
                predicted - row["cascade_ns"]
            ) / row["cascade_ns"],
        })
    if validation_rows:
        summary = summarize_validation([
            ValidationRow(
                predicted_ns=row["predicted_ns"],
                measured_ns=row["measured_ns"],
                label=row["run_id"],
            )
            for row in validation_rows
        ])
        validation = {
            "status": "passed" if summary.passes else "failed",
            "count": summary.count,
            "median_absolute_error": summary.median_absolute_error,
            "p95_absolute_error": summary.p95_absolute_error,
            "mean_signed_error": summary.mean_signed_error,
            "rows": validation_rows,
        }
    else:
        validation = {
            "status": "needs-held-out-runs",
            "reason": "No non-calibration shapes were found.",
        }
    return {
        "schema_version": 1,
        "kv_cache_dtype": "fp8",
        "cascade_attention": cascade.to_rows(),
        "standard_attention": standard.to_rows(),
        "comparisons": calibration,
        "held_out_comparisons": held_out,
        "validation": validation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/runs")
    parser.add_argument("--out", default="results/BATCH_COST_CATALOG.json")
    args = parser.parse_args()
    rows = load_kernel_rows(Path(args.root))
    if not rows:
        raise SystemExit("no cascade-kernel runs found")
    Path(args.out).write_text(json.dumps(
        build_catalog(rows),
        indent=2,
        sort_keys=True,
    ))
    print(f"cataloged {len(rows)} kernel runs")


if __name__ == "__main__":
    main()
