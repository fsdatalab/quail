#!/usr/bin/env python3
"""Build measured attention-shape tables from immutable Modal runs."""

import argparse
import gzip
import json
from pathlib import Path

from docengine.runtime.batch_cost import (
    AttentionShapePoint,
    AttentionShapeTable,
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
    cascade = AttentionShapeTable([
        AttentionShapePoint(
            groups=row["groups"],
            k=row["k"],
            prefix_tokens=row["prefix_tokens"],
            tail_tokens=row["tail_tokens"],
            time_ns=row["cascade_ns"],
        )
        for row in rows
    ])
    standard = AttentionShapeTable([
        AttentionShapePoint(
            groups=row["groups"],
            k=row["k"],
            prefix_tokens=row["prefix_tokens"],
            tail_tokens=row["tail_tokens"],
            time_ns=row["standard_ns"],
        )
        for row in rows
    ])
    return {
        "schema_version": 1,
        "kv_cache_dtype": "fp8",
        "cascade_attention": cascade.to_rows(),
        "standard_attention": standard.to_rows(),
        "comparisons": rows,
        "validation": {
            "status": "needs-held-out-runs",
            "reason": "The first grid supplies calibration points only.",
        },
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
