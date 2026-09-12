"""Write Modal run metadata and the flat measurements table."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_json(path, data):
    """Replace a JSON file only after its contents have been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def measurement_rows(method: str, report: dict) -> list[dict]:
    """Return one flat row per completed query of a run.json report."""
    rows = []
    for item in report["queries"]:
        if item.get("status") != "complete":
            continue
        measured = item["measurements"]
        metrics = item["metrics"]
        answers = metrics["accuracy"]["answer_accuracy"]
        output = metrics["accuracy"]["output_accuracy"]
        rows.append({
            "query": item["id"],
            "method": method,
            "wall_s": float(measured["wall_s"]),
            "fresh_tokens": int(measured["fresh_tokens"]),
            "minimum_tokens": measured.get("minimum_tokens"),
            "regret_tokens": measured.get("regret_tokens"),
            "evaluated_document_pairs": metrics.get("evaluated_document_pairs"),
            "input_rows": int(sum(metrics["input_rows"].values())),
            "answers_evaluated": int(answers["evaluated"]),
            "answers_correct": int(answers["correct"]),
            "predicted_rows": int(output["predicted_rows"]),
            "expected_rows": int(output["expected_rows"]),
            "matching_rows": int(output["matching_rows"]),
            "cost_usd": metrics.get("cost_usd"),
        })
    return rows


MEASUREMENT_SCHEMA = pa.schema([
    ("query", pa.string()),
    ("method", pa.string()),
    ("wall_s", pa.float64()),
    ("fresh_tokens", pa.int64()),
    ("minimum_tokens", pa.int64()),
    ("regret_tokens", pa.int64()),
    ("evaluated_document_pairs", pa.int64()),
    ("input_rows", pa.int64()),
    ("answers_evaluated", pa.int64()),
    ("answers_correct", pa.int64()),
    ("predicted_rows", pa.int64()),
    ("expected_rows", pa.int64()),
    ("matching_rows", pa.int64()),
    ("cost_usd", pa.float64()),
])


def write_measurements(path, rows: list[dict]) -> pa.Table:
    """Write the flat measurements table of a run directory."""
    table = pa.Table.from_pylist(rows, schema=MEASUREMENT_SCHEMA)
    pq.write_table(table, Path(path), compression="zstd")
    return table
