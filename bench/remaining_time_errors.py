"""Compare the time-left estimates of a saved QUAIL-B run with its actual times.

Pull a run's records from the results volume, then pass the directory:

    modal volume get quail-results benchmarks/quailb/<run_id> runs/<run_id>
    uv run python -m bench.remaining_time_errors runs/<run_id>

For each query, prints the planner's SoL seconds, the measured
execution seconds, their ratio, each operator group's slowdown, the
first estimate made, and the error of the predicted finish at 10%,
25%, and 50% of the execution. Then, for every estimate made during
the run, prints the seconds left it gave beside the seconds that were
actually left. Runs on the CPU from the saved
``remaining_time`` measurement.
"""

import json
import sys
from pathlib import Path

from quail.execution.remaining import FRACTIONS, prediction_errors


def query_rows(directory: Path) -> list[dict]:
    """Return one row per query whose run record holds a remaining_time report."""
    rows = {}
    for path in sorted(directory.rglob("run.json")):
        record = json.loads(path.read_text())
        for item in record.get("queries", []):
            report = (item.get("measurements") or {}).get("remaining_time")
            if not report:
                continue
            first = report["history"][0] if report["history"] else None
            # a run's combined record repeats its families' queries
            rows[item["id"]] = {
                "query": item["id"],
                "runtime_s": item.get("runtime_s"),
                "planned_seconds": report["planned_seconds"],
                "finished_seconds": report["finished_seconds"],
                "num_estimates": len(report["history"]),
                "first_estimate": first,
                "errors": prediction_errors(report),
                "groups": report["groups"],
                "history": report["history"],
            }
    return list(rows.values())


def _percent(value) -> str:
    return "x" if value is None else f"{value * 100:+.0f}%"


def main(directory: str) -> None:
    """Print the comparison table for the run records under directory."""
    rows = query_rows(Path(directory))
    header = ["query", "SoL s", "measured s", "vs. SoL", "estimates",
              "first: at s, left s"] + [f"error at {f:.0%}" for f in FRACTIONS]
    print(" | ".join(header))
    print(" | ".join("---" for _ in header))
    for row in rows:
        planned, finished = row["planned_seconds"], row["finished_seconds"]
        first = row["first_estimate"]
        print(" | ".join([
            row["query"], f"{planned:.2f}", f"{finished:.1f}",
            f"{finished / planned:.1f}x" if planned else "x",
            str(row["num_estimates"]),
            f"{first[0]:.0f}, {first[1]:.0f}" if first else "x",
            *(_percent(row["errors"][f]) for f in FRACTIONS)]))
    print()
    print("query | nodes | kind | SoL s | measured s | slowdown")
    print("--- | --- | --- | --- | --- | ---")
    for row in rows:
        for group in row["groups"]:
            slowdown = group["slowdown"]
            print(" | ".join([
                row["query"], ", ".join(group["node_ids"]), group["kind"],
                f"{group['planned_seconds']:.2f}",
                f"{group['measured_seconds']:.1f}",
                "x" if slowdown is None else f"{slowdown:.1f}x"]))
    print()
    print("query | at s | estimated left s | actual left s | error s | error")
    print("--- | --- | --- | --- | --- | ---")
    for row in rows:
        for at, left in row["history"]:
            actual = row["finished_seconds"] - at
            if actual < 0.5:
                continue        # the last count lands as the run finishes
            print(" | ".join([
                row["query"], f"{at:.1f}", f"{left:.1f}", f"{actual:.1f}",
                f"{left - actual:+.1f}", _percent((left - actual) / actual)]))


if __name__ == "__main__":
    main(sys.argv[1])
